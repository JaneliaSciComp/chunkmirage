# Interactivity: changing pipelines while you look

The point of chunkmirage is to *see* the effect of a change immediately. That needs two
things: a way to change the pipeline, and a way to make the viewer refetch.

## Changing the pipeline

| how                     | where                                         | notes |
| ----------------------- | --------------------------------------------- | ----- |
| control page            | `http://localhost:8000/ui`                    | sliders and fields generated from each op's JSON schema; add/remove ops; drives a Neuroglancer window |
| REST                    | `PUT /api/datasets/{name}`                    | any language, `curl`, notebooks, agents |
| Python                  | `registry.add(name, spec)` / `Viewer.set_ops` | in-process, e.g. from a notebook running the server in a thread |

Every edit bumps the registry version and fires `GET /api/events` (Server-Sent Events), so
several control surfaces stay in sync. Only the changed stage and its dependents recompute;
see [Caching](caching.md).

## Making the viewer refetch

Neuroglancer (and most viewers) cache chunks by URL and expose no "invalidate" call. So
chunkmirage embeds the pipeline digest in every source URL:

```
zarr3://http://localhost:8000/thr/@75f296149b33/zarr3
```

A changed pipeline is a changed URL, and a changed URL is a cache miss. Three ways to get
the new URL into the viewer, from least to most convenient:

1. **Paste it.** `PUT` responses and `/api/events` carry the new `sources`. Paste into the
   layer's source field. Camera preserved; manual.
2. **Let the control page drive an external window.** `/ui` opens Neuroglancer in a new
   window (the hosted `neuroglancer-demo.appspot.com` by default) and, on every change,
   navigates that window to a fresh state URL. Cross-origin navigation is allowed, so this
   works with appspot. But cross-origin *reading* is not, so the current camera cannot be
   copied into the new state: **the view resets on each push**. Fine for demos, tiresome
   for real work.
3. **Use the Python viewer.** `chunkmirage serve … --python-viewer` (or
   `chunkmirage.viewer.Viewer` in code) starts a
   [python-neuroglancer](https://github.com/google/neuroglancer/tree/master/python) viewer
   whose layers subscribe to the registry. An edit swaps only that layer's source URL inside
   a state transaction; Neuroglancer refetches that layer and **keeps camera, other layers
   and shader settings**. With `--ng-client appspot` the client code is fetched from the
   hosted demo site (the Python server proxies it), so you are running the same build as
   appspot, just with state sync. This is the recommended interactive mode.

## Transforms on the fly

Two cases, and they belong in different places:

* **Affine** (translate, rotate, scale, shear): Neuroglancer applies a per-layer affine
  transform on the client, editable live in the layer's *Source* tab, with no refetch and
  no server. Do not route these through chunkmirage.
* **Non-affine** (displacement fields, piecewise or non-rigid registration, resampling into
  another dataset's grid): the client cannot do these. They are a chunkmirage op that reads
  the field and resamples with a halo. Changing the field or its parameters changes the
  digest and the viewer refetches as above. The `Resample` op is on the [roadmap](../roadmap.md).

## For your own tooling

`GET /api/events` emits an initial `change` event and one per edit:

```
event: change
data: {"version": 7, "datasets": {"thr": {"digest": "75f2…", "sources": {"zarr3": "zarr3://…", …}}}}
```

Subscribe from JavaScript with `EventSource`, or from Python with `httpx` streaming. Fetch
`GET /api/neuroglancer?format=zarr3` for a complete viewer state containing every dataset.
