# Interactivity: changing pipelines while you look

The point of chunkmirage is to *see* the effect of a change immediately. That needs two
things: a way to change the pipeline, and a way to make the viewer refetch.

## Changing the pipeline

| how                     | where                                         | notes |
| ----------------------- | --------------------------------------------- | ----- |
| control page            | `http://localhost:8000/ui`                    | sliders and fields generated from each op's JSON schema; add/remove ops; Neuroglancer embedded in the page or opened in a new tab |
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
2. **Let the control page drive a viewer.** `/ui` embeds Neuroglancer in the page (the
   hosted `neuroglancer-demo.appspot.com` allows framing) or opens it in a new tab, and on
   every change navigates it to a fresh state URL. Cross-origin navigation is allowed, so
   this works with appspot. But cross-origin *reading* is not, so the current camera cannot
   be copied into the new state: **the view resets on each push**. Fine for demos, tiresome
   for real work.
3. **Use the Python viewer.** `chunkmirage serve … --python-viewer` (or
   `chunkmirage.viewer.Viewer` in code) starts a
   [python-neuroglancer](https://github.com/google/neuroglancer/tree/master/python) viewer
   whose layers subscribe to the registry. An edit swaps only that layer's source URL inside
   a state transaction; Neuroglancer refetches that layer and **keeps camera, other layers
   and shader settings**. With `--ng-client appspot` the client code is fetched from the
   hosted demo site (the Python server proxies it), so you are running the same build as
   appspot, just with state sync. This is the recommended interactive mode. The control
   page detects a running python viewer (the server reports it as `viewer_url` in `GET /`)
   and embeds it by default, so sliders and viewer sit on one page with the camera
   preserved. The viewer's dimensions come from the first dataset by name, or from
   `Viewer.set_dimensions(name)`: spatial axes first, then the rest (time last), with x, y
   and z displayed. `Viewer.rename_dimensions(name, {"c'": "c^"})` renames a dataset's
   dimensions in its layer (here, the channel axis becomes a shader channel); the rename
   is rebuilt from the dataset's axes on every edit. A source that says how to show its
   channels (a `MultiscaleSource` with a `shader`, as `register://`'s `show=pair` and
   `show=field` are) gets that shader and the rename by itself, here and in the printed
   link. `Viewer.hosted_link()` gives the
   current state as an appspot link: a snapshot that does not follow later edits but
   needs no python server.

**Animating: a time axis, not new URLs.** A new URL is a cache miss for everything that
layer has on screen, so it goes blank until the new chunks arrive; stepping a parameter
faster than a screenful computes never shows a finished frame. For a parameter you want
to sweep, serve the sweep as a `t` axis instead (as
[`warp://…&frames=N`](formats.md#warp-sources-procedural-deformations) does, and
`register://…&frames=N` for the steps of a solve) and play it
with Neuroglancer's playback (click the dimension in the top bar, or set `velocity` in the
state). Nothing is invalidated: each frame is fetched once, then comes from the browser's
and the server's caches.

Two browser rules bite when the server is on a private network address (`10.x`,
`192.168.x`) and the viewer is a public https site such as appspot:

* **Local / Private Network Access** (Chrome): a public site needs permission to reach a
  private address. Loopback is exempt, which is why `localhost` always works. The server
  answers the preflight with `Access-Control-Allow-Private-Network: true`, and the control
  page delegates the permission to the embedded viewer with
  `allow="local-network-access"`. Chrome may still show a one-time permission prompt; accept
  it. Firefox does not enforce this yet.
* **Certificate trust is per origin.** Accepting the self-signed certificate at
  `https://localhost:8000` does not cover `https://10.101.10.98:8000`. Open the exact host
  the chunk URLs use, once, in each browser.

Mixed-content rules decide what can be embedded where: an `https` control page (with
`--https`) cannot embed the `http` python viewer, and an `http` control page on a
non-localhost address cannot have the embedded `https` appspot viewer fetch its chunks. The
page explains which case you are in and falls back to "Open in new tab".

## Sharing on your network

`chunkmirage serve` binds to all interfaces by default and prints URLs using the machine's
network address, so anyone on the same network can open the control page and the python
viewer:

```
control UI:    http://10.123.4.56:8000/ui
python viewer: http://10.123.4.56:41595/v/<token>/
```

The python viewer works over plain `http` from anywhere on the network because it serves
its own Neuroglancer page from the same machine as the chunks.

The hosted appspot viewer is `https` and browsers block it from fetching
`http://10.x.x.x` ("mixed content"; only `localhost` is exempt). To use appspot from other
machines, start the server with `--https`. chunkmirage generates a self-signed certificate
whose subject alternative names cover the machine's IP, hostname and `localhost`. Each
browser has to trust it once: open the printed `https://…/` URL, accept the warning, then
open the viewer link. Neuroglancer's chunk fetches will otherwise fail silently.

The examples (`examples/swirl_demo.py`, `examples/fly_brain_registration.py`) do this by
default, so the links they print can be sent to anyone on the network: they bind every
interface, use the machine's IP, and serve https with that certificate. `--host 127.0.0.1`
keeps one local (plain http, `localhost` links); `--no-https` keeps plain http on the
network, where only the python viewer link works from other machines.

Edits are global: everyone viewing dataset `thresh` sees the same pipeline, and a slider
drag by one person changes it for all. For independent exploration, create a copy under
another name (`POST /api/datasets` with the same spec); upstream stages are shared through
the cache, so copies are nearly free.

## Transforms on the fly

Two cases, and they belong in different places:

* **Affine** (translate, rotate, scale, shear): Neuroglancer applies a per-layer affine
  transform on the client, editable live in the layer's *Source* tab, with no refetch and
  no server. Do not route these through chunkmirage.
* **Non-affine** (displacement fields, non-rigid registration, resampling into another
  dataset's grid): the client cannot do these. A
  [`scene://` source](formats.md#scene-sources-ome-zarr-06-transformations) resamples an
  image through OME-Zarr 0.6 transformations on the server, chunk by chunk. Editing the
  scene's transformations and `PUT`ting the dataset again changes the digest, and the
  viewer refetches as above. For a deformation you can change continuously, a
  [`warp://` source](formats.md#warp-sources-procedural-deformations) computes a swirl
  from coordinates, over a `t` axis of frames if you like; `examples/swirl_demo.py` plays
  it in the python viewer with its displacement field shown as an RGB layer. To have the
  field solved rather than given, a
  [`register://` source](formats.md#register-sources-deformable-registration-on-a-gpu)
  fits it on a GPU when it opens, in seconds; `PUT` the URL with another `smooth` or
  `grid` and the viewer shows the new registration moments later.
  `examples/register_demo.py` does this from a terminal prompt and compares fixed and
  registered in a shader on the viewer's GPU, with no refetch. Landmarks edited live in
  the viewer are on the [roadmap](../roadmap.md).

## For your own tooling

`GET /api/events` emits an initial `change` event and one per edit:

```
event: change
data: {"version": 7, "datasets": {"thr": {"digest": "75f2…", "sources": {"zarr3": "zarr3://…", …}}}}
```

Subscribe from JavaScript with `EventSource`, or from Python with `httpx` streaming. Fetch
`GET /api/neuroglancer?format=zarr3` for a complete viewer state containing every dataset.
