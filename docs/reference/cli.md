# CLI

```
chunkmirage serve SOURCE [options]
chunkmirage ops
chunkmirage inspect SOURCE
chunkmirage schema [--out FILE]
```

`SOURCE` is anything chunkmirage reads: a stored array or multiscale group (zarr, N5,
precomputed; local, `s3://`, `gs://`, `http(s)://`), `file.h5::/dataset`, or a computed
`synthetic://`, `scene://`, `warp://`, `register://` or `stitch://` URL
([sources](../concepts/formats.md#sources)).

## `serve`

| option                | default                                   | meaning |
| --------------------- | ----------------------------------------- | ------- |
| `--name`              | `processed`                               | dataset name in URLs |
| `--op`, `-o`          |                                           | op spec, repeatable; `name:k=v,k=v` or JSON |
| `--raw` / `--no-raw`  | on                                        | also serve the unprocessed source as `raw`; shares the cache, appears as a second layer |
| `--chunk`             | source chunks                             | output chunk shape, e.g. `64,64,64` |
| `--select`            | none                                      | pin non-spatial axes, e.g. `c=1,t=0`: the pipeline sees that channel of that time point as a `z, y, x` volume (the spec's `select`) |
| `--mesh`              | none                                      | what the [mesh frontend](../concepts/formats.md#meshes-computed-when-fetched) meshes (the spec's `mesh`): `kind=surface,threshold=255` (an isosurface), `kind=terrain,exaggeration=2` (an elevation model), `level=N`, `lods=4` (levels of detail: finer meshes where the viewer zooms in); `''` for the defaults. The python viewer then shows the mesh too |
| `--host` / `--port`   | `0.0.0.0` / `8000`                        | bind address; without `--port`, the first free port from 8000 up |
| `--https`             | off                                       | serve https; a self-signed certificate is generated in `~/.cache/chunkmirage/` on first use (needs the `https` extra or the `openssl` CLI) |
| `--cert` / `--key`    | auto-generated                            | use your own certificate and key with `--https` |
| `--public-url`        | `http(s)://<lan-ip>:PORT`                 | address clients use in every printed link and layer URL; defaults to this machine's network address when binding `0.0.0.0`, `localhost` when binding `127.0.0.1`; set explicitly behind a tunnel or proxy |
| `--cache-gb`          | `2.0`                                     | in-process chunk cache |
| `--source-cache-gb`   | `0.5`                                     | tensorstore's cache of decoded source chunks, one pool shared by every source the server reads |
| `--viewer`            | `https://neuroglancer-demo.appspot.com`   | viewer for the printed link |
| `--format`            | `zarr3`                                   | format used in the printed link |
| `--threads`           | `2 × CPUs` (min 40)                       | chunk requests computing at once; numpy/scipy/tensorstore release the GIL so this is the effective parallelism. The thread pool itself is larger, so requests waiting on queued work (GPU fits) hold no slot ([caching](../concepts/caching.md#order-of-work-and-requests-given-up-on)) |
| `--workers`           | `1`                                       | uvicorn worker processes; each has its own registry and cache, so live edits reach only one: fixed pipelines only |
| `--server`            | `uvicorn`                                 | `uvicorn` is HTTP/1.1; `hypercorn` adds HTTP/2 over https (lifts the browser's 6-connections-per-host limit) but is experimental: check that chunks load in your browser |
| `--python-viewer`     | off                                       | also start a python-neuroglancer viewer whose layers follow live edits (needs the `viewer` extra) |
| `--ng-client`         | `bundled`                                 | client build for the python viewer: `bundled`, `appspot`, or a URL |
| `--viewer-host`       | same as `--host`                          | bind address of the python viewer; `0.0.0.0` lets other machines open it |
| `--viewer-port`       | random                                    | fixed port for the python viewer (handy for port forwarding) |

`serve` prints the source URL, an appspot link, the control page (`/ui`), the control API
URL and, with `--python-viewer`, the viewer URL.

## `ops`

Lists registered ops with halo, cache flag and parameters.

## `inspect`

Prints shape, chunks, dtype, voxel size, units and axes per scale level as chunkmirage sees
them, which is useful for checking metadata detection before serving.

## `schema`

Prints the JSON Schema of what chunkmirage exchanges: `PipelineSpec`, each registered op's
parameters (tagged with its `op` name, and their union `OpSpec`), and `RegisterParams`, the
query of a [`register://`](../concepts/formats.md#register-sources-deformable-registration-on-a-gpu)
URL. `--out FILE` writes it instead. The browser engine (`web/`) generates its TypeScript
types and form defaults from the copy in `web/src/generated/`, so the two engines share one
definition; `tests/test_schema.py` keeps that copy current, and CI checks the generated types.
