# CLI

```
chunkmirage serve SOURCE [options]
chunkmirage ops
chunkmirage inspect SOURCE
chunkmirage schema [--out FILE]
```

`SOURCE` is anything chunkmirage reads: a stored array or multiscale group (zarr, N5,
precomputed; local, `s3://`, `gs://`, `http(s)://`), `file.h5::/dataset`, or a computed
`synthetic://`, `scene://`, `warp://`, `register://`, `stitch://`, `stack://` or `flip://`
URL ([sources](../concepts/formats.md#sources)).

## `serve`

| option                | default                                   | meaning |
| --------------------- | ----------------------------------------- | ------- |
| `--name`              | `processed`                               | dataset name in URLs |
| `--op`, `-o`          |                                           | op spec, repeatable; `name:k=v,k=v` or JSON; a [`python` op](ops.md#your-own-python-the-python-op)'s `code=@file.py` (either form) is that file's text |
| `--raw` / `--no-raw`  | on                                        | also serve the unprocessed source as `raw`; shares the cache, appears as a second layer |
| `--chunk`             | source chunks                             | output chunk shape, e.g. `64,64,64` |
| `--select`            | none                                      | pin non-spatial axes, e.g. `c=1,t=0`: the pipeline sees that channel of that time point as a `z, y, x` volume (the spec's `select`) |
| `--mesh`              | none                                      | what the [mesh frontend](../concepts/formats.md#meshes-computed-when-fetched) meshes (the spec's `mesh`): `kind=surface,threshold=255` (an isosurface), `kind=terrain,exaggeration=2` (an elevation model), `level=N`, `lods=4` (levels of detail: finer meshes where the viewer zooms in); `''` for the defaults. The python viewer then shows the mesh too |
| `--host` / `--port`   | `0.0.0.0` / `8000`                        | bind address; without `--port`, the first free port from 8000 up; `--port 0`: any free port. The port is bound before anything is printed, so no other process can take it in between |
| `--ready-file`        | none                                      | once the server accepts connections, write `{"url", "port", "pid", "datasets", "neuroglancer"}` (`datasets`: each dataset's source URL in `--format`) to this file, whole; `-` prints it as one line on stdout. The file is removed when the server exits. For launchers that start a server and wait for its address (a cluster job, a test) |
| `--https`             | off                                       | serve https; a self-signed certificate is generated in `~/.cache/chunkmirage/` on first use (needs the `https` extra or the `openssl` CLI) |
| `--cert` / `--key`    | auto-generated                            | use your own certificate and key with `--https` |
| `--public-url`        | `http(s)://<lan-ip>:PORT`                 | address clients use in every printed link and layer URL; defaults to this machine's network address when binding `0.0.0.0`, `localhost` when binding `127.0.0.1`; set explicitly behind a tunnel or proxy. When binding `0.0.0.0`, the server first tries to reach itself at that address (a second at most): a VPN can refuse it, and then the links use `localhost`, which works on this machine only, and a line says so |
| `--cache-gb`          | `2.0`                                     | in-process chunk cache; `0` caches nothing |
| `--compressor`        | `blosc`                                   | how zarr v2 and v3 chunks are compressed: `blosc` (zstd with byte shuffle), `zstd`, `gzip` or `none`. blosc encodes a 16 MB float32 chunk in 30 ms where gzip takes 900 ms; use `gzip` for a client without blosc (Fiji and BigDataViewer read zarr through n5-zarr, which needs the native Blosc library; their N5 frontend stays gzip). In Python: `create_app(..., frontends=cli.frontends_for("gzip"))`, or the frontends' own `compressor` |
| `--source-cache-gb`   | `0.5`                                     | tensorstore's cache of decoded source chunks, one pool shared by every source the server reads |
| `--viewer`            | `https://neuroglancer-demo.appspot.com`   | viewer for the printed link |
| `--format`            | `zarr3`                                   | format of the printed source URLs and link: `n5`, `zarr`, `zarr3` or `precomputed` (every format is served regardless) |
| `--threads`           | `2 × CPUs` (min 40)                       | chunk requests computing at once; numpy/scipy/tensorstore release the GIL so this is the effective parallelism. The thread pool itself is larger, so requests waiting on queued work (GPU fits) hold no slot ([caching](../concepts/caching.md#order-of-work-and-requests-given-up-on)) |
| `--device`            | `cpu`                                     | where `warp://` and `scene://` resample: `cpu` (numpy and scipy), `cuda` or `cuda:N` (PyTorch, the `gpu` extra), or `auto` (a GPU if PyTorch sees one, else the CPU). On a GPU, chains of affines and sampled fields run there, in single precision (values within a grey level of the CPU's, the same cache keys); any other step keeps a chunk on the CPU ([formats](../concepts/formats.md#warp-sources-an-image-through-a-displacement-field)). In Python: `chunkmirage.gpu.set_device` |
| `--resolver`          | none                                      | `module:function` building the pipeline for a dataset name nobody registered, on its first request ([API](api.md#datasets-resolved-by-name)) |
| `--token`             | `CHUNKMIRAGE_TOKEN`                       | require this token for `/api/*` (`Authorization: Bearer <token>` or `?token=`); datasets stay open; the printed control-page link carries it ([API](api.md)) |
| `--allow-code`        | off                                       | let pipelines sent to `/api` carry [`python` ops](ops.md#your-own-python-the-python-op), code the server runs as you; a token is generated (in the printed control-page link) unless `--token` is given, so code runs only for whoever holds it |
| `--workers`           | `1`                                       | uvicorn worker processes; each has its own registry and cache, so live edits reach only one: fixed pipelines only |
| `--server`            | `uvicorn`                                 | `uvicorn` is HTTP/1.1; `hypercorn` adds HTTP/2 over https (lifts the browser's 6-connections-per-host limit) but is experimental: check that chunks load in your browser |
| `--python-viewer`     | off                                       | also start a python-neuroglancer viewer whose layers follow live edits (needs the `viewer` extra) |
| `--ng-client`         | `bundled`                                 | client build for the python viewer: `bundled`, `appspot`, or a URL |
| `--viewer-host`       | same as `--host`                          | bind address of the python viewer; `0.0.0.0` lets other machines open it |
| `--viewer-port`       | random                                    | fixed port for the python viewer (handy for port forwarding) |

`serve` prints a source URL per dataset (`raw` included), a link to the `--viewer` site,
the control page (`/ui`, with the token when there is one), the control API URL and, with
`--python-viewer`, the viewer URL; the port and thread count when they differ from what
was asked for.

## `ops`

Lists registered ops with halo, cache flag and parameters.

## `inspect`

Prints shape, chunks, dtype, voxel size, units and axes per scale level as chunkmirage sees
them, which is useful for checking metadata detection before serving.

## `schema`

Prints the JSON Schema of what chunkmirage exchanges: `PipelineSpec`, each registered op's
parameters (tagged with its `op` name, and their union `OpSpec`), `RegisterParams`, the
query of a [`register://`](../concepts/formats.md#register-sources-deformable-registration-on-a-gpu)
URL, and `StitchParams`, a [`stitch://`](../concepts/formats.md#stitch-sources-tiles-stitched-by-interest-points-and-ransac) one's. `--out FILE` writes it instead. The browser engine (`web/`) generates its TypeScript
types and form defaults from the copy in `web/src/generated/`, so the two engines share one
definition; `tests/test_schema.py` keeps that copy current, and CI checks the generated types.
