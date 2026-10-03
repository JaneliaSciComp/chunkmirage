# REST API

All endpoints are CORS-open. Editing endpoints can be disabled with
`create_app(..., allow_edit=False)`. A server given a token (`--token`, `CHUNKMIRAGE_TOKEN`,
`create_app(..., token=...)`) answers `/api/*` only with it, as `Authorization: Bearer
<token>` or `?token=<token>` (for `/api/events`: a browser's `EventSource` sends no
headers), and 401 otherwise. The datasets themselves and `/` stay open, since viewers send
no headers.

| method   | path                                   | purpose |
| -------- | -------------------------------------- | ------- |
| `GET`    | `/`                                    | index: datasets, their specs, source URLs per format, `viewer_url` of an attached python viewer (or null), cache stats |
| `GET`    | `/api/ops`                             | registered ops with halo, cache flag, docstring and JSON schema |
| `GET`    | `/api/datasets`                        | dataset names |
| `POST`   | `/api/datasets`                        | create: body `{"name": ..., "spec": PipelineSpec}` (or the spec with a `name` field) |
| `GET`    | `/api/datasets/{name}`                 | spec, digest, per-level shape/chunks/dtype/voxel size, source URLs |
| `PUT`    | `/api/datasets/{name}`                 | replace the pipeline live; body is a `PipelineSpec`; returns new digest and URLs |
| `DELETE` | `/api/datasets/{name}`                 | remove |
| `POST`   | `/api/datasets/{name}/refresh`         | rebuild the pipeline so its ops' `cache_token` is read again (weights or files they depend on changed); `{"changed", "digest", "sources"}`, and a `change` event if the digest moved ([caching](../concepts/caching.md#state-outside-the-parameters)) |
| `GET`    | `/api/datasets/{name}/neuroglancer`    | `?format=n5|zarr|zarr3|precomputed&viewer=...` → `{"source", "url"}` |
| `GET`    | `/api/neuroglancer`                    | same query; one viewer state with a layer per dataset → `{"state", "url", "sources"}` |
| `GET`    | `/api/events`                          | Server-Sent Events; `change` event on start and after every edit, with digests and source URLs |
| `GET`    | `/ui`                                  | built-in control page; see [Interactivity](../concepts/interactivity.md) |
| `GET`    | `/api/cache`                           | cache stats |
| `DELETE` | `/api/cache`                           | clear cache |
| `GET`    | `/api/queue`                           | work waiting and running: chunk requests (`requests`) and each queue of expensive work (`refined blocks`, and `op <name>` for each op with `slots`), per level, with what was dropped because its clients left ([caching](../concepts/caching.md#order-of-work-and-requests-given-up-on)) |
| `GET`    | `/{name}/{format}/{path}`              | the spoofed dataset; see [Formats](../concepts/formats.md) |
| `GET`    | `/{name}/@{digest}/{format}/{path}`    | same, with a cache-busting token |

## PipelineSpec

```json
{
  "source": "/path/or/url",          // zarr/n5/precomputed array or multiscale group, or file.h5::/dataset
  "ops": [{"op": "threshold", "low": 120}],
  "chunk_shape": [64, 64, 64],       // optional; default: source chunk shape
  "cache_source": true,              // cache raw chunks (stage 0)
  "voxel_size": [8, 8, 8],           // optional overrides of what the source reports
  "units": ["nm", "nm", "nm"],
  "axes": ["z", "y", "x"],
  "translation": [0, 0, 0],
  "kind": "image",                   // optional: image, label or mask, over what the source guesses
  "padding": "edge"                  // what ops see past the volume's edge: edge (repeated) or zero
}
```

Responses to `POST`, `PUT` and `GET /api/datasets/{name}` include `digest`, `source_dtype`,
per-level `levels` (shape, chunks, dtype, voxel size, units, axes, and `kind`: `image`, `label`, `mask` or `null`), `ops_info` (per op: name, per-axis halo, docstring), and `sources`, a map
from format to Neuroglancer source URL carrying the new digest.

## Datasets resolved by name

A dataset can also exist before anyone registers it. Give the registry a resolver, a
function from a name to a `PipelineSpec` (or a dict of one, or a `Pipeline`), or `None` for
a name it does not know: `DatasetRegistry(resolver=...)`, `create_app(..., resolver=...)`,
or `chunkmirage serve ... --resolver module:function`. The first request for an
unregistered name, whether a viewer's chunk or metadata or `GET /api/datasets/{name}`, asks
the resolver. What it returns is built, registered under that name (so `/api/events` reports
it), and served from then on like any other dataset. Concurrent requests for a new name
build it once. A resolver raising `ValueError` answers 400 with its message, anything else
500.

This makes links that carry their pipeline in the name, decoded by the resolver, work on a
fresh server with no setup step:

```python
def resolve(name):
    if not name.startswith("thr-"):
        return None
    return {"source": "s3://bucket/em.zarr", "ops": [{"op": "threshold", "low": int(name[4:])}]}
```

The token does not guard dataset paths (viewers send no headers), so a resolver is reachable
by anyone who can reach the server. Build only what you would serve to them, and nothing
that names an arbitrary file or URL from the request.

## Routes of your own

A package can add endpoints next to these: a plugin's controls, say. Pass Starlette routes
to `create_app(..., extra_routes=[...])`, or declare a `chunkmirage.routes` entry point
naming a function that takes the app's `DatasetRegistry` and returns routes:

```toml
[project.entry-points."chunkmirage.routes"]
myplugin = "myplugin.server:routes"
```

```python
from starlette.responses import JSONResponse
from starlette.routing import Route

def routes(registry):
    async def names(request):
        return JSONResponse(registry.names())
    return [Route("/api/myplugin/names", names)]
```

Installed plugins' routes are added to every app, `chunkmirage serve`'s included
(`create_app(..., route_plugins=False)` leaves them out). One that fails to load is skipped
with a warning. They are matched after the built-in routes and before the datasets, so a
route of your own wins over a dataset of the same name. Put them under `/api/` to have them
guarded by the token too.

## Mounted in another app

The app works as a sub-app, `Mount("/prefix", create_app(...))` in your own Starlette or
FastAPI app: every path above is then under `/prefix`, the token guards `/prefix/api/*`,
links carry the prefix, and the control page calls the API relative to where it is served.
A mounted app gets no lifespan events, so the threadpool is sized by the first chunk
request instead (it is the host's pool too, and only ever made larger).
