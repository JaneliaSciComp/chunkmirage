# REST API

All endpoints are CORS-open. Editing endpoints can be disabled with
`create_app(..., allow_edit=False)`.

| method   | path                                   | purpose |
| -------- | -------------------------------------- | ------- |
| `GET`    | `/`                                    | index: datasets, their specs, source URLs per format, cache stats |
| `GET`    | `/api/ops`                             | registered ops with halo, cache flag, docstring and JSON schema |
| `GET`    | `/api/datasets`                        | dataset names |
| `POST`   | `/api/datasets`                        | create: body `{"name": ..., "spec": PipelineSpec}` (or the spec with a `name` field) |
| `GET`    | `/api/datasets/{name}`                 | spec, digest, per-level shape/chunks/dtype/voxel size, source URLs |
| `PUT`    | `/api/datasets/{name}`                 | replace the pipeline live; body is a `PipelineSpec`; returns new digest and URLs |
| `DELETE` | `/api/datasets/{name}`                 | remove |
| `GET`    | `/api/datasets/{name}/neuroglancer`    | `?format=n5|zarr|zarr3|precomputed&viewer=...` → `{"source", "url"}` |
| `GET`    | `/api/neuroglancer`                    | same query; one viewer state with a layer per dataset → `{"state", "url", "sources"}` |
| `GET`    | `/api/events`                          | Server-Sent Events; `change` event on start and after every edit, with digests and source URLs |
| `GET`    | `/ui`                                  | built-in control page; see [Interactivity](../concepts/interactivity.md) |
| `GET`    | `/api/cache`                           | cache stats |
| `DELETE` | `/api/cache`                           | clear cache |
| `GET`    | `/{name}/{format}/{path}`              | the spoofed dataset; see [Formats](../concepts/formats.md) |
| `GET`    | `/{name}/@{digest}/{format}/{path}`    | same, with a cache-busting token |

## PipelineSpec

```json
{
  "source": "/path/or/url",          // zarr/n5/precomputed array or s0..sN group, or file.h5::/dataset
  "ops": [{"op": "threshold", "low": 120}],
  "chunk_shape": [64, 64, 64],       // optional; default: source chunk shape
  "cache_source": true,              // cache raw chunks (stage 0)
  "voxel_size": [8, 8, 8],           // optional overrides of what the source reports
  "units": ["nm", "nm", "nm"],
  "axes": ["z", "y", "x"],
  "translation": [0, 0, 0]
}
```

Responses to `POST`, `PUT` and `GET /api/datasets/{name}` include `digest`, `source_dtype`,
per-level `levels`, and `sources`, a map from format to Neuroglancer source URL carrying the
new digest.
