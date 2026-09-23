# chunkmirage

**Spoof chunked array formats over HTTP with on-the-fly processing.**

chunkmirage serves *virtual* datasets that look, to any HTTP-capable viewer
(Neuroglancer, BigDataViewer/Fiji, vizarr, napari, webKnossos, ...), like ordinary
Zarr v2, Zarr v3, N5, or Neuroglancer Precomputed volumes. Nothing exists on disk.
Every chunk is computed when requested: read from a real source (zarr, n5, precomputed,
HDF5, local or S3/GCS/HTTP), pushed through a pipeline of ops (threshold, filter,
model inference, resampling under a registration transform, ...), encoded in whatever
format the viewer asked for, and cached so that tweaking a parameter downstream never
re-reads or re-computes upstream stages.

It generalizes [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5)
and the serving layer of [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow),
and is intended to become the format/serving/caching backbone that cellmap-flow imports.

```
viewer  --HTTP-->  chunkmirage  --tensorstore/h5py-->  real data (zarr/n5/precomputed/hdf5, file/s3/gcs/http)
                      |
                      +-- pipeline: source -> [op, op, ...] -> encoded chunk
                      +-- per-stage chunk cache (raw, inference output, ...) keyed by pipeline hash
                      +-- frontends: n5 | zarr (v2) | zarr3 | precomputed, all served at once
                      +-- REST API for live pipeline edits (UI / MCP / scripts)
```

## Quick start

```bash
uv sync --all-extras --group dev            # or: pip install -e ".[all]"
chunkmirage serve /path/to/data.zarr/em/fibsem-uint8 --op threshold:low=120 --port 8000
```

Then open the printed Neuroglancer link, or point any viewer at one of:

| viewer source URL                                  | format                    |
| -------------------------------------------------- | ------------------------- |
| `n5://http://localhost:8000/<name>/n5`             | N5                        |
| `zarr://http://localhost:8000/<name>/zarr`         | Zarr v2 (+ OME-NGFF 0.4)  |
| `zarr3://http://localhost:8000/<name>/zarr3`       | Zarr v3 (+ OME-NGFF 0.5)  |
| `precomputed://http://localhost:8000/<name>/precomputed` | Neuroglancer precomputed |

Change the pipeline live without restarting:

```bash
curl -X PUT localhost:8000/api/datasets/<name> -H 'content-type: application/json' \
  -d '{"source": "/path/to/data.zarr/em/fibsem-uint8", "ops": [{"op": "threshold", "low": 150}]}'
```

Upstream stages stay cached; only the changed stage and its dependents recompute.

## As a library

```python
from chunkmirage import Pipeline, open_source, create_app
from chunkmirage.ops import Threshold

src = open_source("s3://bucket/data.zarr/em/s0")        # or .n5, precomputed, .h5
pipe = Pipeline(src, ops=[Threshold(low=120)])
app = create_app({"em-thresh": pipe})                   # a Starlette ASGI app
# uvicorn.run(app, port=8000)
```

Custom ops are plain classes: declare a `halo` if you need neighbourhood context and
mark `cache=True` for expensive stages (e.g. model inference). Register via the
`chunkmirage.ops` entry point or `chunkmirage.ops.register`.

```python
from chunkmirage.ops import Op
import numpy as np

class MyModel(Op):
    name = "my_model"
    halo = 16          # voxels of context pulled from upstream on every side
    cache = True       # cache this stage's output chunks

    def output_dtype(self, in_dtype): return np.dtype("uint8")
    def apply(self, block: np.ndarray) -> np.ndarray:
        return run_my_network(block)
```

Full documentation: **https://davidackerman.github.io/chunkmirage/** (built from `docs/` with MkDocs; run `uv run mkdocs serve` locally).

See [docs/design.md](docs/design.md) for the architecture, language/stack rationale,
caching model, extensibility plan (plugins, REST, MCP), and the client-side (browser /
WebGPU) roadmap.

## Status

Early scaffold. Working: tensorstore sources (zarr v2, n5, precomputed over file/s3/gcs/http),
N5 / Zarr v2 / Zarr v3 / precomputed frontends, threshold/cast/scale/filter ops with halo
support, per-stage LRU cache, live REST edits, CLI. Not yet: HDF5 source, GPU ops,
registration/resampling op, MCP server, browser build.

## License

BSD 3-Clause, CellMap Project Team.
