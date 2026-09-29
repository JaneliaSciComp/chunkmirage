# chunkmirage

**Spoof chunked array formats over HTTP with on-the-fly processing.**

chunkmirage serves *virtual* datasets that look, to any HTTP-capable viewer
(Neuroglancer, BigDataViewer/Fiji, vizarr, napari, webKnossos, ...), like ordinary
Zarr v2, Zarr v3, N5, or Neuroglancer Precomputed volumes. Nothing exists on disk.
Every chunk is computed when requested: read from a real source (zarr, n5, precomputed,
HDF5, local or S3/GCS/HTTP) or generated (an image registered on the fly, a procedural
volume), pushed through a pipeline of ops (threshold, filter, model inference, ...),
encoded in whatever format the viewer asked for, and cached so that tweaking a parameter
downstream never re-reads or re-computes upstream stages.

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
uv sync --extra all --group dev            # or: pip install -e ".[all]"
chunkmirage serve /path/to/data.zarr/em/fibsem-uint8 --op threshold:low=120 --port 8000
```

### Try it with no data at all

1. Install and start the demo:

   ```bash
   git clone https://github.com/yuriyzubov/chunkmirage && cd chunkmirage
   uv sync --extra all
   uv run chunkmirage serve "synthetic://blobs+noise?shape=4096,4096,4096" \
       --op gaussian:sigma=1.5 --op threshold:low=110 \
       --op morphology:operation=open,radius=2 --op label:min_size=200 --python-viewer
   ```

2. Open the printed **control UI** URL (`http://<your-ip>:8000/ui`). Neuroglancer is embedded
   in the page with two layers: `raw` (the generated volume) and `processed` (smooth →
   threshold → open → connected components, coloured by object).

3. Drag the `low` slider under `threshold`: objects appear and merge as the view updates in
   place. Change `radius` under `morphology` to remove specks, or `min_size` under `label` to
   drop small objects. Only the changed stage recomputes; the generated data stays cached.

`synthetic://` sources are computed from voxel coordinates on demand, so this 4096³ volume
(69 gigavoxels) exists nowhere and its multiscale pyramid is exact. Kinds: `blobs`, `shells`
(hollow spheres), `noise` (fractal), `julia` (3-D fractal slice), combinable with `+`. To view
from another machine with the hosted Neuroglancer, add `--https` and accept the certificate
once. Details: [getting started](docs/getting-started.md).

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

### Registration without writing it out

A `scene://` source serves an image resampled through OME-Zarr 0.6 coordinate
transformations, displacement fields included, which no viewer applies itself yet. Every
viewer sees the registered volume and nothing is written:

```bash
# download the RFC-5 fly-brain example (~49 MB) and serve fixed, moving and registered
uv run python examples/fly_brain_registration.py /tmp/fly
# or serve just the registered brain
chunkmirage serve "scene:///tmp/fly/fly_brains.zarr?image=FCWB&target=JRC2018F"
```

`register://` sources solve the registration too: a deformable field fitted on the GPU
in seconds from coarse levels, then every level served through it. The demo compares
before and after in the viewer's own GPU shader, and solves again as you change settings:

```bash
uv sync --extra all --extra gpu        # gpu: PyTorch, about 3 GB
uv run python examples/register_demo.py FIXED MOVING --affine fixed_to_moving.npy
```

With no arguments it registers a synthetic volume onto a swirled copy of itself. The whole
thing also runs in the browser: `uv run python examples/browser/serve.py` and open the
printed `register.html` link. The page reads both images straight from their URLs, solves on
the viewer's GPU (WebGPU), and shows before and after in Neuroglancer, the registered volume
computed in the browser as the viewer asks for it. `warp://`
sources make such swirls, optionally along a time axis that Neuroglancer plays:
`uv run python examples/swirl_demo.py --animate`. Details:
[formats](docs/concepts/formats.md#scene-sources-ome-zarr-06-transformations).

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

Full documentation: **https://yuriyzubov.github.io/chunkmirage/** (built from `docs/` with MkDocs; run `uv run mkdocs serve` locally).

See [docs/design.md](docs/design.md) for the architecture, language/stack rationale,
caching model, extensibility plan (plugins, REST, MCP), and the client-side (browser /
WebGPU) roadmap.

## Status

Early, but working:

* **Sources:** zarr v2/v3, N5 and precomputed (file, S3, GCS, HTTP) and HDF5; computed
  `synthetic://` volumes, `scene://` registration through OME-Zarr 0.6 transformations,
  `warp://` procedural deformations, and `register://` deformable registration solved on
  a GPU.
* **Frontends:** N5, Zarr v2, Zarr v3 and precomputed, all served at once.
* **Ops:** threshold, cast, scale, Gaussian, uniform and difference-of-Gaussians filters,
  morphology, connected components; halos handled for you.
* **Live editing:** per-stage LRU cache, REST edits, a control page, and a
  python-neuroglancer viewer that keeps the camera while layers refetch.

Not yet: GPU ops (only registration uses the GPU), MCP server, browser build. See the [roadmap](docs/roadmap.md).

## License

BSD 3-Clause, Howard Hughes Medical Institute. Authors: TBD (collaborative project).
