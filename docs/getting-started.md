# Getting started

## Install

```bash
git clone https://github.com/JaneliaSciComp/chunkmirage
cd chunkmirage
uv sync --all-extras --group dev        # or: pip install -e ".[all]"
```

Requires Python 3.11+. Core dependencies are tensorstore, numpy, numcodecs, starlette,
uvicorn, pydantic and typer. Optional extras: `hdf5` (h5py), `ops` (scipy filters),
`mcp` (planned MCP server).

## Serve something

```bash
chunkmirage serve /path/to/data.zarr/em/fibsem-uint8 --op threshold:low=120 --port 8000
```

`SOURCE` can be a zarr v2/v3 array or multiscale group (`s0`, `s1`, ...), an N5 dataset or
group, a Neuroglancer precomputed volume, or `file.h5::/dataset`. Local paths and
`s3://`, `gs://`, `http(s)://` URLs all work. The command prints:

```
source:       zarr3://http://localhost:8000/data/@<digest>/zarr3
neuroglancer: https://neuroglancer-demo.appspot.com/#!...
control API:  http://localhost:8000/api/datasets/data
```

Open the Neuroglancer link in Chrome or Firefox on the machine that can reach port 8000.
See [FAQ: does this work with the hosted Neuroglancer?](faq.md#does-this-work-with-neuroglancer-demoappspotcom)

## The demo

```bash
uv run python examples/demo.py --port 8000
```

Generates a synthetic multiscale volume of blobs, then serves it twice: `raw` as zarr v3
and `thresh` (gaussian then threshold) as a precomputed segmentation overlay. Both share
one cache, so the raw chunks are read once.

## Change the pipeline live

```bash
curl -X PUT localhost:8000/api/datasets/thresh -H 'content-type: application/json' \
  -d '{"source": "examples/demo-data.zarr/blobs",
       "ops": [{"op": "gaussian", "sigma": 1}, {"op": "threshold", "low": 160}]}'
```

The response contains new source URLs with an updated digest. Viewers cache chunks by URL,
so paste the new URL into the layer's source field and it refetches. Only the changed stage
recomputes; see [Caching](concepts/caching.md).

## Use it as a library

```python
from chunkmirage import Pipeline, open_source, create_app
from chunkmirage.ops import Threshold
import uvicorn

src = open_source("s3://bucket/data.zarr/em")           # multiscale group or single array
pipe = Pipeline(src, [Threshold(low=120)])
app = create_app({"em-thresh": pipe})                   # Starlette ASGI app
uvicorn.run(app, port=8000)
```

## Read the result from Python instead of a viewer

Anything that reads zarr over HTTP can consume a served pipeline as a lazy array:

```python
import tensorstore as ts
arr = ts.open({"driver": "zarr3",
               "kvstore": "http://localhost:8000/em-thresh/zarr3/s0/"}).result()
arr[0:64, 0:64, 0:64].read().result()
```
