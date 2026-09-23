"""Serve a synthetic volume through a threshold, in ~20 lines, as a library user would.

uv run python examples/quickstart.py
"""

import numpy as np
import tensorstore as ts
import uvicorn

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.neuroglancer import source_url, viewer_link
from chunkmirage.ops import Gaussian, Threshold

# 1. Make some data (any zarr/n5/precomputed/hdf5 path or URL works here).
path = "/tmp/chunkmirage-demo.zarr/s0"
z, y, x = np.mgrid[0:128, 0:256, 0:256]
data = ((np.sin(z / 9) * np.cos(y / 13) * np.sin(x / 7) + 1) * 120).astype(np.uint8)
arr = ts.open(
    {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": path},
        "metadata": {"chunks": [64, 64, 64]},
    },
    create=True,
    delete_existing=True,
    dtype=ts.uint8,
    shape=data.shape,
).result()
arr[...] = data

# 2. Build a pipeline. Gaussian has a halo and is marked cacheable in its own right if you
#    subclass it; here the raw chunks are cached, so retuning `low` only re-thresholds.
pipe = Pipeline(
    open_source(path, voxel_size=(8, 8, 8), units=("nm",) * 3),
    [Gaussian(sigma=2), Threshold(low=110)],
)

# 3. Serve through every frontend at once.
name, port = "demo", 8000
app = create_app({name: pipe})
src = source_url(f"http://localhost:{port}", name, "zarr3", "zarr3", pipe.digest())
print("neuroglancer:", viewer_link({name: pipe}, {name: src}))
print(
    f'edit live:    curl -X PUT localhost:8000/api/datasets/demo -d \'{{"source": "{path}", "ops": [{{"op": "threshold", "low": 160}}]}}\''
)
uvicorn.run(app, port=port)
