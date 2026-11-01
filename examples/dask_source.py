"""A lazy dask array served as a pyramid, as it is and with a chunkmirage op on top: each
chunk a viewer asks for computes only the dask chunks under it.

    uv run --with dask python examples/dask_source.py          # in a clone, after uv sync --extra ops
    uv run --with "chunkmirage[ops]>=0.1.0a5" --with dask python dask_source.py   # anywhere else

The graph is noise blurred by ``map_overlap``, which reads each chunk's neighbours, so the
blobs run on across chunk edges. Each coarser level is the mean of 2x2x2 blocks of the one
before (``da.coarsen``), lazy too. Any Python array works the same way: a ``Source`` says
what the array is (``info``) and reads a box of it (``read``).
"""

import dask.array as da
import numpy as np
import uvicorn
from scipy import ndimage as ndi

from chunkmirage import Pipeline, create_app
from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo
from chunkmirage.neuroglancer import source_url, viewer_link
from chunkmirage.sources import MultiscaleSource, Source


class DaskSource(Source):
    """One level: a dask array whose voxels are ``factor`` nm across."""

    def __init__(self, arr: da.Array, factor: int = 1):
        self.arr = arr
        self._info = ArrayInfo(
            shape=arr.shape,
            dtype=arr.dtype,
            chunk_shape=tuple(c[0] for c in arr.chunks),
            voxel_size=(float(factor),) * 3,
            units=("nm",) * 3,
            axes=("z", "y", "x"),
            translation=((factor - 1) / 2,) * 3,  # voxel centres, as OME-Zarr places them
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return "dask:" + self.arr.name  # dask's name for the graph: a new graph, a new cache

    def read(self, box):
        return np.asarray(self.arr[box.slices()].compute())


# 1. Any dask graph: here noise, blurred with each chunk's neighbours read in, and stretched
#    to about 0-1 (the blur's spread is about 0.005) so a viewer shows it.
noise = da.random.default_rng(0).random((512, 512, 512), chunks=128, dtype=np.float32)
blurred = da.map_overlap(
    lambda b: 0.5 + 30 * (ndi.gaussian_filter(b, 4) - 0.5), noise, depth=12, boundary="reflect"
)

# 2. A pyramid of it, each level the mean of 2x2x2 blocks of the one before.
levels = [DaskSource(blurred)]
for f in (2, 4, 8, 16):
    levels.append(
        DaskSource(da.coarsen(np.mean, levels[-1].arr, {0: 2, 1: 2, 2: 2}).rechunk(128), f)
    )

# 3. Served in every format, as it is and with a chunkmirage op on top: blobs where the blur
#    is above its mean. The two share a cache, so dask computes each chunk once.
source, cache, port = MultiscaleSource(levels, name="blurred"), LRUCache(), 8000
pipes = {
    "blurred": Pipeline(source, [], cache=cache),
    "blobs": Pipeline(source, [{"op": "threshold", "low": 0.5}], cache=cache),
}
app = create_app(pipes)
urls = {
    n: source_url(f"http://localhost:{port}", n, "zarr3", "zarr3", p.digest())
    for n, p in pipes.items()
}
print("neuroglancer:", viewer_link(pipes, urls))
uvicorn.run(app, port=port)
