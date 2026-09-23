from __future__ import annotations

from chunkmirage.sources.base import MultiscaleSource
from chunkmirage.sources.tensorstore_source import open_multiscale_tensorstore


def open_source(path: str, *, cache_bytes: int = 0, **kw) -> MultiscaleSource:
    """Open any supported dataset as a ``MultiscaleSource``.

    Dispatch is by content, not extension: zarr v2/v3, n5 and neuroglancer precomputed go
    through tensorstore; ``.h5``/``.hdf5`` paths (``file.h5::/dataset``) go through h5py;
    ``synthetic://kind?shape=...`` generates data procedurally (see ``sources.synthetic``).
    """
    if path.startswith("synthetic://"):
        from chunkmirage.sources.synthetic import open_synthetic

        return open_synthetic(path)
    if "::" in path or path.split("::")[0].endswith((".h5", ".hdf5")):
        from chunkmirage.sources.hdf5_source import open_multiscale_hdf5

        return open_multiscale_hdf5(path, cache_bytes=cache_bytes, **kw)
    return open_multiscale_tensorstore(path, cache_bytes=cache_bytes, **kw)
