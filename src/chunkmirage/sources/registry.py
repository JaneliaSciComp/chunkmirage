from __future__ import annotations

from chunkmirage.cache import LRUCache
from chunkmirage.sources.base import MultiscaleSource
from chunkmirage.sources.tensorstore_source import open_multiscale_tensorstore


def open_source(
    path: str, *, cache_bytes: int = 0, cache: LRUCache | None = None, **kw
) -> MultiscaleSource:
    """Open any supported dataset as a ``MultiscaleSource``.

    Dispatch is by content, not extension: zarr v2/v3, n5 and neuroglancer precomputed go
    through tensorstore; ``.h5``/``.hdf5`` paths (``file.h5::/dataset``) go through h5py;
    ``synthetic://kind?shape=...`` generates data procedurally (see ``sources.synthetic``);
    ``scene://group?image=...&target=...`` resamples an image through OME-Zarr 0.6
    coordinate transformations (see ``sources.scene``); ``warp://image?field=swirl&...``
    twists an image through a procedural deformation (see ``sources.warp``);
    ``register://moving?fixed=...`` registers one image onto another, solving the
    deformation on a GPU when opened (see ``sources.register``); ``stack://a|b`` serves
    images on one grid as the channels of one array, for ops over several images (see
    ``sources.stack``); ``flip://image?axes=y`` mirrors an image stored the other way round
    (see ``sources.flip``); ``stitch://project.xml`` stitches a BigStitcher project's tiles by
    interest points and RANSAC when opened, and fuses them as they are read (see
    ``sources.stitch``); ``.tif``/``.tiff`` paths are (cloud-optimized) GeoTIFFs, read
    tile by tile with their overviews as levels (see ``sources.geotiff``).

    ``cache_bytes`` is tensorstore's pool of decoded source chunks; ``cache`` is the chunk
    cache that computed sources keep their expensive intermediates in (``register://``'s
    refined blocks), normally the pipeline's.
    """
    if path.startswith("synthetic://"):
        from chunkmirage.sources.synthetic import open_synthetic

        return open_synthetic(path)
    if path.startswith("scene://"):
        from chunkmirage.sources.scene import open_scene

        return open_scene(path, cache_bytes=cache_bytes)
    if path.startswith("warp://"):
        from chunkmirage.sources.warp import open_warp

        return open_warp(path, cache_bytes=cache_bytes)
    if path.startswith("register://"):
        from chunkmirage.sources.register import open_register

        return open_register(path, cache_bytes=cache_bytes, cache=cache)
    if path.startswith("stack://"):
        from chunkmirage.sources.stack import open_stack

        return open_stack(path, cache_bytes=cache_bytes, cache=cache)
    if path.startswith("stitch://"):
        from chunkmirage.sources.stitch import open_stitch

        return open_stitch(path, cache_bytes=cache_bytes, cache=cache)
    if path.startswith("flip://"):
        from chunkmirage.sources.flip import open_flip

        return open_flip(path, cache_bytes=cache_bytes, cache=cache)
    from chunkmirage.sources.geotiff import is_geotiff

    if is_geotiff(path):
        from chunkmirage.sources.geotiff import open_geotiff

        return open_geotiff(path, cache_bytes=cache_bytes)
    if "::" in path or path.split("::")[0].endswith((".h5", ".hdf5")):
        from chunkmirage.sources.hdf5_source import open_multiscale_hdf5

        return open_multiscale_hdf5(path, cache_bytes=cache_bytes, **kw)
    return open_multiscale_tensorstore(path, cache_bytes=cache_bytes, **kw)
