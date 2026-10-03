from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from importlib.metadata import entry_points

from chunkmirage.cache import LRUCache
from chunkmirage.sources.base import KindSource, MultiscaleSource
from chunkmirage.sources.tensorstore_source import open_multiscale_tensorstore

Opener = Callable[..., MultiscaleSource]

# URL schemes: ``scheme://...`` opened by ``module:function``. Other packages add theirs
# through the ``chunkmirage.sources`` entry point (name: the scheme; value: an opener taking
# the URL and, if it wants them, ``cache_bytes`` and ``cache``), or ``register_source``.
_BUILTIN = {
    "synthetic": "chunkmirage.sources.synthetic:open_synthetic",
    "scene": "chunkmirage.sources.scene:open_scene",
    "warp": "chunkmirage.sources.warp:open_warp",
    "register": "chunkmirage.sources.register:open_register",
    "stack": "chunkmirage.sources.stack:open_stack",
    "stitch": "chunkmirage.sources.stitch:open_stitch",
    "flip": "chunkmirage.sources.flip:open_flip",
}
_SCHEMES: dict[str, str | Opener] = dict(_BUILTIN)
_ENTRYPOINTS_LOADED = False


def register_source(scheme: str, opener: Opener) -> None:
    """Open ``scheme://...`` URLs with ``opener(url, *, cache_bytes, cache)`` (it is passed
    only those of the two it takes), returning a ``MultiscaleSource``. The built-in schemes
    cannot be replaced."""
    if scheme in _BUILTIN:
        raise ValueError(f"{scheme}:// is built in")
    _SCHEMES[scheme] = opener


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in entry_points(group="chunkmirage.sources"):
        if ep.name not in _BUILTIN:
            _SCHEMES.setdefault(ep.name, ep.value)


def schemes() -> list[str]:
    """The URL schemes ``open_source`` knows: the built-in ones and any registered."""
    _load_entrypoints()
    return sorted(_SCHEMES)


def _opener(scheme: str) -> Opener | None:
    _load_entrypoints()
    found = _SCHEMES.get(scheme)
    if isinstance(found, str):
        module, _, name = found.partition(":")
        found = _SCHEMES[scheme] = getattr(importlib.import_module(module), name)
    return found


def _call(opener: Opener, path: str, **kw) -> MultiscaleSource:
    """``opener(path, ...)`` with those of ``kw`` it takes (all of them if it takes ``**``)."""
    params = inspect.signature(opener).parameters.values()
    if not any(p.kind is p.VAR_KEYWORD for p in params):
        names = {p.name for p in params}
        kw = {k: v for k, v in kw.items() if k in names}
    return opener(path, **kw)


def open_source(
    path: str,
    *,
    cache_bytes: int = 0,
    cache: LRUCache | None = None,
    kind: str | None = None,
    **kw,
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
    tile by tile with their overviews as levels (see ``sources.geotiff``). Other packages
    add schemes of their own (``register_source``, or the ``chunkmirage.sources`` entry
    point).

    ``cache_bytes`` is tensorstore's pool of decoded source chunks; ``cache`` is the chunk
    cache that computed sources keep their expensive intermediates in (``register://``'s
    refined blocks), normally the pipeline's. ``kind`` says what the values are (``image``,
    ``label``, ``mask``) over what the source guessed: stored arrays guess from their dtype
    (``core.kind_for_dtype``), so a uint32 image needs ``kind="image"``.
    """
    source = _open(path, cache_bytes=cache_bytes, cache=cache, **kw)
    if kind is not None and kind != source.levels[0].info.kind:
        source = MultiscaleSource(
            [KindSource(lvl, kind) for lvl in source.levels], name=source.name, shader=source.shader
        )
    return source


def _open(path: str, *, cache_bytes: int, cache: LRUCache | None, **kw) -> MultiscaleSource:
    scheme, sep, _ = path.partition("://")
    if sep and (opener := _opener(scheme)) is not None:
        return _call(opener, path, cache_bytes=cache_bytes, cache=cache)
    from chunkmirage.sources.geotiff import is_geotiff

    if is_geotiff(path):
        from chunkmirage.sources.geotiff import open_geotiff

        return open_geotiff(path, cache_bytes=cache_bytes)
    if "::" in path or path.split("::")[0].endswith((".h5", ".hdf5")):
        from chunkmirage.sources.hdf5_source import open_multiscale_hdf5

        return open_multiscale_hdf5(path, cache_bytes=cache_bytes, **kw)
    return open_multiscale_tensorstore(path, cache_bytes=cache_bytes, **kw)
