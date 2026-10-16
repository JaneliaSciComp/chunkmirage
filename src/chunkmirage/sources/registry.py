from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from importlib.metadata import entry_points

from chunkmirage.cache import LRUCache
from chunkmirage.sources.base import KindSource, MultiscaleSource
from chunkmirage.sources.tensorstore_source import open_multiscale_tensorstore

Opener = Callable[..., MultiscaleSource]

# URL schemes: ``scheme://...`` opened by ``module:function``. The core's are the ones that
# compose other sources; packages add theirs through the ``chunkmirage.sources`` entry
# point (name: the scheme; value: an opener taking the URL and, if it wants them,
# ``cache_bytes`` and ``cache``), or ``register_source``. ``synthetic://``, ``register://``
# and ``stitch://`` come that way from ``chunkmirage_contrib``.
_BUILTIN = {
    "scene": "chunkmirage.sources.scene:open_scene",
    "warp": "chunkmirage.sources.warp:open_warp",
    "stack": "chunkmirage.sources.stack:open_stack",
    "flip": "chunkmirage.sources.flip:open_flip",
    "dataset": "chunkmirage.sources.dataset:open_dataset",
    "coverage": "chunkmirage.sources.dataset:open_coverage",
}
_SCHEMES: dict[str, str | Opener] = dict(_BUILTIN)
# Files told apart by their path rather than a scheme (a GeoTIFF by its extension): a
# matcher takes the path and returns the opener for it, or None. From the
# ``chunkmirage.source_matchers`` entry point (``module:function``) or ``register_matcher``.
Matcher = Callable[[str], Opener | None]
_MATCHERS: dict[str, str | Matcher] = {}
_ENTRYPOINTS_LOADED = False


def register_source(scheme: str, opener: Opener) -> None:
    """Open ``scheme://...`` URLs with ``opener(url, *, cache_bytes, cache)`` (it is passed
    only those of the two it takes), returning a ``MultiscaleSource``. The built-in schemes
    cannot be replaced."""
    if scheme in _BUILTIN:
        raise ValueError(f"{scheme}:// is built in")
    _SCHEMES[scheme] = opener


def register_matcher(name: str, matcher: Matcher) -> None:
    """Have ``matcher(path)`` asked, for a path with no known scheme, for the opener to read
    it with (None: not one of its files); ``name`` identifies it (a format's name)."""
    _MATCHERS[name] = matcher


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    for ep in entry_points(group="chunkmirage.sources"):
        if ep.name not in _BUILTIN:
            _SCHEMES.setdefault(ep.name, ep.value)
    for ep in entry_points(group="chunkmirage.source_matchers"):
        _MATCHERS.setdefault(ep.name, ep.value)


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


def _matchers() -> list[Matcher]:
    _load_entrypoints()
    out = []
    for name, m in _MATCHERS.items():
        if isinstance(m, str):
            module, _, attr = m.partition(":")
            m = _MATCHERS[name] = getattr(importlib.import_module(module), attr)
        out.append(m)
    return out


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
    ``scene://group?image=...&target=...`` resamples an image through OME-Zarr 0.6
    coordinate transformations (see ``sources.scene``); ``warp://image?field=swirl&...``
    twists an image through a procedural deformation (see ``sources.warp``); ``stack://a|b``
    serves images on one grid as the channels of one array, for ops over several images
    (see ``sources.stack``); ``flip://image?axes=y`` mirrors an image stored the other way
    round (see ``sources.flip``); ``dataset://name`` reads another dataset of the server
    in-process, through its stage cache (see ``sources.dataset``). The computed sources
    shipped in ``chunkmirage_contrib``
    come through the entry points like any other package's: ``synthetic://kind?shape=...``
    generates data procedurally; ``register://moving?fixed=...`` registers one image onto
    another, solving the deformation on a GPU when opened; ``stitch://project.xml``
    stitches a BigStitcher project's tiles by interest points and RANSAC when opened, and
    fuses them as they are read; ``.tif``/``.tiff`` paths are (cloud-optimized) GeoTIFFs,
    read tile by tile with their overviews as levels. Other packages add schemes of their
    own (``register_source``, or the ``chunkmirage.sources`` entry point) and files told
    apart by their path (``register_matcher``, or ``chunkmirage.source_matchers``).

    ``cache_bytes`` is tensorstore's pool of decoded source chunks; ``cache`` is the chunk
    cache that computed sources keep their expensive intermediates in (``register://``'s
    refined blocks), normally the pipeline's. ``kind`` says what the values are (``image``,
    ``label``, ``mask``) over what the source guessed: stored arrays guess from their dtype
    (``core.kind_for_dtype``), so a uint32 image needs ``kind="image"``.
    """
    source = _open(path, cache_bytes=cache_bytes, cache=cache, **kw)
    if kind is not None and kind != source.levels[0].info.kind:
        source = MultiscaleSource(
            [KindSource(lvl, kind) for lvl in source.levels],
            name=source.name,
            shader=source.shader,
            cached=source.cached,
        )
    return source


def _open(path: str, *, cache_bytes: int, cache: LRUCache | None, **kw) -> MultiscaleSource:
    scheme, sep, _ = path.partition("://")
    if sep and (opener := _opener(scheme)) is not None:
        return _call(opener, path, cache_bytes=cache_bytes, cache=cache)
    for matcher in _matchers():
        if (opener := matcher(path)) is not None:
            return _call(opener, path, cache_bytes=cache_bytes, cache=cache)
    if "::" in path or path.split("::")[0].endswith((".h5", ".hdf5")):
        from chunkmirage.sources.hdf5_source import open_multiscale_hdf5

        return open_multiscale_hdf5(path, cache_bytes=cache_bytes, **kw)
    return open_multiscale_tensorstore(path, cache_bytes=cache_bytes, **kw)
