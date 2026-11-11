"""``stack://`` sources: several images served as the channels of one array, so an op can
compute something from all of them at once (``contacts``, masks, differences).

    stack://<image>|<image>[|<image>...]

Each image is anything ``open_source`` reads. Its first channel (index 0 on any leading
axis) becomes one channel of the stack, in the order given. The first image sets the grid:
the stack has its levels, down to the coarsest that every other image has a level as coarse
as, so no image is read at a much finer resolution than the level it serves. At each level,
every other image's nearest level is read as it is when it is on that level's grid, and
otherwise resampled onto it as ``scene://`` resamples (``SceneLevelSource``): by nearest
voxel for labels and masks, linearly otherwise, 0 outside the image. The images must have
the same spatial axes, in the same order, and the same units. Nothing is copied: reading a
stack chunk reads the same box of each image (or the box covering it, when resampled),
through each image's own cache.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource
from chunkmirage.transforms import Affine

_SPATIAL = ("z", "y", "x")


class StackSource(Source):
    """One level of a stack: channel ``c`` is ``parts[c]`` read at its leading indices."""

    def __init__(self, parts: list[tuple[Source, tuple[int, ...]]], info: ArrayInfo, key: str):
        self.parts = parts
        self._info = info
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        return self.read_into(box, np.empty(box.shape, dtype=self._info.dtype))

    def read_into(self, box: Box, out: np.ndarray) -> np.ndarray:
        space = Box(box.start[1:], box.stop[1:])
        for i, c in enumerate(range(box.start[0], box.stop[0])):
            src, lead = self.parts[c]
            b = Box(lead + space.start, tuple(v + 1 for v in lead) + space.stop)
            src.read_into(b, out[i][(None,) * len(lead)])  # each image's read lands in its channel
        return out


def _n_lead(ms: MultiscaleSource, url: str) -> int:
    axes = ms.levels[0].info.axes
    n = 0
    while n < len(axes) and axes[n] not in _SPATIAL:
        n += 1
    if len(axes) - n not in (2, 3) or any(a not in _SPATIAL for a in axes[n:]):
        raise ValueError(f"{url}: stack:// needs images with axes ending in z, y, x (or y, x); got {axes}")
    return n


def _same_grid(a: ArrayInfo, na: int, b: ArrayInfo, nb: int) -> bool:
    return (
        a.shape[na:] == b.shape[nb:]
        and a.units[na:] == b.units[nb:]
        and np.allclose(a.voxel_size[na:], b.voxel_size[nb:], rtol=1e-6, atol=0)
        and np.allclose(a.translation[na:], b.translation[nb:], rtol=0, atol=1e-6)
    )


def _check_compatible(first: ArrayInfo, nf: int, info: ArrayInfo, n: int, url: str, first_url: str) -> None:
    """Refuse an image that cannot be put on the first's grid by scaling and shifting it."""
    if tuple(info.axes[n:]) != tuple(first.axes[nf:]):
        raise ValueError(
            f"stack:// {url}: spatial axes {tuple(info.axes[n:])} are not {first_url}'s "
            f"{tuple(first.axes[nf:])}; images with their axes in another order are not reordered"
        )
    if tuple(info.units[n:]) != tuple(first.units[nf:]):
        raise ValueError(
            f"stack:// {url}: units {tuple(info.units[n:])} are not {first_url}'s {tuple(first.units[nf:])}"
        )


def _reaches(ms: MultiscaleSource, n: int, voxel_size: Sequence[float]) -> bool:
    """Whether ``ms`` has a level at least as coarse as ``voxel_size`` along every axis."""
    return any(
        all(v >= w * (1 - 1e-6) for v, w in zip(lvl.info.voxel_size[n:], voxel_size)) for lvl in ms.levels
    )


def _resampled(src: Source, n: int, grid: ArrayInfo, ng: int) -> Source:
    """``src`` (``n`` leading axes, kept) on ``grid``'s spatial voxels: by nearest voxel for
    labels and masks, linearly otherwise; 0 outside ``src``."""
    s = src.info
    info = ArrayInfo(
        shape=tuple(s.shape[:n]) + tuple(grid.shape[ng:]),
        dtype=s.dtype,
        chunk_shape=tuple(s.shape[:n]) + tuple(grid.chunk_shape[ng:]),
        voxel_size=tuple(s.voxel_size[:n]) + tuple(grid.voxel_size[ng:]),
        units=tuple(s.units[:n]) + tuple(grid.units[ng:]),
        axes=tuple(s.axes[:n]) + tuple(grid.axes[ng:]),
        translation=tuple(s.translation[:n]) + tuple(grid.translation[ng:]),
        kind=s.kind,
    )
    sv, st = np.asarray(s.voxel_size[n:], dtype=float), np.asarray(s.translation[n:], dtype=float)
    gv, gt = np.asarray(grid.voxel_size[ng:], dtype=float), np.asarray(grid.translation[ng:], dtype=float)
    # voxel i of the grid is centred at gt + i * gv, which is voxel (gt + i * gv - st) / sv of src
    mapping = Affine.scale_translation(gv / sv, (gt - st) / sv)
    order = 0 if s.kind in ("label", "mask") else 1
    key = f"resampled:{tuple(grid.shape[ng:])}:{tuple(gv)}:{tuple(gt)}:{order}:{src.cache_key()}"
    return SceneLevelSource(src, mapping, info, n, order, key)


def open_stack(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    urls = [u for u in url[len("stack://") :].split("|") if u]
    if len(urls) < 2:
        raise ValueError("stack:// needs two or more images separated by '|': stack://<a>|<b>")
    images = [open_source(u, cache_bytes=cache_bytes, cache=cache) for u in urls]
    leads = [_n_lead(ms, u) for ms, u in zip(images, urls)]
    first, nf = images[0], leads[0]
    others = list(zip(urls[1:], images[1:], leads[1:]))
    for u, ms, n in others:
        _check_compatible(first.levels[0].info, nf, ms.levels[0].info, n, u, urls[0])
    # the first image's levels, down to the coarsest every other image has a level as coarse
    # as: a mask at one resolution, say, ends the stack there rather than being read at its
    # own resolution for every chunk of a much coarser level
    n_levels = 1
    while n_levels < len(first.levels) and all(
        _reaches(ms, n, first.levels[n_levels].info.voxel_size[nf:]) for _, ms, n in others
    ):
        n_levels += 1
    levels: list[Source] = []
    for i in range(n_levels):
        base = first.levels[i].info
        parts: list[tuple[Source, tuple[int, ...]]] = [(first.levels[i], (0,) * nf)]
        for _, ms, n in others:
            lvl = ms.levels[ms.nearest_level(base.voxel_size[nf:])]
            parts.append((lvl if _same_grid(base, nf, lvl.info, n) else _resampled(lvl, n, base, nf), (0,) * n))
        info = ArrayInfo(
            shape=(len(parts), *base.shape[nf:]),
            dtype=np.result_type(*(src.info.dtype for src, _ in parts)),
            chunk_shape=(len(parts), *base.chunk_shape[nf:]),
            voxel_size=(1.0, *base.voxel_size[nf:]),
            units=("", *base.units[nf:]),
            axes=("c", *base.axes[nf:]),
            translation=(0.0, *base.translation[nf:]),
        )
        ident = "|".join(f"{src.cache_key()}@{lead}" for src, lead in parts)
        key = "stack:" + hashlib.sha1(ident.encode()).hexdigest()[:16]
        levels.append(StackSource(parts, info, key))
    name = "+".join(ms.name or u.rstrip("/").rsplit("/", 1)[-1] for ms, u in zip(images, urls))
    return MultiscaleSource(levels, name=name or first.name)
