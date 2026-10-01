"""``stack://`` sources: several images on one grid, served as the channels of one array, so
an op can compute something from all of them at once (``contacts``, masks, differences).

    stack://<image>|<image>[|<image>...]

Each image is anything ``open_source`` reads. Its first channel (index 0 on any leading
axis) becomes one channel of the stack, in the order given. The images must share their
spatial grid level by level: shape, voxel size, translation and units; the stack has as
many levels as the image with the fewest. Nothing is copied: reading a stack chunk reads
the same box of each image, through each image's own cache.
"""

from __future__ import annotations

import hashlib

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

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
        space = Box(box.start[1:], box.stop[1:])
        out = np.empty(box.shape, dtype=self._info.dtype)
        for i, c in enumerate(range(box.start[0], box.stop[0])):
            src, lead = self.parts[c]
            b = Box(lead + space.start, tuple(v + 1 for v in lead) + space.stop)
            out[i] = src.read(b).reshape(space.shape)
        return out


def _n_lead(ms: MultiscaleSource, url: str) -> int:
    axes = ms.levels[0].info.axes
    n = 0
    while n < len(axes) and axes[n] not in _SPATIAL:
        n += 1
    if len(axes) - n != 3 or any(a not in _SPATIAL for a in axes[n:]):
        raise ValueError(f"{url}: stack:// needs images with axes ending in z, y, x; got {axes}")
    return n


def _same_grid(a: ArrayInfo, na: int, b: ArrayInfo, nb: int) -> bool:
    return (
        a.shape[na:] == b.shape[nb:]
        and a.units[na:] == b.units[nb:]
        and np.allclose(a.voxel_size[na:], b.voxel_size[nb:], rtol=1e-6, atol=0)
        and np.allclose(a.translation[na:], b.translation[nb:], rtol=0, atol=1e-6)
    )


def open_stack(url: str, *, cache_bytes: int = 0) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    urls = [u for u in url[len("stack://") :].split("|") if u]
    if len(urls) < 2:
        raise ValueError("stack:// needs two or more images separated by '|': stack://<a>|<b>")
    images = [open_source(u, cache_bytes=cache_bytes) for u in urls]
    leads = [_n_lead(ms, u) for ms, u in zip(images, urls)]
    n_levels = min(len(ms.levels) for ms in images)
    first = images[0]
    levels: list[Source] = []
    for i in range(n_levels):
        infos = [ms.levels[i].info for ms in images]
        for u, info, n in zip(urls[1:], infos[1:], leads[1:]):
            if not _same_grid(infos[0], leads[0], info, n):
                raise ValueError(
                    f"stack:// level {i}: {u} is on a different grid from {urls[0]} "
                    f"(shape {info.shape[n:]} at {info.voxel_size[n:]} vs "
                    f"{infos[0].shape[leads[0] :]} at {infos[0].voxel_size[leads[0] :]}); "
                    "stacked images must share shape, voxel size, translation and units"
                )
        parts = [(ms.levels[i], (0,) * n) for ms, n in zip(images, leads)]
        base, nb = infos[0], leads[0]
        info = ArrayInfo(
            shape=(len(parts), *base.shape[nb:]),
            dtype=np.result_type(*(x.dtype for x in infos)),
            chunk_shape=(len(parts), *base.chunk_shape[nb:]),
            voxel_size=(1.0, *base.voxel_size[nb:]),
            units=("", *base.units[nb:]),
            axes=("c", *base.axes[nb:]),
            translation=(0.0, *base.translation[nb:]),
        )
        ident = "|".join(f"{src.cache_key()}@{lead}" for src, lead in parts)
        key = "stack:" + hashlib.sha1(ident.encode()).hexdigest()[:16]
        levels.append(StackSource(parts, info, key))
    name = "+".join(ms.name or u.rstrip("/").rsplit("/", 1)[-1] for ms, u in zip(images, urls))
    return MultiscaleSource(levels, name=name or first.name)
