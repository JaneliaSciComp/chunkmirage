"""``stack://`` sources: several images on one grid, served as the channels of one array, so
an op can compute something from all of them at once (``contacts``, masks, differences).

    stack://<image>|<image>[|<image>...]

Each image is anything ``open_source`` reads. Its first channel (index 0 on any leading
axis) becomes one channel of the stack, in the order given. Spatial axes may be any
permutation of ``(z, y, x)``; ``ResampleLevel`` transposes at read time. Levels are paired
by nearest voxel size; sources with a different grid, or fewer levels, are resampled to
the primary level's grid. The first source sets the pyramid.

Sources that are not on the primary grid are resampled with ``SceneLevelSource``: points
outside the source image read as 0, matching ``scene://``. Units and the set of spatial
axes must match the primary source; a mismatch is an error, not something to resample.
"""

from __future__ import annotations

import hashlib

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
            src.read_into(b, out[i][(None,) * len(lead)])
        return out


def _n_lead(ms: MultiscaleSource, url: str) -> int:
    axes = ms.levels[0].info.axes
    n = 0
    while n < len(axes) and axes[n] not in _SPATIAL:
        n += 1
    if len(axes) - n not in (2, 3) or any(a not in _SPATIAL for a in axes[n:]):
        raise ValueError(
            f"{url}: stack:// needs images with axes ending in a permutation of z, y, x; got {axes}"
        )
    return n


def _same_grid(a: ArrayInfo, na: int, b: ArrayInfo, nb: int) -> bool:
    if list(a.axes[na:]) != list(b.axes[nb:]):
        return False
    return (
        a.shape[na:] == b.shape[nb:]
        and a.units[na:] == b.units[nb:]
        and np.allclose(a.voxel_size[na:], b.voxel_size[nb:], rtol=1e-6, atol=0)
        and np.allclose(a.translation[na:], b.translation[nb:], rtol=0, atol=1e-6)
    )


def _check_compatible(base: ArrayInfo, nb: int, info: ArrayInfo, ni: int, url: str, primary_url: str) -> None:
    """Reject images whose spatial axes or units cannot be stacked onto the primary's."""
    b_axes = list(base.axes[nb:])
    i_axes = list(info.axes[ni:])
    if set(i_axes) != set(b_axes):
        raise ValueError(
            f"stack:// {url}: spatial axes {i_axes} are not a permutation of "
            f"{primary_url}'s {b_axes}"
        )
    b_units = dict(zip(b_axes, base.units[nb:]))
    i_units = dict(zip(i_axes, info.units[ni:]))
    if i_units != b_units:
        bad = {a: (i_units.get(a), b_units.get(a)) for a in b_axes if i_units.get(a) != b_units.get(a)}
        raise ValueError(
            f"stack:// {url}: spatial units {i_units} do not match {primary_url}'s "
            f"{b_units} (axis: got, expected): {bad}"
        )


def _affine_to_source(src: ArrayInfo, ns: int, tgt: ArrayInfo, nt: int) -> Affine:
    """Source spatial index = A(target spatial index), in source-axis order.

        target_physical = tt + i_t * tv           (target index, target order)
        source_index    = (target_physical - st) / sv
                        = (tt - st) / sv + i_t * (tv / sv)

    The transposition between source and target axis order is folded in so the returned
    affine takes target-order spatial indices and returns source-order spatial indices.
    """
    s_axes = list(src.axes[ns:])
    t_axes = list(tgt.axes[nt:])
    perm = [s_axes.index(a) for a in t_axes]  # target i -> source pos
    sv = np.asarray(src.voxel_size[ns:], dtype=float)          # source order
    tv = np.asarray(tgt.voxel_size[nt:], dtype=float)          # target order
    st = np.asarray(src.translation[ns:], dtype=float)         # source order
    tt = np.asarray(tgt.translation[nt:], dtype=float)         # target order
    scale = tv[perm] / sv
    shift = (tt[perm] - st) / sv
    n = len(s_axes)
    m = np.zeros((n, n + 1))
    m[:, :n] = np.diag(scale)
    m[:, -1] = shift
    return Affine(m)


def _resampled(src: Source, ns: int, tgt: ArrayInfo, nt: int, key: str) -> Source:
    """``src`` resampled onto ``tgt``'s spatial grid, via ``SceneLevelSource``."""
    k = len(tgt.axes) - nt
    tgt_spatial = ArrayInfo(
        shape=tgt.shape[nt:],
        dtype=src.info.dtype,
        chunk_shape=tgt.chunk_shape[nt:],
        voxel_size=tgt.voxel_size[nt:],
        units=tgt.units[nt:],
        axes=tgt.axes[nt:],
        translation=tgt.translation[nt:],
        kind=src.info.kind,
    )
    mapping = _affine_to_source(src.info, ns, tgt, nt)
    return SceneLevelSource(src, mapping, tgt_spatial, ns, order=0, key=key)


def open_stack(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    urls = [u for u in url[len("stack://") :].split("|") if u]
    if len(urls) < 2:
        raise ValueError("stack:// needs two or more images separated by '|': stack://<a>|<b>")
    images = [open_source(u, cache_bytes=cache_bytes, cache=cache) for u in urls]
    leads = [_n_lead(ms, u) for ms, u in zip(images, urls)]

    primary = images[0]
    primary_lead = leads[0]
    n_levels = len(primary.levels)

    p_axes = list(primary.levels[0].info.axes[primary_lead:])
    p_set = set(p_axes)
    for u, ms, n in zip(urls[1:], images[1:], leads[1:]):
        s_axes = list(ms.levels[0].info.axes[n:])
        if set(s_axes) != p_set:
            raise ValueError(
                f"stack:// {u}: spatial axes {s_axes} are not a permutation of "
                f"{urls[0]}'s {p_axes}"
            )

    levels: list[Source] = []
    for i in range(n_levels):
        base = primary.levels[i].info
        parts: list[tuple[Source, tuple[int, ...]]] = [(primary.levels[i], (0,) * primary_lead)]

        for u, ms, n in zip(urls[1:], images[1:], leads[1:]):
            try:
                j = ms.nearest_level(base.voxel_size[primary_lead:], fitted=False)
            except TypeError:
                j = ms.nearest_level(base.voxel_size[primary_lead:])
            j = min(max(j, 0), len(ms.levels) - 1)
            lvl = ms.levels[j]

            _check_compatible(base, primary_lead, lvl.info, n, u, urls[0])

            if _same_grid(lvl.info, n, base, primary_lead):
                parts.append((lvl, (0,) * n))
            else:
                key = (
                    f"resample:{lvl.cache_key()}:{base.shape[primary_lead:]}:"
                    f"{base.voxel_size[primary_lead:]}:{base.translation[primary_lead:]}"
                )
                parts.append((_resampled(lvl, n, base, primary_lead, key), (0,) * n))

        info = ArrayInfo(
            shape=(len(parts), *base.shape[primary_lead:]),
            dtype=np.result_type(*(p[0].info.dtype for p in parts)),
            chunk_shape=(len(parts), *base.chunk_shape[primary_lead:]),
            voxel_size=(1.0, *base.voxel_size[primary_lead:]),
            units=("", *base.units[primary_lead:]),
            axes=("c", *base.axes[primary_lead:]),
            translation=(0.0, *base.translation[primary_lead:]),
        )
        ident = "|".join(f"{src.cache_key()}@{lead}" for src, lead in parts)
        key = "stack:" + hashlib.sha1(ident.encode()).hexdigest()[:16]
        levels.append(StackSource(parts, info, key))

    name = "+".join(ms.name or u.rstrip("/").rsplit("/", 1)[-1] for ms, u in zip(images, urls))
    return MultiscaleSource(levels, name=name or primary.name)