"""``stack://`` sources: several images on one grid, served as the channels of one array, so
an op can compute something from all of them at once (``contacts``, masks, differences).

    stack://<image>|<image>[|<image>...]

Each image is anything ``open_source`` reads. Its first channel (index 0 on any leading
axis) becomes one channel of the stack, in the order given. Spatial axes may be any
permutation of ``(z, y, x)``; ``ResampleLevel`` transposes at read time. Levels are paired
by nearest voxel size; sources with a different grid, or fewer levels, are resampled to
the primary level's grid. The first source sets the pyramid.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.ndimage import affine_transform

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
        return self.read_into(box, np.empty(box.shape, dtype=self._info.dtype))

    def read_into(self, box: Box, out: np.ndarray) -> np.ndarray:
        space = Box(box.start[1:], box.stop[1:])
        for i, c in enumerate(range(box.start[0], box.stop[0])):
            src, lead = self.parts[c]
            b = Box(lead + space.start, tuple(v + 1 for v in lead) + space.stop)
            src.read_into(b, out[i][(None,) * len(lead)])
        return out


class ResampleLevel(Source):

    def __init__(self, src: Source, target_info: ArrayInfo, order: int = 1):
        self.src = src
        self.order = order
        s = src.info
        k = len(s.shape)
        src_axes = list(s.axes[-k:])
        tgt_axes = list(target_info.axes[-k:])
        if set(src_axes) != set(tgt_axes):
            raise ValueError(
                f"ResampleLevel: source spatial axes {src_axes} are not a permutation "
                f"of target spatial axes {tgt_axes}"
            )
        self._perm = [src_axes.index(t) for t in tgt_axes]        # target i -> source pos
        self._inv_perm = [tgt_axes.index(t) for t in src_axes]    # source pos -> target i

        self._info = ArrayInfo(
            shape=target_info.shape[-k:],
            dtype=s.dtype,
            chunk_shape=target_info.chunk_shape[-k:],
            voxel_size=target_info.voxel_size[-k:],
            units=target_info.units[-k:],
            axes=target_info.axes[-k:],
            translation=target_info.translation[-k:],
            kind=s.kind,
        )

        # Build A: source_index = A(target_index), in SOURCE-axis order.
        #   target_physical = tt + i_t * tv         (target index i_t, target order)
        #   source_index    = (target_physical - st) / sv
        #                   = (tt - st) / sv + i_t * (tv / sv)
        sv = np.asarray(s.voxel_size, dtype=float)              # source order
        tv = np.asarray(self._info.voxel_size, dtype=float)     # target order
        st = np.asarray(s.translation, dtype=float)             # source order
        tt = np.asarray(self._info.translation, dtype=float)    # target order
        tv_src = tv[self._perm]
        tt_src = tt[self._perm]
        self._scale = tv_src / sv                                # A's linear part
        self._shift = (tt_src - st) / sv                         # A's translation

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return (
            f"resample:{self.src.cache_key()}:{self._info.shape}:"
            f"{self._info.voxel_size}:{self._info.translation}:"
            f"{self._perm}:{self.order}"
        )

    def _to_source(self, idx_src_order: np.ndarray) -> np.ndarray:
        """Map a point (in source-axis-order target index space) to source index."""
        return idx_src_order * self._scale + self._shift

    def read(self, box: Box) -> np.ndarray:
        out_shape_tgt = np.asarray(box.shape, dtype=int)
        src_shape = np.asarray(self.src.info.shape, dtype=int)

        # Output box, in source-axis order
        start_src = np.asarray(box.start, dtype=int)[self._perm]
        stop_src = np.asarray(box.stop, dtype=int)[self._perm]

        # Source extent needed: continuous, then floor/ceil for the integer box
        lo_f = self._to_source(start_src)
        hi_f = self._to_source(stop_src)
        lo = np.floor(lo_f).astype(int)
        hi = np.ceil(hi_f).astype(int) + 1
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, src_shape)

        out_shape_src = out_shape_tgt[self._perm]

        if np.any(hi_c <= lo_c):
            # Entire output box is outside the source: fill with nearest-edge value.
            edge = np.clip(lo, 0, src_shape - 1)
            sample = self.src.read(Box(tuple(edge), tuple(edge + 1)))
            expanded = np.broadcast_to(
                sample.reshape(sample.shape + (1,) * len(out_shape_src)),
                sample.shape + tuple(int(v) for v in out_shape_src),
            )
            # source-order spatial axes -> target-order spatial axes
            spatial_perm = self._inv_perm
            full_perm = tuple(range(sample.ndim)) + tuple(
                sample.ndim + p for p in spatial_perm
            )
            return expanded.transpose(full_perm).copy()

        data = self.src.read(Box(tuple(lo_c), tuple(hi_c)))

        # Per the affine, output voxel ``start_src`` is at ``lo_f`` in source index
        # space. The sub-box read starts at ``lo_c``, so within ``data`` the same
        # physical position is at ``lo_f - lo_c``.
        offset = lo_f - lo_c

        resampled = affine_transform(
            data,
            np.diag(self._scale),
            offset=offset,
            output_shape=tuple(int(v) for v in out_shape_src),
            order=self.order,
            mode="nearest",
            prefilter=False,
        )
        # resampled is in source-axis order; transpose to target order
        return np.transpose(resampled, self._inv_perm)


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

            if _same_grid(lvl.info, n, base, primary_lead):
                parts.append((lvl, (0,) * n))
            else:
                parts.append((ResampleLevel(lvl, base, order=0), (0,) * n))

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