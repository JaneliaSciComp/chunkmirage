from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from chunkmirage import demand
from chunkmirage.cache import LRUCache
from chunkmirage.core import KINDS, ArrayInfo, Box
from chunkmirage.fused import tent


class Source(ABC):
    """Read-only, random-access array. Everything upstream of an op is a ``Source``."""

    @property
    @abstractmethod
    def info(self) -> ArrayInfo: ...

    @abstractmethod
    def read(self, box: Box) -> np.ndarray:
        """Return data for ``box`` (must lie within ``info.shape``), shape == box.shape."""

    def read_padded(self, box: Box, fill=0, *, edge: bool = False) -> np.ndarray:
        """Read ``box`` even if it pokes outside the array. Out-of-range voxels are ``fill``,
        or with ``edge`` the nearest voxel inside (so a filter sees no step at the border)."""
        clipped = box.clip(self.info.shape)
        if clipped == box:
            return self.read(box)
        if edge and not clipped.empty:
            inner = clipped.relative_to(box)
            pad = [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)]
            return np.pad(self.read(clipped), pad, mode="edge")
        out = np.full(box.shape, fill, dtype=self.info.dtype)
        if not clipped.empty:
            out[clipped.relative_to(box).slices()] = self.read(clipped)
        return out

    def cache_key(self) -> str:
        """Stable identity used to build cache keys for downstream stages."""
        return f"{type(self).__name__}:{id(self)}"


class KindSource(Source):
    """``inner`` with its values said to be ``kind`` (``ArrayInfo.kind``), whatever it
    guessed: a uint32 image, or labels stored as uint16."""

    def __init__(self, inner: Source, kind: str | None):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.inner = inner
        self._info = inner.info.with_(kind=kind)
        self._key = f"kind={kind}:{inner.cache_key()}"  # resampled and downsampled differently

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        return self.inner.read(box)


class ChunkedSource(Source):
    """A source materialized chunk-by-chunk via ``compute_chunk`` and memoized in an LRU.

    ``read(box)`` gathers the covering chunks (from cache or freshly computed) and slices.
    This is the single mechanism behind *both* raw-source caching and per-stage caching:
    every pipeline stage is exposed to the next one as a ``ChunkedSource``.
    """

    def __init__(
        self,
        info: ArrayInfo,
        compute_chunk: Callable[[tuple[int, ...]], np.ndarray],
        cache: LRUCache | None,
        key: str,
    ):
        self._info = info
        self._compute = compute_chunk
        self._cache = cache
        self._key = key
        #: the overlapping blocks this stage is blended from, if any (an op with ``overlap``)
        self.blocks: BlockSource | None = None
        # In-flight computations keyed by chunk index: concurrent requests for the same chunk
        # (a viewer retrying, or two clients) wait for one computation instead of duplicating it.
        self._inflight: dict[tuple[int, ...], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    @property
    def cached(self) -> bool:
        """Whether this stage memoizes its chunks."""
        return self._cache is not None

    def in_cache(self, index: Sequence[int]) -> bool:
        """Whether chunk ``index`` is held in the cache now (computed, and not evicted)."""
        return self._cache is not None and (self._key, tuple(int(i) for i in index)) in self._cache

    def chunk_box(self, index: Sequence[int]) -> Box:
        """What chunk ``index`` covers: ``compute_chunk`` returns an array of its shape."""
        return self._info.chunk_box(index)

    def chunk(self, index: Sequence[int]) -> np.ndarray:
        """Chunk ``index`` as an array of shape ``chunk_box(index).shape`` (edge-clipped)."""
        index = tuple(int(i) for i in index)
        if self._cache is None:
            return self._compute_checked(index)
        while True:
            hit = self._cache.get((self._key, index))
            if hit is not None:
                return hit
            with self._inflight_lock:
                event = self._inflight.get(index)
                if event is None:
                    event = self._inflight[index] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                event.wait()
                continue  # the owner has cached it (or failed); re-check the cache
            try:
                data = self._compute_checked(index)
                self._cache.put((self._key, index), data)
                return data
            finally:
                with self._inflight_lock:
                    self._inflight.pop(index, None)
                event.set()

    def _compute_checked(self, index: tuple[int, ...]) -> np.ndarray:
        data = np.ascontiguousarray(self._compute(index))
        expected = self.chunk_box(index).shape
        if data.shape != expected:
            raise ValueError(
                f"{self._key}: compute_chunk{index} returned shape {data.shape}, expected {expected}"
            )
        return data

    def read(self, box: Box) -> np.ndarray:
        info = self._info
        # Fast path: the box is exactly one chunk.
        idx = tuple(a // c for a, c in zip(box.start, info.chunk_shape))
        if info.chunk_box(idx) == box:
            return self.chunk(idx)
        out = np.empty(box.shape, dtype=info.dtype)

        def put(cidx, cbox, inter):  # chunks are disjoint: they may be put at once
            out[inter.relative_to(box).slices()] = self.chunk(cidx)[inter.relative_to(cbox).slices()]

        parts = []
        for cidx in info.chunks_covering(box):
            cbox = info.chunk_box(cidx)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, cbox.start)),
                tuple(min(a, b) for a, b in zip(box.stop, cbox.stop)),
            )
            if not inter.empty:
                parts.append((cidx, cbox, inter))
        self._each(parts, put)
        return out

    def _each(self, parts: list[tuple], fn: Callable) -> list:
        """``fn(*part)`` for each part, whose first item is a chunk index: those whose chunk
        is cached here, the rest computed at once (``demand.gather``)."""
        out: list = [None] * len(parts)
        todo = []
        for i, part in enumerate(parts):
            if self.in_cache(part[0]):
                out[i] = fn(*part)
            else:
                todo.append(i)
        for i, result in zip(todo, demand.gather([partial(fn, *parts[i]) for i in todo])):
            out[i] = result
        return out


class BlockSource(ChunkedSource):
    """Chunks that overlap: block ``index`` is chunk ``index`` of ``info`` and ``overlap``
    voxels around it along the last axes (past the array's edge too, so every block has the
    one shape), each computed from its inputs alone and memoized as a chunk is. ``read``
    blends them: a voxel is the mean of every block reaching it, each weighted 1 over its
    own chunk and less the further into its overlap (``fused.tent``), so the result turns
    smoothly from one block's values to the next's instead of stepping at a chunk's edge,
    and is the same whatever order the blocks are computed in. An integer output is
    rounded after the mean. The blocks of an op with ``overlap``; the leading axes (before
    the last ``len(overlap)``) are read whole."""

    def __init__(
        self,
        info: ArrayInfo,
        overlap: Sequence[int],
        compute_block: Callable[[tuple[int, ...]], np.ndarray],
        cache: LRUCache | None,
        key: str,
    ):
        super().__init__(info, compute_block, cache, key)
        self.overlap = tuple(int(o) for o in overlap)

    def chunk_box(self, index: Sequence[int]) -> Box:
        """Block ``index``'s extent: chunk ``index``, not clipped to the array along the
        last axes, padded by the overlap."""
        info, k = self._info, len(self.overlap)
        lead = info.ndim - k
        whole = info.chunk_box(index)
        core = Box.from_chunk(index, info.chunk_shape)
        return Box(
            whole.start[:lead] + tuple(a - o for a, o in zip(core.start[lead:], self.overlap)),
            whole.stop[:lead] + tuple(b + o for b, o in zip(core.stop[lead:], self.overlap)),
        )

    def read(self, box: Box) -> np.ndarray:
        info, k = self._info, len(self.overlap)
        lead = info.ndim - k
        chunks, grid = info.chunk_shape[lead:], info.chunk_grid[lead:]
        acc = np.zeros(box.shape, np.float64)
        total = np.zeros(box.shape[lead:], np.float64)
        # the blocks whose extent reaches the box
        first = [max((a - o) // c, 0) for a, o, c in zip(box.start[lead:], self.overlap, chunks)]
        last = [min((b - 1 + o) // c, g - 1) for b, o, c, g in zip(box.stop[lead:], self.overlap, chunks, grid)]
        ahead = tuple(a // c for a, c in zip(box.start[:lead], info.chunk_shape[:lead]))
        parts = []
        for rel in np.ndindex(*[hi - lo + 1 for lo, hi in zip(first, last)]):
            index = ahead + tuple(int(lo + r) for lo, r in zip(first, rel))
            extent = self.chunk_box(index)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, extent.start)),
                tuple(min(a, b) for a, b in zip(box.stop, extent.stop)),
            )
            if not inter.empty:
                parts.append((index, extent, inter))
        # the blocks computed at once, then summed here (they overlap)
        pieces = self._each(parts, lambda index, extent, inter: self.chunk(index)[inter.relative_to(extent).slices()])
        for (index, _, inter), data in zip(parts, pieces):
            core = Box.from_chunk(index[lead:], chunks)
            weight = tent(inter.start[lead:], inter.stop[lead:], core.start, core.stop, self.overlap)
            dst = inter.relative_to(box).slices()
            acc[dst] += weight * data
            total[dst[lead:]] += weight
        out = acc / total
        if info.dtype.kind in "iub":
            out = np.rint(out)
        return out.astype(info.dtype)


class MultiscaleSource:
    """An ordered list of ``Source`` levels, s0 = full resolution. ``shader`` is a
    Neuroglancer shader the viewer should show the data with (its channel axis then being
    a shader channel), for sources whose channels mean something particular. ``cached``
    says a pipeline should keep no copy of the levels' chunks in a stage of its own: they
    are cached stages already (another dataset's, ``dataset://``), or must be read afresh
    each time (``coverage://``)."""

    def __init__(
        self, levels: Sequence[Source], name: str = "", shader: str | None = None, cached: bool = False
    ):
        if not levels:
            raise ValueError("need at least one level")
        self.levels = list(levels)
        self.name = name
        self.shader = shader
        self.cached = cached

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, i: int) -> Source:
        return self.levels[i]

    def __iter__(self):
        return iter(self.levels)

    def cache_key(self) -> str:
        return "|".join(lvl.cache_key() for lvl in self.levels)

    def level_for(self, voxel_size: Sequence[float], rtol: float = 0.01) -> tuple[int, bool]:
        """The level to read data at ``voxel_size`` from (its last axes, in the source's
        units), and whether it is at that size: one that is (within ``rtol``), else the
        coarsest finer on every axis (resampled down, never up), else level 0."""
        want = np.asarray(voxel_size, dtype=float)
        best = 0
        for i, lvl in enumerate(self.levels):
            v = np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)
            if np.allclose(v, want, rtol=rtol, atol=0):
                return i, True
            if np.all(v <= want * (1 + rtol)):
                best = i
        return best, False

    def nearest_level(self, voxel_size: Sequence[float], *, fitted: bool = False) -> int:
        """The level whose voxel size (its last axes) is nearest ``voxel_size``, by ratio
        (the sum over axes of the size's log ratio); the finer of two as near. ``fitted``:
        a level computed on finer voxels than its own is matched by those
        (``ArrayInfo.computed_from``), as a lattice's by the image level it was fitted on."""
        want = np.log(np.asarray(voxel_size, dtype=float))

        def vox(info: ArrayInfo) -> np.ndarray:
            if fitted and info.computed_from is not None:
                info = info.computed_from
            return np.asarray(info.voxel_size[-len(want) :], dtype=float)

        dist = [float(np.abs(np.log(vox(lvl.info)) - want).sum()) for lvl in self.levels]
        return min(range(len(dist)), key=lambda i: (round(dist[i], 9), i))
