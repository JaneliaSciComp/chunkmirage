from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import KINDS, ArrayInfo, Box


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
        # In-flight computations keyed by chunk index: concurrent requests for the same chunk
        # (a viewer retrying, or two clients) wait for one computation instead of duplicating it.
        self._inflight: dict[tuple[int, ...], threading.Event] = {}
        self._inflight_lock = threading.Lock()

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def chunk(self, index: Sequence[int]) -> np.ndarray:
        """Full (edge-clipped) chunk ``index`` as an array of shape ``info.chunk_box(index).shape``."""
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
        expected = self._info.chunk_box(index).shape
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
        for cidx in info.chunks_covering(box):
            cbox = info.chunk_box(cidx)
            inter = Box(
                tuple(max(a, b) for a, b in zip(box.start, cbox.start)),
                tuple(min(a, b) for a, b in zip(box.stop, cbox.stop)),
            )
            if inter.empty:
                continue
            data = self.chunk(cidx)
            out[inter.relative_to(box).slices()] = data[inter.relative_to(cbox).slices()]
        return out


class MultiscaleSource:
    """An ordered list of ``Source`` levels, s0 = full resolution. ``shader`` is a
    Neuroglancer shader the viewer should show the data with (its channel axis then being
    a shader channel), for sources whose channels mean something particular."""

    def __init__(self, levels: Sequence[Source], name: str = "", shader: str | None = None):
        if not levels:
            raise ValueError("need at least one level")
        self.levels = list(levels)
        self.name = name
        self.shader = shader

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

    def nearest_level(self, voxel_size: Sequence[float]) -> int:
        """The level whose voxel size (its last axes) is nearest ``voxel_size``, by ratio
        (the sum over axes of the size's log ratio); the finer of two as near."""
        want = np.log(np.asarray(voxel_size, dtype=float))
        dist = [
            float(np.abs(np.log(np.asarray(lvl.info.voxel_size[-len(want) :], dtype=float)) - want).sum())
            for lvl in self.levels
        ]
        return min(range(len(dist)), key=lambda i: (round(dist[i], 9), i))
