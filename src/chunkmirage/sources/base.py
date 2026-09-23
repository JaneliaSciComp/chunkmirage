from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box


class Source(ABC):
    """Read-only, random-access array. Everything upstream of an op is a ``Source``."""

    @property
    @abstractmethod
    def info(self) -> ArrayInfo: ...

    @abstractmethod
    def read(self, box: Box) -> np.ndarray:
        """Return data for ``box`` (must lie within ``info.shape``), shape == box.shape."""

    def read_padded(self, box: Box, fill=0) -> np.ndarray:
        """Read ``box`` even if it pokes outside the array; out-of-range voxels are ``fill``."""
        clipped = box.clip(self.info.shape)
        if clipped == box:
            return self.read(box)
        out = np.full(box.shape, fill, dtype=self.info.dtype)
        if not clipped.empty:
            out[clipped.relative_to(box).slices()] = self.read(clipped)
        return out

    def cache_key(self) -> str:
        """Stable identity used to build cache keys for downstream stages."""
        return f"{type(self).__name__}:{id(self)}"


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

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def chunk(self, index: Sequence[int]) -> np.ndarray:
        """Full (edge-clipped) chunk ``index`` as an array of shape ``info.chunk_box(index).shape``."""
        index = tuple(int(i) for i in index)
        if self._cache is not None:
            hit = self._cache.get((self._key, index))
            if hit is not None:
                return hit
        data = np.ascontiguousarray(self._compute(index))
        expected = self._info.chunk_box(index).shape
        if data.shape != expected:
            raise ValueError(
                f"{self._key}: compute_chunk{index} returned shape {data.shape}, expected {expected}"
            )
        if self._cache is not None:
            self._cache.put((self._key, index), data)
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
    """An ordered list of ``Source`` levels, s0 = full resolution."""

    def __init__(self, levels: Sequence[Source], name: str = ""):
        if not levels:
            raise ValueError("need at least one level")
        self.levels = list(levels)
        self.name = name

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, i: int) -> Source:
        return self.levels[i]

    def __iter__(self):
        return iter(self.levels)

    def cache_key(self) -> str:
        return "|".join(lvl.cache_key() for lvl in self.levels)
