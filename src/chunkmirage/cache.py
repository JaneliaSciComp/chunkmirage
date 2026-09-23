"""Byte-bounded LRU cache for numpy chunks (and encoded bytes)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


def _nbytes(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return 1


class LRUCache(Generic[T]):
    """Thread-safe LRU keyed by any hashable, bounded by total bytes.

    A single shared instance is normally used for every pipeline stage; keys embed the
    stage hash so that changing a parameter downstream never evicts upstream entries
    except through normal LRU pressure.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3):
        self.max_bytes = int(max_bytes)
        self._data: OrderedDict[Hashable, T] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> T | None:
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                self.misses += 1
                return None
            self._data[key] = value
            self.hits += 1
            return value

    def put(self, key: Hashable, value: T) -> None:
        size = _nbytes(value)
        if size > self.max_bytes:
            return
        with self._lock:
            if key in self._data:
                self._bytes -= _nbytes(self._data.pop(key))
            self._data[key] = value
            self._bytes += size
            while self._bytes > self.max_bytes and self._data:
                _, evicted = self._data.popitem(last=False)
                self._bytes -= _nbytes(evicted)

    def invalidate(self, prefix: Hashable | None = None) -> int:
        """Drop entries; if ``prefix`` is given, only tuple keys starting with it."""
        with self._lock:
            if prefix is None:
                n = len(self._data)
                self._data.clear()
                self._bytes = 0
                return n
            doomed = [k for k in self._data if isinstance(k, tuple) and k[:1] == (prefix,)]
            for k in doomed:
                self._bytes -= _nbytes(self._data.pop(k))
            return len(doomed)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "entries": len(self._data),
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
