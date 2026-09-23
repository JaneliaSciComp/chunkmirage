"""Data sources: anything that can describe itself as an ``ArrayInfo`` and read a ``Box``."""

from chunkmirage.sources.base import ChunkedSource, MultiscaleSource, Source
from chunkmirage.sources.registry import open_source
from chunkmirage.sources.tensorstore_source import TensorStoreSource, open_tensorstore

__all__ = [
    "ChunkedSource",
    "MultiscaleSource",
    "Source",
    "TensorStoreSource",
    "open_source",
    "open_tensorstore",
]
