"""Frontends: make a pipeline look like a real on-disk dataset in a given format.

Each frontend knows (a) which paths are metadata and what JSON to emit, (b) how to parse a
chunk path into ``(level, chunk_index)``, and (c) how to encode a numpy block as a chunk.
"""

from chunkmirage.frontends.base import FRONTENDS, ChunkRequest, Frontend, Metadata, get_frontend
from chunkmirage.frontends.mesh import MeshFrontend
from chunkmirage.frontends.n5 import N5Frontend
from chunkmirage.frontends.precomputed import PrecomputedFrontend
from chunkmirage.frontends.zarr2 import Zarr2Frontend
from chunkmirage.frontends.zarr3 import Zarr3Frontend

__all__ = [
    "FRONTENDS",
    "ChunkRequest",
    "Frontend",
    "MeshFrontend",
    "Metadata",
    "N5Frontend",
    "PrecomputedFrontend",
    "Zarr2Frontend",
    "Zarr3Frontend",
    "get_frontend",
]
