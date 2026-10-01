from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from chunkmirage.core import ArrayInfo
from chunkmirage.pipeline import Pipeline

JSON_CT = "application/json"
BIN_CT = "application/octet-stream"


@dataclass
class Metadata:
    body: bytes
    content_type: str = JSON_CT

    @staticmethod
    def json(obj) -> Metadata:
        return Metadata(json.dumps(obj, indent=1).encode(), JSON_CT)


@dataclass
class ChunkRequest:
    level: int
    index: tuple[int, ...]
    # a byte range [start, stop) of a file of `total` bytes (multi-resolution mesh fragments)
    part: tuple[int, int] | None = None
    total: int | None = None


class Frontend(ABC):
    name: ClassVar[str]
    neuroglancer_scheme: ClassVar[str]  # e.g. "n5" -> n5://http://...

    @abstractmethod
    def resolve(self, pipeline: Pipeline, path: str) -> Metadata | ChunkRequest | None:
        """Map a request path (relative to the dataset+format prefix) to metadata or a chunk."""

    @abstractmethod
    def encode(self, info: ArrayInfo, index: tuple[int, ...], block: np.ndarray) -> bytes:
        """Encode the (edge-clipped) block for chunk ``index`` in this format."""

    def resolve_range(self, pipeline: Pipeline, path: str, start: int, stop: int) -> ChunkRequest | None:
        """A request for bytes ``[start, stop)`` of ``path``, for files read by HTTP Range
        requests; ``None`` serves the whole file as ``resolve`` does."""
        return None

    def compute(self, pipeline: Pipeline, req: ChunkRequest) -> bytes:
        """The body for ``req``: its chunk, encoded. Frontends that serve something made from
        the data rather than its chunks (meshes) override this."""
        return self.encode(
            pipeline.info(req.level), req.index, pipeline.chunk(req.level, req.index)
        )

    root_keys: ClassVar[tuple[str, ...]] = ()  # metadata keys of the group
    level_keys: ClassVar[tuple[str, ...]] = ()  # metadata keys of each level

    def listing(self, pipeline: Pipeline, path: str) -> list[str] | None:
        """What a directory listing of ``path`` shows: the group's metadata keys and one
        directory per level, or a level's metadata keys (never its chunks); None if ``path``
        is neither, or the format has no directories to list."""
        path = path.strip("/")
        if not self.root_keys:
            return None
        if path == "":
            return [*self.root_keys, *(f"s{i}/" for i in range(pipeline.num_levels))]
        m = re.fullmatch(r"s(\d+)", path)
        if m and int(m.group(1)) < pipeline.num_levels:
            return list(self.level_keys)
        return None

    # Helpers shared by frontends -----------------------------------------------------
    @staticmethod
    def relative_scales(pipeline: Pipeline) -> list[list[float]]:
        """Downsampling factor of every level relative to s0, per C-order axis."""
        base = pipeline.info(0).voxel_size
        out = []
        for lvl in range(pipeline.num_levels):
            vs = pipeline.info(lvl).voxel_size
            out.append([v / b if b else 1.0 for v, b in zip(vs, base)])
        return out

    @staticmethod
    def pad_to_full(
        info: ArrayInfo, index: tuple[int, ...], block: np.ndarray, fill=0
    ) -> np.ndarray:
        """Zarr requires full-size chunks even at the edge; pad with ``fill``."""
        if block.shape == info.chunk_shape:
            return block
        out = np.full(info.chunk_shape, fill, dtype=block.dtype)
        out[tuple(slice(0, s) for s in block.shape)] = block
        return out


FRONTENDS: dict[str, type[Frontend]] = {}


def register_frontend(cls: type[Frontend]) -> type[Frontend]:
    FRONTENDS[cls.name] = cls
    return cls


def get_frontend(name: str, **kw) -> Frontend:
    try:
        return FRONTENDS[name](**kw)
    except KeyError:
        raise KeyError(f"unknown frontend {name!r}; known: {sorted(FRONTENDS)}") from None
