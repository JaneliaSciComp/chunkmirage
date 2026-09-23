"""Pipeline: a multiscale source followed by ops, exposed level-by-level as ChunkedSources.

Stage *k* is a ``ChunkedSource`` whose ``compute_chunk(idx)`` reads the halo-padded box
from stage *k-1* (itself cached if requested), applies op *k*, and crops. Cache keys are
``(stage_hash, chunk_index)`` where ``stage_hash`` covers the source identity and every op
up to and including *k*, so editing op *k* invalidates exactly stages *k..n*.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op, ops_from_specs
from chunkmirage.sources.base import ChunkedSource, MultiscaleSource, Source


class PipelineSpec(BaseModel):
    """JSON-serializable description of a pipeline (what the REST API and CLI exchange)."""

    source: str
    ops: list[dict[str, Any]] = Field(default_factory=list)
    chunk_shape: list[int] | None = None
    cache_source: bool = True
    voxel_size: list[float] | None = None
    units: list[str] | None = None
    axes: list[str] | None = None
    translation: list[float] | None = None


def _stage(prev: Source, op: Op, cache: LRUCache | None, key: str, chunk_shape) -> ChunkedSource:
    info = op.output_info(prev.info).with_(chunk_shape=tuple(chunk_shape))
    halo = op.halo_for(info.ndim)

    def compute(idx: tuple[int, ...]) -> np.ndarray:
        out_box = info.chunk_box(idx)
        in_box = out_box.pad(halo)
        block = prev.read_padded(in_box)
        result = op.apply(block)
        if result.shape[-info.ndim :] != block.shape:
            raise ValueError(f"op {op.name!r} changed block shape {block.shape} -> {result.shape}")
        crop = out_box.relative_to(in_box)
        return np.asarray(result[crop.slices()], dtype=info.dtype)

    return ChunkedSource(info, compute, cache if op.cache else None, key)


class Pipeline:
    def __init__(
        self,
        source: MultiscaleSource,
        ops: Sequence[Op | dict] = (),
        *,
        cache: LRUCache | None = None,
        chunk_shape: Sequence[int] | None = None,
        cache_source: bool = True,
        spec: PipelineSpec | None = None,
    ):
        self.source = source
        self.ops: list[Op] = ops_from_specs(ops)
        self.cache = cache if cache is not None else LRUCache()
        self.spec = spec
        self.levels: list[ChunkedSource] = []
        for lvl_i, raw in enumerate(source.levels):
            cs = tuple(chunk_shape) if chunk_shape else raw.info.chunk_shape
            h = hashlib.sha1(f"{raw.cache_key()}|{lvl_i}|{cs}".encode()).hexdigest()[:12]
            # Stage 0: the raw source, re-chunked to `cs` and (optionally) cached.
            stage: Source = ChunkedSource(
                raw.info.with_(chunk_shape=cs),
                lambda idx, raw=raw, cs=cs: raw.read(raw.info.with_(chunk_shape=cs).chunk_box(idx)),
                self.cache if cache_source else None,
                f"raw:{h}",
            )
            for op in self.ops:
                h = hashlib.sha1(f"{h}|{op.digest()}".encode()).hexdigest()[:12]
                stage = _stage(stage, op, self.cache, f"{op.name}:{h}", cs)
            self.levels.append(stage)  # type: ignore[arg-type]

    @classmethod
    def from_spec(
        cls,
        spec: PipelineSpec | dict,
        *,
        cache: LRUCache | None = None,
        source_cache_bytes: int = 0,
    ) -> Pipeline:
        from chunkmirage.sources import open_source

        spec = spec if isinstance(spec, PipelineSpec) else PipelineSpec.model_validate(spec)
        src = open_source(
            spec.source,
            cache_bytes=source_cache_bytes,
            voxel_size=spec.voxel_size,
            units=spec.units,
            axes=spec.axes,
            translation=spec.translation,
        )
        return cls(
            src,
            spec.ops,
            cache=cache,
            chunk_shape=spec.chunk_shape,
            cache_source=spec.cache_source,
            spec=spec,
        )

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def info(self, level: int = 0) -> ArrayInfo:
        return self.levels[level].info

    def chunk(self, level: int, index: Sequence[int]) -> np.ndarray:
        """Compute (or fetch cached) output chunk ``index`` at scale ``level``."""
        return self.levels[level].chunk(index)

    def read(self, level: int, box: Box) -> np.ndarray:
        return self.levels[level].read(box)

    def digest(self) -> str:
        return hashlib.sha1("|".join(lvl.cache_key() for lvl in self.levels).encode()).hexdigest()[
            :12
        ]
