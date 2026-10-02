"""Pipeline: a multiscale source followed by ops, exposed level-by-level as ChunkedSources.

Ops are grouped into segments ending at each ``cache=True`` op. A segment is one
``ChunkedSource`` stage whose ``compute_chunk(idx)`` reads the box padded by the segment's
total halo from the previous stage, runs its ops back to back, and crops. Cache keys are
``(stage_hash, chunk_index)`` where ``stage_hash`` covers the source identity and every op
up to and including *k*, so editing op *k* invalidates exactly stages *k..n*.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from chunkmirage import fused
from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.meshes import MeshSpec
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
    select: dict[str, int] | None = Field(
        None,
        description='Pin non-spatial axes to one index each, e.g. {"c": 1, "t": 0}: the '
        "pipeline sees that channel of that time point, without those axes",
    )
    mesh: MeshSpec | None = Field(
        None, description="What the dataset's mesh frontend meshes (default: a surface at 128)"
    )


class SelectSource(Source):
    """``inner`` with some axes pinned to one index each and dropped: one channel of a
    multichannel image, one time point of a series. Reads only that index."""

    def __init__(self, inner: Source, pinned: dict[int, int]):
        self.inner = inner
        self.pinned = pinned  # axis -> index
        i = inner.info
        keep = [a for a in range(i.ndim) if a not in pinned]
        self._keep = keep
        self._info = ArrayInfo(
            shape=tuple(i.shape[a] for a in keep),
            dtype=i.dtype,
            chunk_shape=tuple(i.chunk_shape[a] for a in keep),
            voxel_size=tuple(i.voxel_size[a] for a in keep),
            units=tuple(i.units[a] for a in keep),
            axes=tuple(i.axes[a] for a in keep),
            translation=tuple(i.translation[a] for a in keep),
        )
        tag = ",".join(f"{i.axes[a]}={v}" for a, v in sorted(pinned.items()))
        self._key = f"select:{tag}:{inner.cache_key()}"

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        start, stop, k = [], [], 0
        for a in range(self.inner.info.ndim):
            if a in self.pinned:
                start.append(self.pinned[a])
                stop.append(self.pinned[a] + 1)
            else:
                start.append(box.start[k])
                stop.append(box.stop[k])
                k += 1
        return self.inner.read(Box(tuple(start), tuple(stop))).reshape(box.shape)


def select_axes(source: MultiscaleSource, select: dict[str, int]) -> MultiscaleSource:
    """``source`` with the axes named in ``select`` pinned to those indices, every level."""
    info = source.levels[0].info
    pinned = {}
    for name, index in select.items():
        if name not in info.axes:
            raise ValueError(f"select {name}={index}: the source has axes {info.axes}")
        a = info.axes.index(name)
        if name in ("z", "y", "x"):
            raise ValueError(f"select {name}={index}: only non-spatial axes can be pinned")
        if not 0 <= index < info.shape[a]:
            raise ValueError(f"select {name}={index}: the {name} axis has {info.shape[a]} entries")
        pinned[a] = int(index)
    levels = [SelectSource(lvl, pinned) for lvl in source.levels]
    return MultiscaleSource(levels, name=source.name)


def _fused_stage(
    prev: Source, ops: Sequence[Op], cache: LRUCache | None, key: str, chunk_shape
) -> ChunkedSource:
    """One pipeline stage running ``ops`` back to back on a block padded by their total halo.

    Fusing consecutive uncached ops keeps the read footprint small: one output chunk reads
    the upstream box once, padded by the *sum* of the halos, instead of pulling overlapping
    chunks through a separate grid per op (which compounds to hundreds of upstream chunks).
    Values near the padded border are wrong after each op, but the padding is exactly the
    sum of halos so the cropped centre is correct.
    """
    prev_info = prev.info
    # An op may consume leading axes (the channels of a stack:// source, for `contacts`):
    # those are read whole, and the halo pads the axes the output keeps (chunkmirage.fused,
    # which the browser engine runs too).
    info, lead, total_halo = fused.plan(prev_info, ops)
    info = info.with_(chunk_shape=tuple(chunk_shape)[len(chunk_shape) - info.ndim :])

    def compute(idx: tuple[int, ...]) -> np.ndarray:
        out_box = info.chunk_box(idx)
        in_box = fused.input_box(prev_info, out_box, lead, total_halo)
        block = prev.read_padded(in_box, edge=True)  # no step at the volume border
        return fused.run(ops, block, in_box, out_box, info)

    return ChunkedSource(info, compute, cache, key)


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
            if len(cs) < raw.info.ndim:  # spatial chunks given for a source with leading axes
                cs = raw.info.chunk_shape[: raw.info.ndim - len(cs)] + cs
            h = hashlib.sha1(f"{raw.cache_key()}|{lvl_i}|{cs}".encode()).hexdigest()[:12]
            # Stage 0: the raw source, re-chunked to `cs` and (optionally) cached.
            stage: Source = ChunkedSource(
                raw.info.with_(chunk_shape=cs),
                lambda idx, raw=raw, cs=cs: raw.read(raw.info.with_(chunk_shape=cs).chunk_box(idx)),
                self.cache if cache_source else None,
                f"raw:{h}",
            )
            # Group ops into segments; a segment ends at an op with cache=True (its output is
            # memoized) or at the end of the pipeline. Each segment is one fused stage.
            segments: list[list[Op]] = []
            current: list[Op] = []
            for op in self.ops:
                current.append(op)
                if op.cached:
                    segments.append(current)
                    current = []
            if current:
                segments.append(current)
            for seg in segments:
                for op in seg:
                    h = hashlib.sha1(f"{h}|{op.digest()}".encode()).hexdigest()[:12]
                stage = _fused_stage(
                    stage, seg, self.cache if seg[-1].cached else None, f"{seg[-1].name}:{h}", cs
                )
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
        cache = cache if cache is not None else LRUCache()  # the stages' and the source's
        src = open_source(
            spec.source,
            cache_bytes=source_cache_bytes,
            cache=cache,
            voxel_size=spec.voxel_size,
            units=spec.units,
            axes=spec.axes,
            translation=spec.translation,
        )
        if spec.select:
            src = select_axes(src, spec.select)
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
