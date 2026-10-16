"""``dataset://`` and ``coverage://`` sources: another dataset of the same server, read
in-process, and what its cache holds.

    dataset://<name>
    coverage://<name>

A ``dataset://`` source's levels are the named dataset's own output stages, so a chunk is
read through that dataset's stage cache, never over HTTP and never computed twice, and
every stage key downstream holds its keys: an edit to the named dataset is an edit to
everything built on it, which the registry then rebuilds (``DatasetRegistry.add``). A chain
of such datasets (a field, the image warped by it, the next pass fitted on that, its field,
...) costs one read per stage. A dataset may not read itself by any chain: a cycle is
refused when the dataset is added.

A ``coverage://`` source has one voxel per chunk of the named dataset's last cached stage
at each level, saying what the cache holds there (``CoverageLevel``), read afresh at every
request: for telling, while browsing, a result computed at full resolution from the coarse
preview a zoomed-out view shows.

Both resolve against the registry that is building the dataset holding them
(``building``), so they are opened through a ``DatasetRegistry`` or the REST API, not by
``open_source`` on its own.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import ChunkedSource, MultiscaleSource, Source

if TYPE_CHECKING:
    from chunkmirage.pipeline import Pipeline
    from chunkmirage.server import DatasetRegistry

# while a registry builds a dataset: the registry, the names being built (the outermost
# first: a dataset a resolver builds while another is being built sees the whole chain),
# and the set the names it reads are collected into
_BUILDING: contextvars.ContextVar[tuple[DatasetRegistry, tuple[str, ...], set[str]] | None] = (
    contextvars.ContextVar("chunkmirage.dataset.building", default=None)
)


@contextmanager
def building(registry: DatasetRegistry, name: str) -> Iterator[set[str]]:
    """Within it, ``dataset://`` and ``coverage://`` URLs resolve against ``registry`` and
    name the datasets ``name`` reads into the set yielded (its upstreams, for the registry
    to rebuild it when one of them changes)."""
    outer = _BUILDING.get()
    chain = (outer[1] if outer else ()) + (name,)
    upstreams: set[str] = set()
    token = _BUILDING.set((registry, chain, upstreams))
    try:
        yield upstreams
    finally:
        _BUILDING.reset(token)


def _resolve(url: str, scheme: str) -> tuple[str, Pipeline]:
    """The dataset ``url`` names (``<scheme>://<name>``) in the registry building the
    dataset that holds it, noted as one it reads; or why there is none."""
    name = url[len(scheme) + 3 :].strip("/")
    if not name:
        raise ValueError(f"{scheme}:// needs a name: {scheme}://<name>")
    ctx = _BUILDING.get()
    if ctx is None:
        raise ValueError(
            f"{url}: {scheme}:// reads another dataset of the server building this one, so it "
            "is opened through a DatasetRegistry (or the REST API), not on its own"
        )
    registry, chain, upstreams = ctx
    # a cycle: the dataset being built, or one being built around it, is this one or
    # reads it (through datasets already registered)
    for n in chain:
        if n == name or n in registry.upstreams(name):
            raise ValueError(f"{url}: a cycle: {' -> '.join((*chain, name))}")
    pipeline = registry.resolve(name)
    if pipeline is None:
        raise ValueError(f"{url}: no dataset named {name!r}")
    upstreams.add(name)
    return name, pipeline


def open_dataset(url: str) -> MultiscaleSource:
    name, pipeline = _resolve(url, "dataset")
    return MultiscaleSource(list(pipeline.levels), name=name, cached=True)


class CoverageLevel(Source):
    """One voxel per chunk of ``stage``, a level's last cached stage of a dataset, saying
    what the cache holds there now: bit 1 when that chunk is in the cache, bit 2 when every
    chunk of ``fine``, the finest level's stage, under it is. So 3 is a chunk computed whose
    full-resolution result exists, 1 a preview (the ops run on this coarse level, the fine
    chunks under it not all computed), 2 the fine result alone, 0 nothing. Served as labels,
    read afresh at every request and never cached."""

    def __init__(self, stage: ChunkedSource, fine: ChunkedSource):
        self.stage, self.fine = stage, fine
        info = stage.info
        grid = info.chunk_grid
        # a voxel is a chunk: the chunk's size, placed at the chunk's centre (voxel
        # positions are centres)
        self._info = ArrayInfo(
            shape=grid,
            dtype=np.dtype("uint8"),
            chunk_shape=grid,
            voxel_size=tuple(v * c for v, c in zip(info.voxel_size, info.chunk_shape)),
            units=info.units,
            axes=info.axes,
            translation=tuple(t + (c - 1) / 2 * v for t, c, v in zip(info.translation, info.chunk_shape, info.voxel_size)),
            kind="label",
        )
        self._key = f"coverage:{stage.cache_key()}:{fine.cache_key()}"

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def _fine_chunks(self, index: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        """The chunks of the finest level's stage under chunk ``index`` of this one."""
        s, f = self.stage.info, self.fine.info
        box = s.chunk_box(index)
        # the chunk's edges in physical units (a voxel's position is its centre), then the
        # fine voxels they fall in
        lo = [t + (a - 0.5) * v for a, t, v in zip(box.start, s.translation, s.voxel_size)]
        hi = [t + (b - 0.5) * v for b, t, v in zip(box.stop, s.translation, s.voxel_size)]
        start = [int(np.floor((p - t) / v + 0.5 + 1e-6)) for p, t, v in zip(lo, f.translation, f.voxel_size)]
        stop = [int(np.ceil((p - t) / v + 0.5 - 1e-6)) for p, t, v in zip(hi, f.translation, f.voxel_size)]
        return f.chunks_covering(Box(tuple(start), tuple(stop)).clip(f.shape))

    def _complete(self, index: tuple[int, ...]) -> bool:
        if self.fine is self.stage:
            return self.stage.in_cache(index)
        return all(self.fine.in_cache(i) for i in self._fine_chunks(index))

    def read(self, box: Box) -> np.ndarray:
        out = np.zeros(box.shape, np.uint8)
        for rel in np.ndindex(*box.shape):
            index = tuple(int(a + r) for a, r in zip(box.start, rel))
            out[rel] = (1 if self.stage.in_cache(index) else 0) | (2 if self._complete(index) else 0)
        return out


def _computing(pipeline: Pipeline, level: int) -> ChunkedSource | None:
    """The stage of ``level`` whose chunks say what was computed there: the last cached one,
    an op with an overlap by its blocks (where the op ran); None if none is cached."""
    for stage in reversed(pipeline.stages[level]):
        src = stage.blocks or stage
        if src.cached:
            return src
    return None


def coverage_levels(pipeline: Pipeline, name: str = "") -> list[CoverageLevel]:
    """A ``CoverageLevel`` per level of ``pipeline``; or why there is none."""
    stages = [_computing(pipeline, i) for i in range(pipeline.num_levels)]
    if any(s is None for s in stages):
        raise ValueError(
            f"nothing of {name or 'the dataset'!s} is cached (its stages are computed at each "
            "request), so no chunk of it is ever computed: look at the datasets it reads"
        )
    return [CoverageLevel(s, stages[0]) for s in stages]  # type: ignore[arg-type]


def open_coverage(url: str) -> MultiscaleSource:
    name, pipeline = _resolve(url, "coverage")
    try:
        levels = coverage_levels(pipeline, repr(name))
    except ValueError as e:
        raise ValueError(f"{url}: {e}") from None
    return MultiscaleSource(levels, name=f"{name}~coverage", cached=True)


def coverage_report(pipeline: Pipeline, name: str = "") -> dict:
    """What the cache holds of ``pipeline``'s last cached stage, per level: the chunk grid,
    how many chunks are ``computed`` and how many are ``complete`` (every finest-level chunk
    under them computed); ``stage`` names it (``GET /api/datasets/{name}/coverage``)."""
    levels = coverage_levels(pipeline, name)
    out = []
    for lvl in levels:
        values = lvl.read(Box((0,) * lvl.info.ndim, lvl.info.shape))
        out.append(
            {
                "chunks": list(lvl.info.shape),
                "computed": int((values & 1).sum()),
                "complete": int((values & 2).sum() // 2),
            }
        )
    return {"stage": levels[0].stage.cache_key().rsplit(":", 1)[0], "levels": out}
