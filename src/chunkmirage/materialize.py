"""A dataset written to a store as the server serves it: a pipeline's levels, or a box of
them, computed and written as the files a frontend serves (zarr v3 with OME-Zarr 0.5, zarr
v2 with OME-Zarr 0.4, or N5), the metadata and each chunk byte for byte what a client would
fetch, so the store reads back as the dataset did, a stored source of its own. A stage
computed once (a registration's pass, a model's output over a region) can then be read
from disk instead of computed again. The browser engine's ``Engine.materialize``
(``web/src/materialize.ts``) writes a view's region the same way.

The metadata goes first, every level's, so a store whose writing stopped part way is still
the dataset, zero where no chunk was written yet. Chunks are computed several at once
(``demand.gather``), and one of nothing but zeros (the fill value) is left out unless
asked for. The store is anything tensorstore writes: a directory, or a ``file://``,
``gs://`` or ``s3://`` URL.

A box is written as exactly the box (``extent="box"``, the default): each level an array
of the box's shape on it, placed where the box is (its translation moved by the box's
corner), in chunks of the dataset's shape counted from that corner, each read from the
dataset's chunks it overlaps. ``extent="level"`` writes the box's chunks into arrays of the
levels' whole shapes instead, so several boxes can go into one store.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from chunkmirage import demand
from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.frontends import get_frontend
from chunkmirage.frontends.base import Metadata
from chunkmirage.pipeline import Pipeline
from chunkmirage.sources.base import ChunkedSource

FORMATS = ("zarr3", "zarr", "n5")
EXTENTS = ("box", "level")
BATCH = 256  # chunks asked for at once: the pool's work, and what one wait holds
#: most a box's write keeps of the dataset's chunks it stitches from, when its last stage
#: keeps none itself: enough for a slab of them, each computed once; past it, recomputed
BOX_BUFFER_BYTES = 1 << 30


def _kvstore_url(where: str) -> str:
    if "://" not in where:
        where = "file://" + os.path.abspath(where)
    return where if where.endswith("/") else where + "/"


def level_box(pipeline: Pipeline, level: int, box: Box | None) -> Box:
    """``box`` (voxels of level 0 along the last axes it has; the leading ones whole) on
    ``level``'s voxels, rounded out and clipped to the level; the whole level if None."""
    info = pipeline.info(level)
    if box is None:
        return Box((0,) * info.ndim, info.shape)
    k, base = box.ndim, pipeline.info(0)
    if k > info.ndim:
        raise ValueError(f"a box of {k} axes for a dataset of {info.ndim}")
    ratio = [float(b) / float(v) for b, v in zip(base.voxel_size[-k:], info.voxel_size[-k:])]
    lo = [int(np.floor(a * r)) for a, r in zip(box.start, ratio)]
    hi = [int(np.ceil(b * r)) for b, r in zip(box.stop, ratio)]
    lead = info.ndim - k
    return Box((0,) * lead + tuple(lo), info.shape[:lead] + tuple(hi)).clip(info.shape)


class _Cropped:
    """``pipeline`` cut to ``boxes``, one per level, as ``extent="box"`` writes it: each
    level an array of its box's shape whose voxel 0 is the box's first (its translation
    moved there), chunked as the level is, counted from the box's corner. A chunk is read
    from the level's chunks it overlaps, through a cache of them when the level keeps none
    (an uncached last stage), so each is computed once while a slab of them fits."""

    def __init__(self, pipeline: Pipeline, boxes: list[Box]):
        self.pipeline, self.source, self.boxes = pipeline, pipeline.source, boxes
        self._infos: list[ArrayInfo] = []
        self._reads = []
        for level, b in enumerate(boxes):
            info = pipeline.info(level)
            moved = tuple(t + s * v for t, s, v in zip(info.translation, b.start, info.voxel_size))
            self._infos.append(info.with_(shape=b.shape, translation=moved))
            stage = pipeline.levels[level]
            if not isinstance(stage, ChunkedSource) or stage.cached:
                self._reads.append(stage)
            else:
                self._reads.append(ChunkedSource(stage.info, stage.chunk, LRUCache(BOX_BUFFER_BYTES), f"box:{stage.cache_key()}"))

    @property
    def num_levels(self) -> int:
        return self.pipeline.num_levels

    def info(self, level: int = 0) -> ArrayInfo:
        return self._infos[level]

    def chunk(self, level: int, index) -> np.ndarray:
        at, b = self._infos[level].chunk_box(index), self.boxes[level]
        return self._reads[level].read(Box(tuple(s + a for s, a in zip(b.start, at.start)),
                                           tuple(s + a for s, a in zip(b.start, at.stop))))

    def walk(self, level: int) -> list[tuple[int, ...]]:
        """The box's chunks in the order that keeps fewest of the level's open: the axes
        its corner is on the level's chunk edges along first, then those it is not, the one
        with the most chunks outermost (a chunk across an edge waits for the far side's)."""
        info, b = self._infos[level], self.boxes[level]
        grid = info.chunk_grid
        split = [a for a in range(info.ndim) if b.start[a] % info.chunk_shape[a]]
        order = [a for a in range(info.ndim) if a not in split] + sorted(split, key=lambda a: -grid[a])
        return sorted(np.ndindex(*grid), key=lambda idx: tuple(idx[a] for a in order))


def materialize(
    pipeline: Pipeline,
    where: str,
    *,
    format: str = "zarr3",
    levels: Sequence[int] | None = None,
    box: Box | None = None,
    extent: str = "box",
    compressor: str | None = None,
    skip_empty: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Write ``pipeline`` into the store at ``where`` as ``format`` serves it: ``zarr3``
    (OME-Zarr 0.5), ``zarr`` (zarr v2, OME-Zarr 0.4) or ``n5``; ``compressor`` as the
    frontend takes it (blosc by default for zarr, gzip for N5). ``levels``: which levels'
    chunks to write (all by default; every level's metadata is written whatever it is).
    ``box``: voxels of level 0 along its last axes, each level the same region on its own
    voxels (rounded out to whole voxels), written as ``extent`` says: ``box`` (the default)
    an array of exactly the box, placed where it is; ``level`` its chunks (rounded out to
    whole chunks) in an array of the level's whole shape. ``progress`` is told the counts
    after each batch of chunks. Returns ``{chunks, written, skipped, bytes}``."""
    import tensorstore as ts

    from chunkmirage.sources.tensorstore_source import shared_context

    if format not in FORMATS:
        raise ValueError(f"format {format!r}: one of {', '.join(FORMATS)}")
    if extent not in EXTENTS:
        raise ValueError(f"extent {extent!r}: one of {', '.join(EXTENTS)}")
    fe = get_frontend(format, **({} if compressor is None else {"compressor": compressor}))
    kv = ts.KvStore.open(_kvstore_url(where), context=shared_context()).result()  # watched by worldarray
    levels = list(range(pipeline.num_levels)) if levels is None else [int(lv) for lv in levels]
    for lv in levels:
        if not 0 <= lv < pipeline.num_levels:
            raise ValueError(f"no level {lv}: the dataset has {pipeline.num_levels}")
    source = pipeline
    if box is not None and extent == "box":
        boxes = [level_box(pipeline, lv, box) for lv in range(pipeline.num_levels)]
        for lv, b in enumerate(boxes):
            if b.empty:
                raise ValueError(f"the box {box} holds no voxel of level {lv}, which is {pipeline.info(lv).shape}")
        source = _Cropped(pipeline, boxes)

    # the metadata first: from then on the store is the dataset, zero where nothing is yet
    keys = [*fe.root_keys, *(f"s{i}/{k}" for i in range(pipeline.num_levels) for k in fe.level_keys)]
    for key in keys:
        meta = fe.resolve(source, key)
        if isinstance(meta, Metadata):
            kv.write(key, meta.body).result()

    counts = {"chunks": 0, "written": 0, "skipped": 0, "bytes": 0}

    def one(level: int, index: tuple[int, ...]) -> int | None:
        """Chunk ``index`` of ``level`` computed and written: its size, or None if empty."""
        block = source.chunk(level, index)
        if skip_empty and not block.any():
            return None
        data = fe.encode(source.info(level), index, block)
        kv.write(fe.chunk_key(level, index), data).result()
        return len(data)

    for level in levels:
        if isinstance(source, _Cropped):
            todo = source.walk(level)
        else:
            todo = list(pipeline.info(level).chunks_covering(level_box(pipeline, level, box)))
        counts["chunks"] += len(todo)
        for start in range(0, len(todo), BATCH):
            for size in demand.gather([partial(one, level, idx) for idx in todo[start : start + BATCH]]):
                if size is None:
                    counts["skipped"] += 1
                else:
                    counts["written"] += 1
                    counts["bytes"] += size
            if progress is not None:
                progress({"level": level, **counts})
    return counts
