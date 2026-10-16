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
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from chunkmirage import demand
from chunkmirage.core import Box
from chunkmirage.frontends import get_frontend
from chunkmirage.frontends.base import Metadata
from chunkmirage.pipeline import Pipeline

FORMATS = ("zarr3", "zarr", "n5")
BATCH = 256  # chunks asked for at once: the pool's work, and what one wait holds


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


def materialize(
    pipeline: Pipeline,
    where: str,
    *,
    format: str = "zarr3",
    levels: Sequence[int] | None = None,
    box: Box | None = None,
    compressor: str | None = None,
    skip_empty: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Write ``pipeline`` into the store at ``where`` as ``format`` serves it: ``zarr3``
    (OME-Zarr 0.5), ``zarr`` (zarr v2, OME-Zarr 0.4) or ``n5``; ``compressor`` as the
    frontend takes it (blosc by default for zarr, gzip for N5). ``levels``: which levels'
    chunks to write (all by default; every level's metadata is written whatever it is).
    ``box``: voxels of level 0 along its last axes, each level the same region on its own
    voxels, rounded out to whole chunks. ``progress`` is told the counts after each batch
    of chunks. Returns ``{chunks, written, skipped, bytes}``."""
    import tensorstore as ts

    if format not in FORMATS:
        raise ValueError(f"format {format!r}: one of {', '.join(FORMATS)}")
    fe = get_frontend(format, **({} if compressor is None else {"compressor": compressor}))
    kv = ts.KvStore.open(_kvstore_url(where)).result()
    levels = list(range(pipeline.num_levels)) if levels is None else [int(lv) for lv in levels]
    for lv in levels:
        if not 0 <= lv < pipeline.num_levels:
            raise ValueError(f"no level {lv}: the dataset has {pipeline.num_levels}")

    # the metadata first: from then on the store is the dataset, zero where nothing is yet
    keys = [*fe.root_keys, *(f"s{i}/{k}" for i in range(pipeline.num_levels) for k in fe.level_keys)]
    for key in keys:
        meta = fe.resolve(pipeline, key)
        if isinstance(meta, Metadata):
            kv.write(key, meta.body).result()

    counts = {"chunks": 0, "written": 0, "skipped": 0, "bytes": 0}

    def one(level: int, index: tuple[int, ...]) -> int | None:
        """Chunk ``index`` of ``level`` computed and written: its size, or None if empty."""
        block = pipeline.chunk(level, index)
        if skip_empty and not block.any():
            return None
        data = fe.encode(pipeline.info(level), index, block)
        kv.write(fe.chunk_key(level, index), data).result()
        return len(data)

    for level in levels:
        info = pipeline.info(level)
        todo = list(info.chunks_covering(level_box(pipeline, level, box)))
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
