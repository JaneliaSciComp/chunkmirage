"""``stitch://`` sources: the tiles of a BigStitcher project stitched by interest points and
fused, region by region as they are read; no fused copy is written.

    stitch://<project.xml>?channel=0&model=translation&epsilon=5&threshold=0.005

When opened, each overlap of two tiles' stage positions (grown by ``margin``) is read at
``level``, its interest points found, matched between the two tiles and filtered by RANSAC,
and every tile's placement fitted to the kept matches (``chunkmirage.stitching``; the
parameters are ``StitchParams``). Each level of the fused volume then has the tiles' voxel
size at that level and covers them all; a region of it is each tile that reaches it sampled
where its placement puts it, blended at the seams. The project's images must be OME-Zarr
(BigStitcher-Spark's ``bdv.multimg.zarr``); the outermost transforms named "Stitching
Transform" (a stitching done before) are left out of the stage positions, and the result is
compared with them in the log.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from urllib.parse import parse_qs

import numpy as np

from chunkmirage import stitching as S
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

log = logging.getLogger("chunkmirage")
CHUNK = (64, 128, 128)


class FusedLevel(Source):
    """One level of the fused tiles."""

    def __init__(self, tiles, sources, placements, level: int, grid: dict, info: ArrayInfo,
                 blend: float, key: str):
        self.tiles, self.sources, self.placements = tiles, sources, placements
        self.level, self.grid, self._info, self.blend, self._key = level, grid, info, blend, key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        lo, hi = S.scene_box(self.grid, box.start, box.stop)
        blocks = []
        for t, tile in enumerate(self.tiles):
            r = S.region(tile, self.level, lo, hi, self.placements[t])
            if r is None:
                blocks.append(None)
                continue
            start, stop = r
            blocks.append((self.sources[t][self.level].read(Box(tuple(start), tuple(stop))), start))
        return S.fuse(self.tiles, self.placements, self.level, self.grid, box.start, box.stop,
                      blocks, self.blend, self._info.dtype)


def _fetch(url: str) -> str:
    if "://" not in url or url.startswith("file://"):
        with open(url.removeprefix("file://")) as f:
            return f.read()
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def stitch(tiles: list[dict], sources: list[MultiscaleSource], p: S.StitchParams) -> dict:
    """Interest points in every overlap at ``p.level``, then ``stitching.register``."""
    points = []
    for i, j, lo, hi in S.overlaps(tiles, p.margin):
        both = []
        for t in (i, j):
            r = S.region(tiles[t], p.level, lo, hi)
            if r is None:
                both.append(np.zeros((0, 3)))
                continue
            level = sources[t][p.level]
            block = level.read(Box(tuple(r[0]), tuple(r[1])))
            both.append(S.points_in(tiles[t], p.level, block, r[0], level.info.voxel_size, lo, hi, p))
        points.append(both)
    return S.register(tiles, points, p)


def open_stitch(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.pipeline import select_axes
    from chunkmirage.sources.registry import open_source

    location, _, query = url[len("stitch://") :].rpartition("?")
    if not location:
        location, query = query, ""
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    p = S.StitchParams.from_query(q)
    t0 = time.perf_counter()
    tiles = S.tiles_from_bdv(_fetch(location), location.rsplit("/", 1)[0], p.channel)
    sources = []
    for t in tiles:
        ms = open_source(t["url"], cache_bytes=cache_bytes, cache=cache)
        ms = select_axes(ms, {k: v for k, v in t["select"].items() if k in ms[0].info.axes})
        sources.append(ms)
        t["shape"] = [list(level.info.shape) for level in ms]
    if p.level >= (n := min(len(s) for s in sources)):
        raise ValueError(f"stitch:// level={p.level}: the tiles have levels 0..{n - 1}")
    found = stitch(tiles, sources, p)
    placements = [np.asarray(a) for a in found["placements"]]
    kept = [r for r in found["pairs"] if r["kept"]]
    log.info(
        "stitch://: %d of %d overlaps kept (%s inliers) in %.1f s%s", len(kept),
        len(found["pairs"]), ", ".join(str(r["inliers"]) for r in kept), time.perf_counter() - t0,
        "" if (d := S.compare(tiles, placements)) is None else f"; {d:.2f} from the project's stitching",
    )
    base = sources[0][0].info
    levels = []
    for k, grid in enumerate(S.grids(tiles, placements)):
        info = ArrayInfo(
            shape=tuple(grid["shape"]), dtype=base.dtype,
            chunk_shape=tuple(min(c, s) for c, s in zip(CHUNK, grid["shape"])),
            voxel_size=tuple(grid["voxel"]), units=tuple(base.units[-3:]), axes=("z", "y", "x"),
            translation=tuple(grid["origin"]),
        )
        levels.append(FusedLevel(tiles, sources, placements, k, grid, info, p.blend, f"stitch:{url}:{k}"))
    return MultiscaleSource(levels, name=url)
