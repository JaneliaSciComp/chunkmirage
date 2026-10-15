"""Stitching overlapping tiles by interest points, as BigStitcher registers them: blobs found
in each overlap by a difference of Gaussians, matched between neighbouring tiles by the
constellation of their nearest neighbours, the matches filtered by RANSAC, and every tile's
placement fitted to all the kept matches at once. Then the tiles are fused, blended where
they overlap, region by region: nothing is written.

Coordinates are physical, C order ``(z, y, x)``. A tile is placed by an affine from its
voxels to the scene (rows ``[A | t]``): its stage position (where the microscope put it),
which stitching corrects by a model of its own (translation, rigid or affine) on top.

Each step is a function of arrays and JSON-able values, so the browser engine's workers run
them as they are (numpy and scipy only), and ``stitch://`` (``sources.stitch``) runs them
when it is opened. ``tiles_from_bdv`` reads the tiles and their placement from a BigStitcher
(BigDataViewer) project, the XML its tools write.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MIN_MATCHES = {"translation": 1, "rigid": 3, "affine": 4}  # a model's smallest sample
BATCH = 512  # RANSAC hypotheses scored at once


class StitchParams(BaseModel):
    """How tiles are stitched: the interest points, their matching, RANSAC and the fit."""

    channel: int = Field(0, ge=0, description="The tiles' channel (setup attribute) stitched")
    level: int = Field(1, ge=0, description="The tiles' level interest points are found on")
    sigma: float = Field(
        1.8, gt=0, description="Blob size: the difference of Gaussians' smaller sigma, in "
        "voxels of that level along x (scaled along the other axes by their spacing)",
    )
    threshold: float = Field(
        0.005, gt=0, description="Smallest difference-of-Gaussians peak kept, in units of the "
        "tile's intensity range (BigStitcher's threshold)",
    )
    margin: float = Field(
        20.0, ge=0, description="How far tiles may be from their stage positions: the "
        "overlaps searched are grown by this (physical units)",
    )
    neighbors: int = Field(3, ge=2, le=6, description="Nearest neighbours a point's descriptor holds")
    redundancy: int = Field(1, ge=0, le=3, description="Extra neighbours: subsets of them are tried too")
    significance: float = Field(
        3.0, ge=1, description="A match's descriptor must be this many times closer than the "
        "next best candidate's",
    )
    model: Literal["translation", "rigid", "affine"] = Field(
        "translation", description="What each tile may do beyond its stage position"
    )
    epsilon: float = Field(5.0, gt=0, description="RANSAC: largest error of an inlier (physical units)")
    min_inlier_ratio: float = Field(0.1, ge=0, le=1, description="RANSAC: smallest share of inliers")
    min_inliers: int = Field(6, ge=1, description="RANSAC: fewest inliers for a pair to count")
    iterations: int = Field(10000, ge=1, le=200000, description="RANSAC: hypotheses tried")
    seed: int = Field(0, description="RANSAC's random draws")
    blend: float = Field(
        40.0, ge=0, description="Fusion: the band at a tile's edges its weight falls off over "
        "(physical units along y and x)",
    )

    @classmethod
    def from_query(cls, q: dict[str, str]) -> StitchParams:
        if unknown := set(q) - set(cls.model_fields):
            raise ValueError(
                f"unknown stitch:// parameters {sorted(unknown)}; allowed: {sorted(cls.model_fields)}"
            )
        return cls(**q)


# ------------------------------------------------------------------ affines
def to4(a) -> np.ndarray:
    m = np.eye(4)
    m[:3] = np.asarray(a, float).reshape(3, 4)
    return m


def apply(a, p: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return p @ a[:3, :3].T + a[:3, 3]


def compose(*affines) -> np.ndarray:
    """``compose(a, b)(p) == a(b(p))``."""
    m = np.eye(4)
    for a in affines:
        m = m @ to4(a)
    return m[:3]


def invert(a) -> np.ndarray:
    return np.linalg.inv(to4(a))[:3]


# ------------------------------------------------------------------ interest points
def _gaussian(x: np.ndarray, sigma) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(x, sigma, mode="nearest", truncate=3.0)


def detect(block: np.ndarray, voxel, sigma: float, threshold: float, lo: float, hi: float) -> np.ndarray:
    """Bright blobs in ``block``: local maxima of a difference of Gaussians (sigma and
    2^(1/4) sigma, BigStitcher's), at least ``threshold`` of the intensity range [lo, hi],
    to subvoxel precision. Returns ``(n, 3)`` voxel positions in ``block``; none within a
    voxel of its faces (their maxima are cut off)."""
    from scipy.ndimage import maximum_filter

    if min(block.shape) < 3:
        return np.zeros((0, 3))
    voxel = np.asarray(voxel, float)
    s = sigma * voxel[-1] / voxel  # the same physical size along every axis
    x = (np.asarray(block, np.float32) - lo) / max(hi - lo, 1e-12)
    dog = _gaussian(x, s) - _gaussian(x, s * 2 ** 0.25)
    peak = (dog == maximum_filter(dog, size=3, mode="nearest")) & (dog >= threshold)
    peak[[0, -1]] = peak[:, [0, -1]] = peak[:, :, [0, -1]] = False
    at = np.argwhere(peak)
    if not len(at):
        return np.zeros((0, 3))
    # a parabola through each peak and its two neighbours, per axis
    out = at.astype(float)
    v = dog[tuple(at.T)]
    for a in range(3):
        up, down = at.copy(), at.copy()
        up[:, a] += 1
        down[:, a] -= 1
        f1, f0 = dog[tuple(up.T)], dog[tuple(down.T)]
        curve = f1 + f0 - 2 * v
        out[:, a] += np.where(curve < 0, 0.5 * (f0 - f1) / np.where(curve < 0, curve, -1), 0).clip(-0.5, 0.5)
    return out


def points_in(tile: dict, level: int, block: np.ndarray, start, voxel, lo, hi, p: StitchParams) -> np.ndarray:
    """A tile's interest points in the scene box ``[lo, hi]``, from ``block`` (its level's
    voxels from ``start``, as ``region`` gives them): scene positions at its stage placement."""
    block = np.asarray(block)
    v = detect(block, voxel, p.sigma, p.threshold, float(block.min()), float(block.max()))
    w = apply(compose(tile["stage"], level_to_base(tile, level)), v + np.asarray(start))
    return w[((w >= lo) & (w <= hi)).all(1)]


# ------------------------------------------------------------------ matching
def descriptors(points: np.ndarray, neighbors: int, redundancy: int):
    """Each point's constellation: the offsets to ``neighbors`` of its nearest
    ``neighbors + redundancy`` (every such subset, nearest first), concatenated. Unchanged
    by a translation, so tiles placed only roughly still match. Returns the descriptors and
    the point each belongs to."""
    from scipy.spatial import cKDTree

    k = neighbors + redundancy
    if len(points) <= k:
        return np.zeros((0, 3 * neighbors)), np.zeros(0, int)
    _, near = cKDTree(points).query(points, k + 1)
    offsets = points[near[:, 1:]] - points[:, None, :]  # (n, k, 3), nearest first
    subsets = list(itertools.combinations(range(k), neighbors))
    d = np.concatenate([offsets[:, list(s)].reshape(len(points), -1) for s in subsets])
    owner = np.tile(np.arange(len(points)), len(subsets))
    return d, owner


def match(a: np.ndarray, b: np.ndarray, neighbors: int = 3, redundancy: int = 1, significance: float = 3.0) -> np.ndarray:
    """Candidate matches ``(i, j)`` of points ``a[i]`` and ``b[j]``: ``b[j]`` is the point
    whose descriptors come closest to one of ``a[i]``'s, ``significance`` times closer than
    any other point's, and the other way round."""
    from scipy.spatial import cKDTree

    da, oa = descriptors(a, neighbors, redundancy)
    db, ob = descriptors(b, neighbors, redundancy)
    if not len(da) or not len(db):
        return np.zeros((0, 2), int)

    def best(dx, ox, dy, oy, nx):
        """For each point of x: its nearest point of y, and how much nearer than the next."""
        k = min(len(dy), 8)
        dist, at = cKDTree(dy).query(dx, k)
        dist, at = dist.reshape(len(dx), k), at.reshape(len(dx), k)
        rows = np.repeat(ox, k)
        cols = oy[at.ravel()]
        dist = dist.ravel()
        # each (x point, y point): the closest of their descriptors
        order = np.lexsort((dist, cols, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        first = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
        rows, cols, dist = rows[first], cols[first], dist[first]
        order = np.lexsort((dist, rows))
        rows, cols, dist = rows[order], cols[order], dist[order]
        start = np.r_[True, rows[1:] != rows[:-1]]
        idx = np.flatnonzero(start)
        nearest = np.full(nx, -1)
        ratio = np.zeros(nx)
        nearest[rows[idx]] = cols[idx]
        nxt = idx + 1
        has = (nxt < len(rows)) & (np.r_[rows, -1][nxt] == rows[idx])
        second = np.where(has, np.r_[dist, np.inf][np.where(has, nxt, len(dist))], np.inf)
        ratio[rows[idx]] = second / np.maximum(dist[idx], 1e-12)
        return nearest, ratio

    ab, ra = best(da, oa, db, ob, len(a))
    ba, _ = best(db, ob, da, oa, len(b))
    i = np.flatnonzero((ab >= 0) & (ra >= significance))
    i = i[ba[ab[i]] == i]  # mutual
    return np.stack([i, ab[i]], 1)


# ------------------------------------------------------------------ models
def fit(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The ``model`` taking points ``p`` nearest to ``q`` (least squares), as ``[A | t]``."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if model == "translation":
        return np.hstack([np.eye(3), (q - p).mean(0)[:, None]])
    cp, cq = p.mean(0), q.mean(0)
    if model == "rigid":
        u, _, vt = np.linalg.svd((q - cq).T @ (p - cp))
        r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        return np.hstack([r, (cq - r @ cp)[:, None]])
    x = np.hstack([p - cp, np.ones((len(p), 1))])
    sol = np.linalg.lstsq(x, q - cq, rcond=None)[0]  # (4, 3)
    a = sol[:3].T
    return np.hstack([a, (cq + sol[3] - a @ cp)[:, None]])


def _fit_many(model: str, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``fit`` of each of a batch of minimal samples ``(m, k, 3)``: ``(m, 3, 4)``."""
    cp, cq = p.mean(1, keepdims=True), q.mean(1, keepdims=True)
    m = len(p)
    if model == "translation":
        out = np.zeros((m, 3, 4))
        out[:, :, :3] = np.eye(3)
        out[:, :, 3] = (cq - cp)[:, 0]
        return out
    if model == "rigid":
        u, _, vt = np.linalg.svd(np.einsum("mki,mkj->mij", q - cq, p - cp))
        d = np.sign(np.linalg.det(u @ vt))
        u[:, :, 2] *= d[:, None]
        a = u @ vt
    else:
        x = np.concatenate([p - cp, np.ones(p.shape[:2] + (1,))], 2)  # (m, 4, 4)
        x = x + np.eye(4)[None] * 1e-9  # degenerate samples stay finite (and score badly)
        sol = np.linalg.solve(x, q - cq)  # (m, 4, 3)
        a = np.transpose(sol[:, :3], (0, 2, 1))
        cq = cq + sol[:, 3:4]
    t = cq[:, 0] - np.einsum("mij,mj->mi", a, cp[:, 0])
    return np.concatenate([a, t[:, :, None]], 2)


def ransac(p: np.ndarray, q: np.ndarray, model: str, epsilon: float, min_inlier_ratio: float,
           min_inliers: int, iterations: int, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """The ``model`` that most candidate matches ``p[i] -> q[i]`` agree with to within
    ``epsilon``, from ``iterations`` random minimal samples, refitted to its inliers until
    they stop changing. Returns the model (``None`` if too few agree) and the inliers."""
    n, k = len(p), MIN_MATCHES[model]
    none = np.zeros(n, bool)
    if n < max(k, min_inliers):
        return None, none
    rng = np.random.default_rng(seed)
    best, best_count = none, 0
    for start in range(0, iterations, BATCH):
        m = min(BATCH, iterations - start)
        pick = np.argsort(rng.random((m, n)), 1)[:, :k] if k > 1 else rng.integers(0, n, (m, 1))
        models = _fit_many(model, p[pick], q[pick])
        d = p @ models[:, :, :3].transpose(0, 2, 1) + models[:, None, :, 3] - q  # (m, n, 3)
        err = np.sqrt((d * d).sum(2))
        counts = (err < epsilon).sum(1)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), err[i] < epsilon
    for _ in range(20):  # refit to the inliers until they settle
        if best.sum() < k:
            return None, none
        a = fit(model, p[best], q[best])
        now = np.linalg.norm(apply(a, p) - q, axis=1) < epsilon
        if (now == best).all():
            break
        best = now
    if best.sum() < max(k, min_inliers) or best.mean() < min_inlier_ratio:
        return None, best
    return fit(model, p[best], q[best]), best


def optimize(n: int, links: list[tuple[int, int, np.ndarray, np.ndarray]], model: str,
             fixed: int = 0, rounds: int = 2000, tolerance: float = 1e-4) -> tuple[list[np.ndarray], list[bool]]:
    """Every tile's correction (applied after its stage placement) fitted to all the kept
    matches at once: link ``(i, j, p, q)`` asks tile ``i``'s points ``p`` to meet tile
    ``j``'s ``q`` (both placed by their stage positions). Each round refits each tile to
    where its neighbours put their ends of its matches. Of each group of tiles that links
    join, one stays where it is: ``fixed`` in its group, the first tile in the others.
    Returns the corrections and which tiles were joined to another."""
    identity = np.hstack([np.eye(3), np.zeros((3, 1))])
    models = [identity.copy() for _ in range(n)]
    group = list(range(n))

    def find(t):
        while group[t] != t:
            t = group[t]
        return t

    for i, j, _, _ in links:
        group[find(i)] = find(j)
    anchors = {}
    for t in [fixed, *range(n)]:
        anchors.setdefault(find(t), t)
    joined = [any(t in (i, j) for i, j, _, _ in links) for t in range(n)]
    order = [t for t in range(n) if joined[t] and t not in anchors.values()]
    for _ in range(rounds):
        moved = 0.0
        for t in order:
            p, q = [], []
            for i, j, pi, qj in links:
                if i == t:
                    p.append(pi)
                    q.append(apply(models[j], qj))
                elif j == t:
                    p.append(qj)
                    q.append(apply(models[i], pi))
            if not p:
                continue
            p, q = np.concatenate(p), np.concatenate(q)
            new = fit(model, p, q)
            moved = max(moved, float(np.abs(apply(new, p) - apply(models[t], p)).max()))
            models[t] = new
        if moved < tolerance:
            break
    return models, joined


# ------------------------------------------------------------------ the whole registration
def overlaps(tiles: list[dict], margin: float) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Pairs of tiles whose stage placements overlap, and the overlap's scene box (grown by
    ``margin``), as ``(i, j, lo, hi)``."""
    boxes = [bounds(t["stage"], t["shape"][0]) for t in tiles]
    out = []
    for i, j in itertools.combinations(range(len(tiles)), 2):
        lo = np.maximum(boxes[i][0], boxes[j][0])
        hi = np.minimum(boxes[i][1], boxes[j][1])
        if (hi > lo).all():
            out.append((i, j, lo - margin, hi + margin))
    return out


def bounds(affine, shape) -> tuple[np.ndarray, np.ndarray]:
    """The scene box an affine puts a volume of ``shape`` voxels in (voxel centres at
    integers, so a voxel's extent is +-0.5)."""
    corners = np.array(list(itertools.product(*[(-0.5, s - 0.5) for s in shape])))
    w = apply(affine, corners)
    return w.min(0), w.max(0)


def region(tile: dict, level: int, lo, hi, placement=None) -> tuple[list[int], list[int]] | None:
    """The voxels ``[start, stop)`` of a tile's level holding the scene box ``[lo, hi]`` (a
    voxel more on each side, for interpolation) where ``placement`` (default: its stage
    position) puts it, or ``None`` if it holds none of it."""
    to_scene = compose(tile["stage"] if placement is None else placement, level_to_base(tile, level))
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    v = apply(invert(to_scene), corners)
    start = np.maximum(np.floor(v.min(0)) - 1, 0).astype(int)
    stop = np.minimum(np.ceil(v.max(0)) + 2, tile["shape"][level]).astype(int)
    if (stop <= start).any():
        return None
    return start.tolist(), stop.tolist()


def level_to_base(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxels to its level-0 voxels: each level halves (or so) its shape;
    its voxel centres sit at the centres of the level-0 voxels they cover."""
    f = np.array(tile["shape"][0], float) / np.array(tile["shape"][level], float)
    f = np.round(f) if np.allclose(f, np.round(f), atol=0.05) else f
    return np.hstack([np.diag(f), ((f - 1) / 2)[:, None]])


def register(tiles: list[dict], points: list[tuple[np.ndarray, np.ndarray]], p: StitchParams,
             fixed: int = 0) -> dict:
    """Matches, RANSAC and the global fit from each overlapping pair's interest points
    (``points[k]``: pair ``k`` of ``overlaps``, both tiles' points in scene coordinates at
    their stage placement). JSON-able: per pair the candidates, inliers and errors; per
    tile its correction and its final placement."""
    pairs = overlaps(tiles, p.margin)
    links, report = [], []
    for k, (i, j, _, _) in enumerate(pairs):
        a, b = (np.asarray(x, float).reshape(-1, 3) for x in points[k])
        m = match(a, b, p.neighbors, p.redundancy, p.significance)
        model, inliers = ransac(a[m[:, 0]], b[m[:, 1]], p.model, p.epsilon, p.min_inlier_ratio,
                                p.min_inliers, p.iterations, p.seed)
        kept = m[inliers] if model is not None else m[:0]
        if model is not None:
            links.append((i, j, a[kept[:, 0]], b[kept[:, 1]]))
        report.append({
            "tiles": [i, j], "points": [len(a), len(b)], "candidates": len(m),
            "inliers": int(len(kept)), "kept": model is not None,
            # the matches, scene positions of both ends at the stage placement
            "a": a[m[:, 0]].round(3).tolist(), "b": b[m[:, 1]].round(3).tolist(),
            "inlier": inliers.tolist(),
        })
    corrections, placed = optimize(len(tiles), links, p.model, fixed)
    for r in report:
        i, j = r["tiles"]
        if r["kept"]:
            sel = np.array(r["inlier"], bool)
            a, b = np.array(r["a"])[sel], np.array(r["b"])[sel]
            e = np.linalg.norm(apply(corrections[i], a) - apply(corrections[j], b), axis=1)
            r["error"] = {"mean": float(e.mean()), "max": float(e.max())}
    return {
        "pairs": report,
        "corrections": [c.tolist() for c in corrections],
        "placed": placed,
        "placements": [compose(c, t["stage"]).tolist() for c, t in zip(corrections, tiles)],
    }


def compare(tiles: list[dict], placements: list) -> float | None:
    """How far the found placements are from the tiles' reference ones (a registration done
    before), as the root-mean-square distance of the tiles' centres once the two are
    lined up on average (stitching fixes the scene only up to a common shift)."""
    if not all(t.get("reference") is not None for t in tiles):
        return None
    c = [np.array(t["shape"][0], float)[None] / 2 - 0.5 for t in tiles]
    ours = np.concatenate([apply(a, x) for a, x in zip(placements, c)])
    theirs = np.concatenate([apply(t["reference"], x) for t, x in zip(tiles, c)])
    d = ours - theirs
    return float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))


# ------------------------------------------------------------------ fusion
def level_voxel(tile: dict, level: int) -> np.ndarray:
    """A tile level's voxel size in the scene, at its stage placement."""
    return np.linalg.norm(compose(tile["stage"], level_to_base(tile, level))[:, :3], axis=0)


def grids(tiles: list[dict], placements: list) -> list[dict]:
    """The fused volume's levels, one per level every tile has: the first tile's voxel size
    there, and the origin and shape covering every tile where ``placements`` put them."""
    lo, hi = zip(*(bounds(a, t["shape"][0]) for a, t in zip(placements, tiles)))
    lo, hi = np.min(lo, 0), np.max(hi, 0)
    out = []
    for k in range(min(len(t["shape"]) for t in tiles)):
        voxel = level_voxel(tiles[0], k)
        shape = np.ceil((hi - lo) / voxel - 1e-6).astype(int)
        out.append({"shape": shape.tolist(), "voxel": voxel.tolist(), "origin": (lo + voxel / 2).tolist()})
    return out


def scene_box(grid: dict, out_lo, out_hi) -> tuple[np.ndarray, np.ndarray]:
    """The scene box that voxels ``[out_lo, out_hi)`` of a fused level cover."""
    voxel, origin = np.asarray(grid["voxel"]), np.asarray(grid["origin"])
    return origin + (np.asarray(out_lo) - 0.5) * voxel, origin + (np.asarray(out_hi) - 0.5) * voxel


def _weight(v: np.ndarray, shape, band: np.ndarray) -> np.ndarray:
    """A tile's blending weight at its voxel positions ``v`` (n, 3): 1 inside, falling off as
    a half cosine over ``band`` voxels from each face, 0 outside."""
    w = np.ones(len(v))
    for a in range(3):
        inside = (v[:, a] >= -0.5) & (v[:, a] <= shape[a] - 0.5)
        w = np.where(inside, w, 0)
        if band[a] > 0:
            d = np.minimum(v[:, a] + 0.5, shape[a] - 0.5 - v[:, a])
            w = w * np.where(d < band[a], 0.5 - 0.5 * np.cos(np.pi * np.clip(d, 0, None) / band[a]), 1)
    return w


def fuse(tiles: list[dict], placements: list, level: int, grid: dict, out_lo, out_hi,
         blocks: list[tuple[np.ndarray, list[int]] | None], blend: float, dtype) -> np.ndarray:
    """Voxels ``[out_lo, out_hi)`` of the fused volume on ``grid``: each tile sampled
    (trilinearly) where its placement puts it, from ``blocks[t]`` (its level's voxels from
    a start, or ``None`` where it has none of the region), averaged with weights that fall
    off near its edges, so the seams do not show."""
    from scipy.ndimage import map_coordinates

    out_lo, out_hi = np.asarray(out_lo), np.asarray(out_hi)
    shape = tuple(out_hi - out_lo)
    idx = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(out_lo, out_hi)], indexing="ij"), -1)
    w_scene = np.asarray(grid["origin"]) + idx.reshape(-1, 3) * np.asarray(grid["voxel"])
    total = np.zeros(len(w_scene))
    weight = np.zeros(len(w_scene))
    for t, got in enumerate(blocks):
        if got is None:
            continue
        block, start = got
        tile = tiles[t]
        to_scene = compose(placements[t], level_to_base(tile, level))
        v = apply(invert(to_scene), w_scene)
        vox = np.abs(np.linalg.det(np.asarray(to_scene)[:, :3])) ** (1 / 3)
        band = np.array([0.0, blend, blend]) / vox  # along y and x: tiles share their z range
        w = _weight(v, tile["shape"][level], band)
        hit = w > 0
        if not hit.any():
            continue
        local = (v[hit] - np.asarray(start)).T
        total[hit] += w[hit] * map_coordinates(np.asarray(block, np.float32), local, order=1, mode="nearest")
        weight[hit] += w[hit]
    out = np.where(weight > 0, total / np.maximum(weight, 1e-12), 0).reshape(shape)
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(dtype)


# ------------------------------------------------------------------ BigStitcher projects
def _affine_xyz(text: str) -> np.ndarray:
    """A BigDataViewer affine (12 numbers, rows of x, y, z) in C order (z, y, x)."""
    m = np.array([float(v) for v in text.split()]).reshape(3, 4)
    a = np.zeros((3, 4))
    a[:, :3] = m[::-1, :3][:, ::-1]
    a[:, 3] = m[::-1, 3]
    return a


def tiles_from_bdv(xml: str, base: str, channel: int = 0, drop: str = "Stitching Transform") -> list[dict]:
    """The tiles of one channel of a BigStitcher project (its XML text; ``base``: the URL
    of the folder it is in), each with its stage placement: every view transform but the
    outermost ones named ``drop`` (the stitching found before, kept as the reference). Reads
    projects whose images are OME-Zarr (BigStitcher-Spark's ``bdv.multimg.zarr`` loader),
    one group per setup and time point."""
    root = ET.fromstring(xml)
    loader = root.find("SequenceDescription/ImageLoader")
    if loader is None or loader.get("format") != "bdv.multimg.zarr":
        raise ValueError(f"only OME-Zarr BigStitcher projects are read (bdv.multimg.zarr), not {loader.get('format') if loader is not None else 'none'}")
    zarr = loader.findtext("zarr", "").strip()
    folder = zarr if "://" in zarr or zarr.startswith("/") else f"{base.rstrip('/')}/{zarr}"
    groups = {(g.get("setup"), g.get("tp")): g.get("path") for g in loader.iter("zgroup")}
    tp = min({tp for _, tp in groups}, key=int)
    regs = {r.get("setup"): r for r in root.iter("ViewRegistration") if r.get("timepoint") == tp}
    tiles = []
    for vs in root.iter("ViewSetup"):
        attrs = vs.find("attributes")
        if attrs is None or int(attrs.findtext("channel", "0")) != channel:
            continue
        sid = vs.findtext("id").strip()
        size = [int(v) for v in vs.findtext("size").split()][::-1]  # x y z -> z y x
        transforms = regs[sid].findall("ViewTransform")
        names = [t.findtext("Name", "") for t in transforms]
        affines = [_affine_xyz(t.findtext("affine")) for t in transforms]
        n = 0
        while n < len(names) and names[n] == drop:
            n += 1
        tiles.append({
            "name": f"tile {attrs.findtext('tile', sid)}", "setup": int(sid),
            "url": f"{folder}/{groups[(sid, tp)]}", "select": {"t": 0, "c": 0},
            "shape": [size], "stage": compose(*affines[n:]).tolist(),
            "reference": compose(*affines).tolist() if n else None,
        })
    if not tiles:
        raise ValueError(f"no tiles of channel {channel}")
    return tiles
