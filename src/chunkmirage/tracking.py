"""Following one object through a time series of label images, frame by frame, as the
frames are read: in each next frame it is the label that overlaps it most. Labels need not
keep their ids from frame to frame (a segmentation done frame by frame rarely does); an
object that moves less than its own size between frames is followed by its overlap.

Each frame reads only a box around the object (its last bounding box, grown by a margin),
so following one nucleus through a whole time-lapse reads a sliver of it. A nucleus that
divides first collapses (its envelope breaks down in mitosis, and the segmentation loses
it); new nuclei then appear near where it was, as labels nothing covered the frame before
(``newborns``), and are taken for its daughters and followed in turn, so a track becomes a
lineage. That is a guess from where and when they appear: the segmentation does not say
which nucleus a new one came from, and in a dense colony a neighbour's daughter can be
taken for this one's. ``step`` and
``newborns`` are the work of one frame, numpy only, which the browser engine's workers run
as they are; ``follow`` and ``lineage`` loop them over a source for Python callers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

MIN_OVERLAP = 0.2  # of the object's voxels: less and it is lost (it left, or the labels failed)
DIVIDED = 0.65  # a volume this fraction of the frame before's, or less: a division
NEWBORN = 0.4  # a label no label of the frame before covers this share of: new (a daughter, say)
GAP = 24  # frames searched for daughters after a nucleus collapses into mitosis and is lost
WITHIN = 18.0  # physical units (µm) from the mother that daughters are looked for


def measure(block: np.ndarray, label: int, start, voxel) -> dict | None:
    """Label ``label`` in ``block`` (whose voxel 0 is ``start`` of the level): its volume
    (voxels times their size), centroid and bounding box (level voxels, ``[lo, hi)``), and
    whether it touches the block's faces (the block may not hold all of it)."""
    at = np.argwhere(np.asarray(block) == label)
    if not len(at):
        return None
    lo, hi = at.min(0), at.max(0) + 1
    shape = np.asarray(block.shape)
    touches = bool(((lo == 0) & (np.asarray(start) > 0)).any() or (hi == shape).any())
    start = np.asarray(start)
    return {
        "label": int(label),
        "volume": float(len(at) * np.prod(voxel)),
        "centroid": (at.mean(0) + start).tolist(),
        "lo": (lo + start).tolist(),
        "hi": (hi + start).tolist(),
        "touches": touches,
    }


def step(prev: np.ndarray, label: int, nxt: np.ndarray, start, voxel) -> dict | None:
    """The object labelled ``label`` in ``prev``, in the next frame ``nxt`` (the same box of
    the level, from ``start``): the label it overlaps most there, measured, with the overlap
    (its share of the object's voxels). ``None`` if under ``MIN_OVERLAP``."""
    mask = np.asarray(prev) == label
    if not mask.any():
        return None
    under = np.asarray(nxt)[mask]
    under = under[under > 0]
    if len(under) < MIN_OVERLAP * mask.sum():
        return None
    ids, counts = np.unique(under, return_counts=True)
    best = int(ids[np.argmax(counts)])
    found = measure(nxt, best, start, voxel)
    if found is not None:
        found["overlap"] = float(counts.max() / mask.sum())
    return found


def box_around(record: dict, shape, margin) -> tuple[list[int], list[int]]:
    """The box the next frame is read over: the object's bounding box grown by ``margin``
    voxels, within the level."""
    lo = np.maximum(np.asarray(record["lo"]) - margin, 0)
    hi = np.minimum(np.asarray(record["hi"]) + margin, shape)
    return lo.tolist(), hi.tolist()


def follow(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
           label: int, frames: range, margin) -> Iterator[tuple[int, dict]]:
    """The object labelled ``label`` at frame ``t0``, through ``frames`` (counting up or down
    from ``t0``); ``read(t, lo, hi)`` gives a frame's labels over a box of the level. Yields
    ``(t, record)`` for ``t0`` and each frame it is found in, until it is lost; a record
    says ``divided`` when its volume fell to ``DIVIDED`` of the frame before's."""
    margin = np.asarray(margin)
    # the whole object at t0: start from a box around its voxels in the whole frame
    first = measure(read(t0, [0] * len(shape), list(shape)), label, [0] * len(shape), voxel)
    if first is None:
        raise ValueError(f"no label {label} at frame {t0}")
    yield t0, first
    record, t = first, t0
    for nt in frames:
        lo, hi = box_around(record, shape, margin)
        prev, nxt = read(t, lo, hi), read(nt, lo, hi)
        found = step(prev, record["label"], nxt, lo, voxel)
        if found is not None and found["touches"]:  # the box cut it: read all of it
            lo, hi = box_around(found, shape, 2 * margin)
            found = step(read(t, lo, hi), record["label"], read(nt, lo, hi), lo, voxel)
        if found is None:
            return
        found["divided"] = found["volume"] <= DIVIDED * record["volume"]
        yield nt, found
        record, t = found, nt


def newborns(prev: np.ndarray, nxt: np.ndarray, start, voxel, centre, within: float = WITHIN,
             min_volume: float = 0.0) -> list[dict]:
    """Labels of ``nxt`` that are new: covered less than ``NEWBORN`` by any one label of the
    frame before, ``prev`` (the same box, from ``start``); within ``within`` (physical units)
    of ``centre`` (level voxels) and of ``min_volume`` or more. After a mitosis, which hides
    the nucleus for a few frames, these are likely its daughters (a guess: nothing says which
    nucleus a new one came from). Nearest first."""
    nxt, prev, voxel = np.asarray(nxt), np.asarray(prev), np.asarray(voxel)
    out = []
    for i in np.unique(nxt[nxt > 0]):
        mask = nxt == i
        under = prev[mask]
        under = under[under > 0]
        if len(under) and np.bincount(under).max() >= NEWBORN * mask.sum():
            continue
        found = measure(nxt, int(i), start, voxel)
        far = np.linalg.norm((np.asarray(found["centroid"]) - np.asarray(centre)) * voxel)
        if far <= within and found["volume"] >= min_volume:
            found["distance"] = float(far)
            out.append(found)
    return sorted(out, key=lambda f: f["distance"])


def mother_of(branch: dict[int, dict], lost: int) -> dict | None:
    """If a branch lost at frame ``lost`` had collapsed first (its last volume ``DIVIDED`` of
    its largest in the dozen frames before, or less), the record at that largest: the
    nucleus that went into mitosis. ``None`` if it was just lost."""
    recent = [branch[t] for t in sorted(branch) if lost - 12 <= t < lost]
    if not recent:
        return None
    biggest = max(recent, key=lambda r: r["volume"])
    return biggest if recent[-1]["volume"] <= DIVIDED * biggest["volume"] else None


def daughters(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, mother: dict,
              lost: int, frames: int, gap: int = GAP, within: float = WITHIN) -> list[tuple[int, dict]]:
    """Up to two likely daughters of ``mother`` (a record, see ``mother_of``): the first new
    nuclei (``newborns``) near it, looked for from frame
    ``lost`` for ``gap`` frames in a box ``within`` around it: ``(frame, record)`` where each
    first appears."""
    voxel = np.asarray(voxel)
    reach = np.ceil(within / voxel).astype(int)
    lo = np.maximum(np.round(mother["centroid"]).astype(int) - reach, 0).tolist()
    hi = np.minimum(np.round(mother["centroid"]).astype(int) + reach + 1, shape).tolist()
    found: list[tuple[int, dict]] = []
    for t in range(lost, min(lost + gap, frames)):
        for d in newborns(read(t - 1, lo, hi), read(t, lo, hi), lo, voxel, mother["centroid"], within, 0.1 * mother["volume"]):
            found.append((t, d))
            if len(found) == 2:
                return found
    return found


def lineage(read: Callable[[int, list[int], list[int]], np.ndarray], shape, voxel, t0: int,
            label: int, frames: int, margin, max_branches: int = 8) -> list[dict[int, dict]]:
    """The object labelled ``label`` at frame ``t0`` followed forward and back (``follow``),
    and forward through its divisions: the daughters of each nucleus that collapses and is
    lost are followed too. Returns branches (frame -> record), the first the object's own;
    each daughter's first record says ``mother``, the branch it came from."""
    main = dict(follow(read, shape, voxel, t0, label, range(t0 + 1, frames), margin))
    main.update(follow(read, shape, voxel, t0, label, range(t0 - 1, -1, -1), margin))
    branches, todo = [main], [0]
    while todo and len(branches) < max_branches:
        k = todo.pop(0)
        last = max(branches[k])
        mother = mother_of(branches[k], last + 1)
        if mother is None or last + 1 >= frames:
            continue
        for t, d in daughters(read, shape, voxel, mother, last + 1, frames):
            if len(branches) >= max_branches:
                break
            branch = dict(follow(read, shape, voxel, t, d["label"], range(t + 1, frames), margin))
            branch[t]["mother"] = k
            branches.append(branch)
            todo.append(len(branches) - 1)
    return branches
