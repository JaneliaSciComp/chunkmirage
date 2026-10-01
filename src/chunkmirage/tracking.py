"""Following one object through a time series of label images, frame by frame, as the
frames are read: in each next frame it is the label that overlaps it most. Labels need not
keep their ids from frame to frame (a segmentation done frame by frame rarely does); an
object that moves less than its own size between frames is followed by its overlap.

Each frame reads only a box around the object (its last bounding box, grown by a margin),
so following one nucleus through a whole time-lapse reads a sliver of it. ``step`` is the
work of one frame, numpy only, which the browser engine's workers run as it is; ``follow``
loops it over a source for Python callers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

MIN_OVERLAP = 0.2  # of the object's voxels: less and it is lost (it left, or the labels failed)
DIVIDED = 0.65  # a volume this fraction of the frame before's, or less: a division (one daughter followed)


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
