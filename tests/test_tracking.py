"""Following an object through frames whose labels are numbered afresh: a ball that drifts and
grows is followed by its overlap, its volume measured, a division noticed, and the box read
each frame kept around it."""

import numpy as np

from chunkmirage import tracking


def frames(n=16, shape=(16, 48, 64), seed=0):
    """A ball drifting in x and growing, beside a still one, relabelled at random each frame;
    at frame 8 it collapses into mitosis (a small remnant), at 9 and 10 it is gone, and at 11
    two daughters appear either side of where it was and drift on."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    out, ids = [], []
    for t in range(n):
        a, b, c = rng.choice(np.arange(1, 90), 3, replace=False)
        f = np.zeros(shape, np.uint16)
        f[(z - 8) ** 2 + (y - 8) ** 2 + (x - 56) ** 2 <= 9] = b  # a neighbour that stays put
        if t < 8:
            r = 4 + 0.25 * t
            f[(z - 8) ** 2 + (y - 24) ** 2 + (x - 12 - t) ** 2 <= r * r] = a
        elif t == 8:
            f[(z - 8) ** 2 + (y - 24) ** 2 + (x - 19) ** 2 <= 12.25] = a  # the collapsing remnant
        elif t >= 11:
            for cy, label in ((18, a), (30, c)):
                f[(z - 8) ** 2 + (y - cy) ** 2 + (x - 8 - t) ** 2 <= 9 + 0.5 * (t - 11)] = label
        out.append(f)
        ids.append(int(a))
    return out, ids


def test_an_object_is_followed_by_its_overlap_through_new_labels():
    fs, ids = frames()
    read_boxes = []

    def read(t, lo, hi):
        read_boxes.append((t, tuple(lo), tuple(hi)))
        return fs[t][tuple(slice(a, b) for a, b in zip(lo, hi))]

    got = dict(tracking.follow(read, fs[0].shape, (1, 1, 1), 0, ids[0], range(1, len(fs)), (2, 2, 2)))
    assert sorted(got) == list(range(9))  # up to the collapse, then lost
    assert [got[t]["label"] for t in sorted(got)] == ids[:9]  # the drifting ball, whatever its number
    vols = [got[t]["volume"] for t in sorted(got)]
    assert vols[7] > vols[0] and got[8]["divided"]  # it grew, then collapsed
    assert abs(got[5]["centroid"][2] - 17) < 0.5
    # after the first frame only boxes around it are read, never a whole frame
    assert all(np.prod(np.subtract(hi, lo)) < fs[0].size / 4 for t, lo, hi in read_boxes[1:])


def test_a_division_is_followed_into_both_daughters():
    fs, ids = frames()

    def read(t, lo, hi):
        return fs[t][tuple(slice(a, b) for a, b in zip(lo, hi))]

    branches = tracking.lineage(read, fs[0].shape, (1, 1, 1), 3, ids[3], len(fs), (2, 2, 2))
    assert len(branches) == 3 and sorted(branches[0]) == list(range(9))
    for b in branches[1:]:  # each daughter from its first frame to the end, its mother the first
        assert min(b) == 11 and max(b) == len(fs) - 1 and b[11]["mother"] == 0
    ys = sorted(b[11]["centroid"][1] for b in branches[1:])
    assert abs(ys[0] - 18) < 0.5 and abs(ys[1] - 30) < 0.5
    # the neighbour is not a daughter: it was there the frame before
    assert all(abs(b[11]["centroid"][2] - 56) > 5 for b in branches[1:])


def test_a_lost_object_ends_the_track():
    fs, ids = frames(4)
    fs[2][:] = 0  # the segmentation missed it
    got = dict(tracking.follow(lambda t, lo, hi: fs[t][tuple(slice(a, b) for a, b in zip(lo, hi))],
                               fs[0].shape, (1, 1, 1), 0, ids[0], range(1, 4), (2, 2, 2)))
    assert sorted(got) == [0, 1]
