"""Following an object through frames whose labels are numbered afresh: a ball that drifts and
grows is followed by its overlap, its volume measured, a division noticed, and the box read
each frame kept around it."""

import numpy as np

from chunkmirage import tracking


def frames(n=12, shape=(16, 48, 48), seed=0):
    """A ball drifting in x and growing, beside a still one, relabelled at random each frame;
    at frame 8 it divides (half its volume, one daughter kept on its path)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]]
    out, ids = [], []
    for t in range(n):
        r = 4 + 0.15 * t if t < 8 else (4 + 0.15 * t) / 2 ** (1 / 3)
        ball = (z - 8) ** 2 + (y - 24) ** 2 + (x - 12 - t) ** 2 <= r * r
        other = (z - 8) ** 2 + (y - 10) ** 2 + (x - 38) ** 2 <= 9
        a, b = rng.choice(np.arange(1, 50), 2, replace=False)
        f = np.zeros(shape, np.uint16)
        f[ball], f[other] = a, b
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
    assert sorted(got) == list(range(len(fs)))
    assert [got[t]["label"] for t in sorted(got)] == ids  # the drifting ball, whatever its number
    vols = [got[t]["volume"] for t in sorted(got)]
    assert vols[7] > vols[0] and vols[8] < 0.65 * vols[7]  # it grew, then divided
    assert [t for t in got if got[t].get("divided")] == [8]
    assert abs(got[5]["centroid"][2] - 17) < 0.5
    # after the first frame only boxes around it are read, never a whole frame
    assert all(np.prod(np.subtract(hi, lo)) < fs[0].size / 4 for t, lo, hi in read_boxes[1:])


def test_a_lost_object_ends_the_track():
    fs, ids = frames(4)
    fs[2][:] = 0  # the segmentation missed it
    got = dict(tracking.follow(lambda t, lo, hi: fs[t][tuple(slice(a, b) for a, b in zip(lo, hi))],
                               fs[0].shape, (1, 1, 1), 0, ids[0], range(1, 4), (2, 2, 2)))
    assert sorted(got) == [0, 1]
