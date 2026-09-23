import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box

scipy = pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def synth():
    return open_source("synthetic://blobs+noise?shape=512,512,512&chunk=32,32,32&levels=3&seed=3")


def test_synthetic_is_deterministic_and_multiscale_consistent(synth):
    assert len(synth) == 3
    s0, s1 = synth[0], synth[1]
    assert s0.info.shape == (512, 512, 512) and s1.info.shape == (256, 256, 256)
    assert s1.info.voxel_size == (16, 16, 16)
    a = s0.read(Box((0, 0, 0), (32, 32, 32)))
    b = s0.read(Box((0, 0, 0), (32, 32, 32)))
    np.testing.assert_array_equal(a, b)
    assert a.std() > 5  # not flat
    # level 1 is exactly level 0 sampled every 2 voxels
    coarse = s1.read(Box((0, 0, 0), (16, 16, 16)))
    np.testing.assert_array_equal(coarse, a[::2, ::2, ::2])


def test_synthetic_chunks_tile_seamlessly(synth):
    s0 = synth[0]
    whole = s0.read(Box((0, 0, 0), (64, 32, 32)))
    top, bottom = s0.read(Box((0, 0, 0), (32, 32, 32))), s0.read(Box((32, 0, 0), (64, 32, 32)))
    np.testing.assert_array_equal(np.concatenate([top, bottom]), whole)


def test_synthetic_kinds_and_errors():
    j = open_source("synthetic://julia?shape=128,128,128&levels=1")[0].read(
        Box((0, 0, 0), (32, 128, 128))
    )
    assert j.max() > 0
    sh = open_source("synthetic://shells?shape=256,256,256&levels=1")[0].read(
        Box((0, 0, 0), (64, 64, 64))
    )
    assert sh.dtype == np.uint8
    with pytest.raises(ValueError):
        open_source("synthetic://nope?shape=8,8,8")


def test_dog_and_morphology_match_global(synth):
    from scipy.ndimage import binary_opening, gaussian_filter

    box = Box((0, 0, 0), (96, 96, 96))
    raw = synth[0].read(box).astype(np.float32)
    p = Pipeline(synth, [{"op": "dog", "sigma": 2.0}], chunk_shape=(32, 32, 32))
    got = p.read(0, box).astype(np.float32)
    d = gaussian_filter(raw, 2.0, mode="nearest") - gaussian_filter(raw, 3.2, mode="nearest")
    exp = np.clip(d * 4 + 128, 0, 255).astype(np.uint8).astype(np.float32)
    h = 12
    np.testing.assert_allclose(got[h:-h, h:-h, h:-h], exp[h:-h, h:-h, h:-h], atol=1)

    p2 = Pipeline(
        synth,
        [{"op": "threshold", "low": 120}, {"op": "morphology", "operation": "open", "radius": 2}],
        chunk_shape=(32, 32, 32),
    )
    got2 = p2.read(0, box)
    r = 2
    zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    ball = (zz**2 + yy**2 + xx**2) <= r * r
    exp2 = binary_opening(raw >= 120, structure=ball).astype(np.uint8)
    np.testing.assert_array_equal(got2[8:-8, 8:-8, 8:-8], exp2[8:-8, 8:-8, 8:-8])


def test_label_unique_per_chunk_and_size_filter(synth):
    p = Pipeline(
        synth,
        [{"op": "threshold", "low": 120}, {"op": "label", "min_size": 50}],
        chunk_shape=(32, 32, 32),
    )
    # find two chunks that actually contain objects (blob placement is random per seed)
    found = []
    for idx in np.ndindex(4, 4, 4):
        arr = p.chunk(0, idx)
        if arr.any():
            found.append((idx, arr))
        if len(found) == 2:
            break
    assert len(found) == 2, "expected at least two chunks with objects"
    (ia_idx, a), (ib_idx, b) = found
    assert a.dtype == np.uint32
    ia, ib = set(np.unique(a)) - {0}, set(np.unique(b)) - {0}
    assert not (ia & ib)  # different chunks never share a label
    # without the size filter there are at least as many components
    p0 = Pipeline(
        synth, [{"op": "threshold", "low": 120}, {"op": "label"}], chunk_shape=(32, 32, 32)
    )
    assert len(np.unique(p0.chunk(0, ia_idx))) >= len(np.unique(a))
