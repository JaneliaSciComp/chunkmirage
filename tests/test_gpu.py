"""The resampler's own CUDA kernel (``gpu.interpolate``) against ``cupyx.scipy.ndimage``,
which it must match bit for bit. Needs CuPy and a GPU: skipped without them, so CI (no GPU)
never runs these; run them on a GPU node before changing the kernel or the code around it."""

from urllib.parse import quote

import numpy as np
import pytest

from chunkmirage import gpu, open_source
from chunkmirage.core import Box
from chunkmirage.transforms import Affine, VectorField

cupy = pytest.importorskip("cupy")

IMG = "synthetic://julia?shape=32,48,48&chunk=16,16,16&levels=1&seed=2&voxel_size=1"
PASS1 = f"warp://{IMG}?field=swirl&angle=60&centre=16,20,24&radius=16&show=field"
PASS2 = f"warp://{IMG}?field=swirl&angle=-45&centre=16,28,20&radius=14&show=field"


@pytest.fixture(autouse=True)
def on_the_gpu():
    gpu.set_device("auto")
    if gpu.device() is None:
        pytest.skip("no GPU")
    yield
    gpu.set_device("cpu")


def both(monkeypatch, fn):
    """``fn()`` with the kernel, then through cupyx, as numpy arrays."""
    monkeypatch.setattr(gpu, "KERNELS", True)
    fast = gpu.host(fn())
    monkeypatch.setattr(gpu, "KERNELS", False)
    return fast, gpu.host(fn())


def same_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_field_samples_are_the_bits_cupyx_gives(monkeypatch, dtype):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(3, 20, 30, 40)).astype(dtype)
    field = VectorField(
        lambda sl: vectors[sl], vectors.shape, 0, Affine.scale_translation([2.0, 1.5, 1.0], [3, -2, 5]), "f"
    )
    # inside, beyond every edge (clamped), on grid nodes (one corner per axis), and rows
    # that are not finite (NaN out)
    pts = rng.uniform([-10, -20, -10], [50, 60, 60], size=(5000, 3))
    pts[:200] = np.round(pts[:200])
    pts[300, 1], pts[301, 0] = np.nan, np.inf
    for p in (pts, pts[302:]):
        dev = cupy.asarray(p, dtype=np.float32)
        same_bits(*both(monkeypatch, lambda dev=dev: field.sample(dev)))


def test_a_warp_through_two_fields_is_the_bits_cupyx_gives(monkeypatch):
    p1, p2 = quote(PASS1, safe=""), quote(PASS2, safe="")
    level = open_source(f"warp://{IMG}?field={p2}|{p1}&affine=1,0,0,0.5,0,1,0,0.25,0,0,1,-0.5").levels[0]
    box = Box((0, 0, 0), (32, 48, 48))
    same_bits(*both(monkeypatch, lambda: level.read(box)))


def test_points_outside_the_image_and_channels_are_the_bits_cupyx_gives(monkeypatch):
    # two channels, shifted so part of each slab falls off the image (those voxels are 0)
    other = IMG.replace("seed=2", "seed=5")
    level = open_source(f"warp://stack://{IMG}|{other}?affine=1,0,0,9.5,0,1,0,-7.25,0,0,1,0").levels[0]
    assert level.n_lead == 1
    box = Box((0,) * 4, level.info.shape)
    fast, ref = both(monkeypatch, lambda: level.read(box))
    same_bits(fast, ref)
    assert (ref == 0).any() and (ref != 0).any()
