"""The spots op, and a spec's select (one channel of a multichannel image)."""

import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box
from chunkmirage.pipeline import PipelineSpec, select_axes

scipy = pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def fish(tmp_path_factory):
    """A (c, z, y, x) zarr: channel 0 flat, channel 1 background plus noise and Gaussian
    spots of sigma 1 voxel at known places."""
    import tensorstore as ts
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(3)
    shape = (40, 96, 96)
    centres = rng.integers((6, 8, 8), (34, 88, 88), size=(25, 3))
    spots = np.zeros(shape, np.float32)
    spots[tuple(centres.T)] = 1
    spots = gaussian_filter(spots, (0.8, 1.0, 1.0)) * 4000
    img = np.stack([np.full(shape, 300.0), 200 + spots + rng.normal(0, 3, shape)])
    path = tmp_path_factory.mktemp("fish") / "img.zarr"
    arr = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {"chunks": [1, 16, 32, 32], "dtype": "<u2"},
        },
        create=True,
        shape=img.shape,
        dtype=ts.uint16,
    ).result()
    arr[...] = img.astype(np.uint16)
    return str(path), centres


def test_select_pins_a_channel():
    stack = open_source(
        "stack://synthetic://blobs?shape=32,32,32|synthetic://shells?shape=32,32,32"
    )
    one = select_axes(stack, {"c": 1})
    info = one.levels[0].info
    assert info.axes == ("z", "y", "x") and info.shape == (32, 32, 32)
    whole = Box((0, 0, 0), (32, 32, 32))
    expected = open_source("synthetic://shells?shape=32,32,32").levels[0].read(whole)
    np.testing.assert_array_equal(one.levels[0].read(whole), expected)
    with pytest.raises(ValueError, match="entries"):
        select_axes(stack, {"c": 2})
    with pytest.raises(ValueError, match="non-spatial"):
        select_axes(stack, {"z": 0})


def test_spots_finds_the_spots_and_tiles_seamlessly(fish):
    path, centres = fish
    spec = PipelineSpec(
        source=path,
        select={"c": 1},
        chunk_shape=[16, 32, 32],
        ops=[{"op": "spots", "threshold": 20, "radius": 0}],
    )
    p = Pipeline.from_spec(spec)
    info = p.info(0)
    assert info.axes == ("z", "y", "x") and info.dtype == np.uint32
    out = p.read(0, Box((0, 0, 0), info.shape))
    found = np.argwhere(out > 0)
    # every planted spot found within a voxel, and nothing else
    dist = np.abs(found[:, None, :] - centres[None, :, :]).max(-1)
    assert (dist.min(0) <= 1).all(), "a planted spot was missed"
    assert (dist.min(1) <= 1).all(), "a spot was found where none was planted"
    # ids come from positions: a spot found by any chunk has the same id
    ids = out[out > 0]
    assert len(np.unique(ids)) == len(ids)
    # chunk by chunk equals one block computed at once
    from chunkmirage.ops import Spots

    op = Spots(threshold=20, radius=0)
    src = select_axes(open_source(path), {"c": 1}).levels[0]
    h = op.halo
    pad = Box(tuple(-v for v in h), tuple(s + v for s, v in zip(info.shape, h)))
    whole = op.apply_at(src.read_padded(pad, edge=True), pad)[
        h[0] : -h[0], h[1] : -h[1], h[2] : -h[2]
    ]
    np.testing.assert_array_equal(out, whole)


def test_spots_needs_a_volume(fish):
    path, _ = fish
    with pytest.raises(ValueError, match="select"):
        Pipeline(open_source(path), [{"op": "spots"}])
