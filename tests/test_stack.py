"""stack:// sources, and ops that consume their channels (contacts)."""

import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box

scipy = pytest.importorskip("scipy")

A = "synthetic://blobs+noise?shape=128,128,128&chunk=32,32,32&levels=2&seed=1"
B = f"warp://{A}?field=swirl&angle=30"  # the same blobs, twisted: near the originals, not on them
WHOLE = Box((0, 0, 0), (128, 128, 128))


def read_all(url, level=0):
    s = open_source(url).levels[level]
    return s.read(Box((0,) * s.info.ndim, s.info.shape))


def test_stack_serves_images_as_channels():
    stack = open_source(f"stack://{A}|{B}")
    assert len(stack.levels) == 2
    info = stack.levels[0].info
    assert info.axes == ("c", "z", "y", "x")
    assert info.shape == (2, 128, 128, 128) and info.chunk_shape == (2, 32, 32, 32)
    assert info.voxel_size[1:] == open_source(A).levels[0].info.voxel_size
    box = Box((8, 0, 16), (40, 64, 48))
    data = stack.levels[0].read(Box((0, *box.start), (2, *box.stop)))
    np.testing.assert_array_equal(data[0], open_source(A).levels[0].read(box))
    np.testing.assert_array_equal(data[1], open_source(B).levels[0].read(box))
    small = Box((0, 0, 0), (32, 32, 32))
    one = stack.levels[1].read(Box((1, *small.start), (2, *small.stop)))  # one channel, level 1
    np.testing.assert_array_equal(one[0], open_source(B).levels[1].read(small))


def test_stack_rejects_mismatched_grids_and_single_images():
    other = "synthetic://blobs+noise?shape=64,64,32&chunk=32,32,32&levels=2"
    with pytest.raises(ValueError, match="different grid"):
        open_source(f"stack://{A}|{other}")
    with pytest.raises(ValueError, match="two or more"):
        open_source(f"stack://{A}")


def test_contacts_match_a_global_computation():
    from scipy.ndimage import distance_transform_edt

    stack = open_source(f"stack://{A}|{B}")
    a, b = read_all(A), read_all(B)
    op = {"op": "contacts", "radius": 2.5, "a_low": 120, "b_low": 120}
    p = Pipeline(stack, [op], chunk_shape=(16, 16, 16))  # spatial chunks: the c axis is consumed
    info = p.info(0)
    assert info.axes == ("z", "y", "x") and info.dtype == np.uint8
    assert info.chunk_shape == (16, 16, 16) and info.shape == (128, 128, 128)
    got = p.read(0, WHOLE)

    def near(mask):
        return distance_transform_edt(~mask) <= 2.5

    expected = (near(a >= 120) & near(b >= 120)).astype(np.uint8)
    h = 4  # the padded read's zero fill only differs at the volume's own border
    inner = (slice(h, -h),) * 3
    assert expected[inner].any() and not expected[inner].all()
    np.testing.assert_array_equal(got[inner], expected[inner])


def test_contacts_then_label_and_the_errors():
    stack = open_source(f"stack://{A}|{B}")
    contacts = {"op": "contacts", "radius": 2, "a_low": 120, "b_low": 120}
    mask = Pipeline(stack, [contacts], chunk_shape=(32, 32, 32)).read(0, WHOLE)
    labels = Pipeline(
        stack, [contacts, {"op": "label", "min_size": 10}], chunk_shape=(32, 32, 32)
    ).read(0, WHOLE)
    assert labels.dtype == np.uint32
    assert np.all((labels > 0) <= (mask > 0))  # labels only where the mask is
    with pytest.raises(ValueError, match="channel axis"):
        Pipeline(open_source(A), [contacts])


@pytest.fixture(scope="module")
def centred(tmp_path_factory):
    """A two-level OME-Zarr pyramid whose levels share their centre (OME's half-voxel shift),
    which is what mirroring every level in place needs."""
    import json

    import tensorstore as ts

    group = tmp_path_factory.mktemp("flip") / "img.zarr"
    group.mkdir()
    (group / ".zgroup").write_text('{"zarr_format": 2}')
    datasets = [
        {
            "path": f"s{k}",
            "coordinateTransformations": [
                {"type": "scale", "scale": [4.0 * 2**k] * 3},
                {"type": "translation", "translation": [2.0 * (2**k - 1)] * 3},
            ],
        }
        for k in (0, 1)
    ]
    axes = [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"]
    (group / ".zattrs").write_text(
        json.dumps({"multiscales": [{"version": "0.4", "axes": axes, "datasets": datasets}]})
    )
    s0 = np.random.default_rng(5).integers(0, 255, (16, 32, 24), dtype=np.uint8)
    s1 = s0.reshape(8, 2, 16, 2, 12, 2).mean(axis=(1, 3, 5)).astype(np.uint8)
    for k, data in enumerate((s0, s1)):
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(group / f"s{k}")},
            "metadata": {"chunks": [8, 8, 8], "compressor": {"id": "gzip"}, "dtype": "|u1"},
        }
        arr = ts.open(spec, create=True, delete_existing=True, dtype=ts.uint8, shape=data.shape)
        arr.result()[...] = data
    return str(group), (s0, s1)


def test_flip_mirrors_every_level_in_place(centred):
    path, levels = centred
    flipped = open_source(f"flip://{path}?axes=y")
    plain = open_source(path)
    assert len(flipped.levels) == 2
    for k, data in enumerate(levels):
        assert flipped.levels[k].info == plain.levels[k].info
        whole = Box((0, 0, 0), data.shape)
        np.testing.assert_array_equal(flipped.levels[k].read(whole), np.flip(data, axis=1))
    box = Box((3, 10, 5), (12, 25, 20))  # any box: the mirrored one is read
    np.testing.assert_array_equal(
        flipped.levels[0].read(box), np.flip(levels[0], axis=1)[3:12, 10:25, 5:20]
    )
    twice = open_source(f"flip://flip://{path}?axes=y?axes=y").levels[0]  # twice is the image
    np.testing.assert_array_equal(twice.read(Box((0, 0, 0), levels[0].shape)), levels[0])


def test_flip_refuses_levels_that_would_disagree_and_bad_axes():
    # synthetic:// levels share their corner, not their centre: each mirrored in place would
    # put coarse voxels half a voxel away from the fine ones they summarise
    with pytest.raises(ValueError, match="move them apart"):
        open_source(f"flip://{A}?axes=y")
    one = A.replace("levels=2", "levels=1")
    with pytest.raises(ValueError, match="axes"):
        open_source(f"flip://{one}")
    with pytest.raises(ValueError, match="has axes"):
        open_source(f"flip://{one}?axes=q")


def test_flipped_images_stack(centred):
    path, levels = centred
    stack = open_source(f"stack://flip://{path}?axes=y|{path}").levels[0]
    data = stack.read(Box((0, 0, 0, 0), (2, *levels[0].shape)))
    np.testing.assert_array_equal(data[0], np.flip(levels[0], axis=1))
    np.testing.assert_array_equal(data[1], levels[0])
