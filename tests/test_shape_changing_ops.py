"""Ops that return less than they were given (a valid convolution, shaved by its halo) and
ops that add a leading axis (a model's channels): served chunk by chunk, every level and
every format, they are the whole array's result."""

import gzip
import struct

import numcodecs
import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, fused, open_source
from chunkmirage.core import Box
from chunkmirage.ops import Gradient, Op, Scale


def central(vol, voxel, axes):
    """The gradient along ``axes`` at the voxels whose neighbours are all inside."""
    inner = tuple(slice(1, -1) if a in axes else slice(None) for a in range(vol.ndim))
    return np.stack([np.gradient(vol.astype(np.float64), voxel[a], axis=a)[inner] for a in axes])


def test_a_gradient_chunk_by_chunk_is_the_whole_arrays(zarr2_path, volume):
    p = Pipeline(open_source(zarr2_path), [Gradient()], chunk_shape=(16, 16, 32))
    info = p.info(0)
    assert info.axes == ("c", "z", "y", "x") and info.shape == (3, *volume.shape)
    assert info.chunk_shape == (3, 16, 16, 32) and info.dtype == np.float32
    whole = p.read(0, Box((0, 0, 0, 0), info.shape))
    # per nanometre: the level's voxels are 8 nm
    np.testing.assert_allclose(whole[:, 1:-1, 1:-1, 1:-1], central(volume, (8, 8, 8), (0, 1, 2)), rtol=1e-5, atol=1e-5)
    s1 = p.read(1, Box((0, 0, 0, 0), p.info(1).shape))  # and the next level, per nanometre still
    np.testing.assert_allclose(s1[:, 1:-1, 1:-1, 1:-1], central(volume[::2, ::2, ::2], (16, 16, 16), (0, 1, 2)), rtol=1e-5, atol=1e-5)


def test_ops_before_and_after_share_the_stage(zarr2_path, volume):
    """A blur before the gradient adds its halo to the read; a scale after it sees the channels."""
    blur = {"op": "gaussian", "sigma": 1}
    p = Pipeline(open_source(zarr2_path), [blur, Gradient(axes=[1, 2]), Scale(factor=8)], chunk_shape=(16, 16, 32))
    assert p.info(0).shape == (2, *volume.shape)
    whole = p.read(0, Box((0, 0, 0, 0), p.info(0).shape))
    smooth = Pipeline(open_source(zarr2_path), [blur], chunk_shape=(16, 16, 32)).read(0, Box((0, 0, 0), volume.shape))
    np.testing.assert_allclose(whole[:, :, 1:-1, 1:-1], 8 * central(smooth, (8, 8, 8), (1, 2)), rtol=1e-4, atol=1e-4)


def test_every_format_serves_the_channels(zarr2_path, volume):
    p = Pipeline(open_source(zarr2_path), [Gradient()], chunk_shape=(16, 16, 32))
    client = TestClient(create_app({"g": p}))
    want = p.chunk(0, (0, 1, 1, 0))  # channels, then the chunk at z 1, y 1, x 0
    z3 = client.get("/g/zarr3/s0/zarr.json").json()
    assert z3["shape"] == [3, *volume.shape] and z3["dimension_names"] == ["c", "z", "y", "x"]
    ome = client.get("/g/zarr3/zarr.json").json()["attributes"]["ome"]["multiscales"][0]["axes"]
    assert ome[0] == {"name": "c", "type": "channel"}
    z2 = client.get("/g/zarr/s0/.zarray").json()
    assert z2["chunks"] == [3, 16, 16, 32]
    got = np.frombuffer(numcodecs.get_codec(z2["compressor"]).decode(client.get("/g/zarr/s0/0.1.1.0").content), "<f4")
    np.testing.assert_array_equal(got.reshape(want.shape), want)
    r = client.get("/g/n5/s0/0/1/1/0").content  # N5: x, y, z, c
    ndim = struct.unpack(">HH", r[:4])[1]
    dims = struct.unpack(f">{ndim}I", r[4 : 4 + 4 * ndim])
    n5 = np.frombuffer(gzip.decompress(r[4 + 4 * ndim :]), ">f4").reshape(dims[::-1])
    np.testing.assert_array_equal(n5, want)
    pc = client.get("/g/precomputed/info").json()
    assert pc["num_channels"] == 3 and pc["data_type"] == "float32"


class _Interior(Op):
    """Test op: a 3x3x3 box mean computed only where the box is inside (a valid convolution)."""

    name = "_test_interior"
    halo = 1

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block):
        b = block.astype(np.float32)
        z, y, x = (n - 2 for n in b.shape)
        out = sum(b[i : i + z, j : j + y, k : k + x] for i in range(3) for j in range(3) for k in range(3))
        return out / 27


def test_a_valid_convolution_equals_the_same_filter_cropped(zarr2_path, volume):
    from scipy.ndimage import uniform_filter

    p = Pipeline(open_source(zarr2_path), [_Interior()], chunk_shape=(16, 16, 32))
    whole = p.read(0, Box((0, 0, 0), volume.shape))
    want = uniform_filter(volume.astype(np.float32), 3, mode="nearest")  # the volume's edge repeated, as read
    np.testing.assert_allclose(whole, want, rtol=1e-5, atol=1e-4)


def test_a_wrong_shape_is_still_refused():
    class Wrong(Op):
        name = "_test_wrong"
        halo = 1

        def apply(self, block):
            return block[1:]  # one voxel off one side only

    info = open_source("synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1").levels[0].info
    out, lead, halo = fused.plan(info, [Wrong()])
    with pytest.raises(ValueError, match="changed block shape"):
        fused.run([Wrong()], np.zeros((10, 10, 10)), Box((0, 0, 0), (10, 10, 10)), Box((1, 1, 1), (9, 9, 9)), out, halo)


def test_the_browsers_way_computes_the_same_channels(zarr2_path):
    """The browser engine reads the padded box and calls fused.run, without the pipeline."""
    ms = open_source(zarr2_path)
    ops = [Gradient(axes=[1, 2])]
    p = Pipeline(ms, ops, chunk_shape=(16, 16, 32))
    src = ms.levels[0]
    out, lead, halo = fused.plan(src.info, ops)
    out = out.with_(chunk_shape=p.info(0).chunk_shape)
    for idx in [(0, 0, 0, 0), (0, 1, 2, 1), tuple(g - 1 for g in out.chunk_grid)]:
        out_box = out.chunk_box(idx)
        in_box = fused.input_box(src.info, out_box, lead, halo)
        block = fused.pad_edge(src.read(in_box.clip(src.info.shape)), in_box, src.info.shape)
        np.testing.assert_array_equal(fused.run(ops, block, in_box, out_box, out, halo), p.chunk(0, idx))
