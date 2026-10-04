"""An op that reads at a voxel size of its own and writes another, as a model trained at one
resolution does: it runs once, on one level (the source's at that size, else one resampled
to it), and the coarser levels are made from its cached output by ``downsample``."""

from typing import ClassVar

import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, fused, open_source
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops import Downsample, Op, Scale

pytest.importorskip("scipy")


def block_mean(a, f=2):
    """``a``'s ``f``-voxel blocks averaged, its last block along an axis padded with its edge."""
    pad = [(0, -(-n // f) * f - n) for n in a.shape]
    a = np.pad(a.astype(np.float64), pad, "edge")
    z, y, x = (n // f for n in a.shape)
    return a.reshape(z, f, y, f, x, f).mean((1, 3, 5))


class FakeModel(Op):
    """Shaped like a model: reads 8 nm voxels with 8 of context on each side, writes three
    channels of 16 nm voxels, valid convolution style (only the interior comes back)."""

    name = "_fake_model"
    cache = True
    halo = 8
    calls: ClassVar[list] = []

    def input_voxel_size(self):
        return (8.0, 8.0, 8.0)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        out = info.rescaled((16.0, 16.0, 16.0))
        return out.with_(shape=(3, *out.shape), chunk_shape=(3, *out.chunk_shape), dtype=np.dtype("float32"),
                         voxel_size=(1.0, *out.voxel_size), units=("", *out.units), axes=("c", *out.axes),
                         translation=(0.0, *out.translation), kind="image")

    def apply(self, block):
        FakeModel.calls.append(block.shape)
        inner = block[8:-8, 8:-8, 8:-8].astype(np.float32)
        z, y, x = (n // 2 for n in inner.shape)
        m = inner.reshape(z, 2, y, 2, x, 2).mean((1, 3, 5))
        return np.stack([m, 2 * m, 3 * m])


def test_the_model_runs_once_and_the_pyramid_is_made_from_it(zarr2_path, volume):
    FakeModel.calls.clear()
    p = Pipeline(open_source(zarr2_path), [FakeModel()], chunk_shape=(8, 16, 16))
    assert p.num_levels == 2  # from the 8 nm level down: 16 nm, then 32 nm
    i0, i1 = p.info(0), p.info(1)
    assert i0.axes == ("c", "z", "y", "x") and i0.shape == (3, 20, 25, 35) and i0.voxel_size == (1.0, 16.0, 16.0, 16.0)
    assert i1.shape == (3, 10, 13, 18) and i1.voxel_size[1:] == (32.0, 32.0, 32.0)
    # each voxel's position its centre: 16 nm voxels over 8 nm ones start half an old voxel on
    assert i0.translation == (0.0, 4.0, 4.0, 4.0) and i1.translation == (0.0, 12.0, 12.0, 12.0)
    coarse = p.read(1, Box((0, 0, 0, 0), i1.shape))  # the coarse level first: the model runs on level 0
    calls = len(FakeModel.calls)
    assert calls == np.prod(p.levels[0].info.chunk_grid)  # once per chunk of the one level it runs on
    fine = p.read(0, Box((0, 0, 0, 0), i0.shape))
    assert len(FakeModel.calls) == calls  # level 0 was cached: nothing ran again
    want = block_mean(volume)
    for c in range(3):
        np.testing.assert_allclose(fine[c], (c + 1) * want, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(coarse[c], block_mean(fine[c]), rtol=1e-5, atol=1e-4)
    assert (8 * 2 + 16, 16 * 2 + 16, 16 * 2 + 16) in FakeModel.calls  # a whole chunk: twice its voxels, plus 8 each side


def test_every_format_serves_it(zarr2_path):
    p = Pipeline(open_source(zarr2_path), [FakeModel()], chunk_shape=(8, 16, 16))
    c = TestClient(create_app({"m": p}))
    ms = c.get("/m/zarr3/zarr.json").json()["attributes"]["ome"]["multiscales"][0]
    assert [a["name"] for a in ms["axes"]] == ["c", "z", "y", "x"]
    t = [d["coordinateTransformations"] for d in ms["datasets"]]
    assert t[0][0]["scale"] == [1.0, 16.0, 16.0, 16.0] and t[0][1]["translation"] == [0.0, 4.0, 4.0, 4.0]
    assert t[1][0]["scale"] == [1.0, 32.0, 32.0, 32.0] and t[1][1]["translation"] == [0.0, 12.0, 12.0, 12.0]
    for path in ("/m/zarr3/s1/c/0/0/0/0", "/m/zarr/s0/0.1.1.1", "/m/n5/s1/0/0/0/0", "/m/precomputed/info"):
        assert c.get(path).status_code == 200, path


def test_the_level_nearest_the_voxel_size_is_read():
    ms = open_source("synthetic://blobs?shape=64,64,64&chunk=16,16,16&levels=4&voxel_size=8")
    assert ms.level_for((16, 16, 16)) == (1, True)
    assert ms.level_for((12, 12, 12)) == (0, False)  # finer, resampled down
    assert ms.level_for((40, 40, 40)) == (2, False)
    assert ms.level_for((4, 4, 4)) == (0, False)  # finer than any: the finest


def test_a_voxel_size_no_level_has_is_resampled_to(zarr2_path, volume):
    from scipy.ndimage import map_coordinates

    class At12(FakeModel):
        name = "_fake_12"

        def input_voxel_size(self):
            return (12.0, 12.0, 12.0)

        def output_info(self, info):
            return info.with_(dtype=np.dtype("float32"))

        def apply(self, block):
            return block.astype(np.float32)

    p = Pipeline(open_source(zarr2_path), [At12()], chunk_shape=(8, 16, 16))
    info = p.info(0)
    assert info.voxel_size == (12.0, 12.0, 12.0) and info.shape == (27, 34, 47) and info.translation == (2.0, 2.0, 2.0)
    got = p.read(0, Box((0, 0, 0), (6, 6, 6)))
    j = np.indices((6, 6, 6)).reshape(3, -1)
    src = (2.0 + 12.0 * j) / 8.0  # each 12 nm voxel's centre, in the 8 nm level's voxels
    want = map_coordinates(volume.astype(np.float64), src, order=1, mode="nearest").reshape(6, 6, 6)
    np.testing.assert_allclose(got, want, atol=1.01)  # a voxel's value, read as the source's integers


def test_downsample_alone_and_labels_by_their_most_common_value(zarr2_path, volume):
    p = Pipeline(open_source(zarr2_path), [Downsample(factor=[1, 2, 2])], chunk_shape=(8, 16, 16))
    assert p.info(0).voxel_size == (8.0, 16.0, 16.0) and p.info(0).shape == (40, 25, 35)
    np.testing.assert_array_equal(p.read(0, Box((0, 0, 0), (40, 25, 35))),
                                  np.rint(volume.astype(float).reshape(40, 25, 2, 35, 2).mean((2, 4))).astype(np.uint8))
    labels = Pipeline(open_source(zarr2_path), [{"op": "threshold", "low": 128}, {"op": "label"}, {"op": "downsample"}],
                      chunk_shape=(8, 16, 16))
    assert labels.info(0).kind == "label"
    ids = labels.read(0, Box((0, 0, 0), labels.info(0).shape))
    fine = Pipeline(open_source(zarr2_path), [{"op": "threshold", "low": 128}, {"op": "label"}], chunk_shape=(8, 16, 16))
    assert set(np.unique(ids)) <= set(np.unique(fine.read(0, Box((0, 0, 0), volume.shape))))  # no ids averaged into new ones


def test_only_a_stages_first_op_changes_the_grid():
    info = open_source("synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1").levels[0].info
    assert fused.scale(info, [Downsample(), Scale(factor=2)]) == (2.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="first op"):
        fused.scale(info, [Scale(factor=2), Downsample()])
    # a pipeline puts such an op at the start of a stage of its own
    p = Pipeline(open_source("synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"), [Scale(factor=2), Downsample()])
    assert p.info(0).shape == (8, 8, 8)


class AtTen(Op):
    """Reads 10 nm voxels, which no level of a 4, 8, 16 nm pyramid has."""

    name = "_at_ten"

    def input_voxel_size(self):
        return (10.0, 10.0, 10.0)

    def apply(self, block):
        return block


PYRAMID = "synthetic://blobs?shape=64,64,64&chunk=16,16,16&levels=3&voxel_size=4"


def test_nearest_level_reads_a_level_as_it_is_and_says_which():
    resampled = Pipeline(open_source(PYRAMID), [AtTen()])
    assert resampled.info(0).voxel_size == (10.0, 10.0, 10.0)
    assert resampled.input_read == {"op": "_at_ten", "index": 0, "wanted": [10.0] * 3, "level": 1,
                                    "voxel_size": [10.0] * 3, "resampled": True}
    nearest = Pipeline(open_source(PYRAMID), [AtTen()], input_level="nearest")
    assert nearest.info(0).voxel_size == (8.0, 8.0, 8.0) and nearest.num_levels == 2
    assert nearest.input_read["level"] == 1 and nearest.input_read["voxel_size"] == [8.0] * 3
    assert nearest.input_read["resampled"] is False and nearest.digest() != resampled.digest()
    spec = {"source": PYRAMID, "ops": [{"op": "threshold", "low": 1}]}
    app = create_app({"d": Pipeline(open_source(PYRAMID), [AtTen()], input_level="nearest"), "plain": spec})
    c = TestClient(app)
    assert c.get("/api/datasets/d").json()["reads"]["voxel_size"] == [8.0] * 3
    assert c.get("/api/datasets/plain").json()["reads"] is None


def test_the_level_tolerance_is_the_pipelines_to_set():
    class AtNearlyEight(AtTen):
        def input_voxel_size(self):
            return (8.5, 8.5, 8.5)

    assert Pipeline(open_source(PYRAMID), [AtNearlyEight()]).input_read["resampled"] is True
    loose = Pipeline(open_source(PYRAMID), [AtNearlyEight()], level_rtol=0.1)
    assert loose.input_read["resampled"] is False and loose.info(0).voxel_size == (8.0, 8.0, 8.0)


class AtVoxel(Op):
    """Reads voxels of a given size, which the source's one level is not."""

    name = "_at_voxel"
    want: tuple[float, float, float] = (12.0, 12.0, 12.0)

    def input_voxel_size(self):
        return self.want

    def apply(self, block):
        return block


ONE_LEVEL = "synthetic://blobs+noise?shape=24,30,36&chunk=12,15,18&levels=1&voxel_size=4"


def test_a_whole_factor_is_resampled_as_the_mean_of_each_block():
    """Three 4 nm voxels to one 12 nm voxel: their mean, rounded to the dtype, as a stored
    pyramid level is, not linear interpolation between the middle two."""
    src = open_source(ONE_LEVEL)
    raw = src[0].read(Box((0, 0, 0), src[0].info.shape)).astype(np.float64)
    expected = np.rint(raw.reshape(8, 3, 10, 3, 12, 3).mean((1, 3, 5))).astype(np.uint8)
    p = Pipeline(src, [AtVoxel()])
    out = p.levels[0]
    assert out.info.voxel_size == (12.0, 12.0, 12.0) and out.info.translation == (4.0, 4.0, 4.0)
    np.testing.assert_array_equal(out.read(Box((0, 0, 0), out.info.shape)), expected)


def test_mixed_factors_average_the_whole_ones_and_interpolate_the_rest():
    p = Pipeline(open_source(ONE_LEVEL), [AtVoxel(want=(12.0, 6.0, 8.0))])
    info = open_source(ONE_LEVEL)[0].info.rescaled((12.0, 6.0, 8.0))
    out = p.levels[0]
    assert out.info.shape == info.shape and out.info.translation == info.translation
    assert out.read(Box((0, 0, 0), out.info.shape)).dtype == np.uint8
