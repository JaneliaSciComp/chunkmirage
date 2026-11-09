"""An op with ``overlap``: its blocks, chunk plus overlap, computed once each and cached as a
stage of their own, and the served chunks their blend, smooth where blocks meet."""

from typing import ClassVar

import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box
from chunkmirage.ops import base
from chunkmirage.ops.base import Op


class BlockCentre(Op):
    """Every voxel the x coordinate of its block's centre: blocks computed alone step at
    every chunk's edge, so blending shows."""

    name: ClassVar[str] = "block_centre_test"
    overlap: ClassVar[int] = 4
    calls: ClassVar[int] = 0

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply_at(self, block, box):
        BlockCentre.calls += 1
        return np.full(block.shape, (box.start[-1] + box.stop[-1]) / 2, np.float32)


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setitem(base._REGISTRY, BlockCentre.name, BlockCentre)


SRC = "synthetic://noise?shape=16,16,32&chunk=8,8,8&levels=1&seed=1"
WHOLE = Box((0, 0, 0), (16, 16, 32))


def test_blocks_are_computed_once_and_blended_smoothly():
    p = Pipeline(open_source(SRC), [{"op": "block_centre_test"}])
    stage = p.levels[0]
    assert stage.blocks is not None and not stage.cached and stage.blocks.cached, "the blocks are cached, the op's output as it says"
    assert stage.blocks.chunk_box((1, 1, 1)) == Box((4, 4, 4), (20, 20, 20)), "a block is its chunk and the overlap around it"
    assert len(p.stages[0]) == 2 and p.stages[0][1] is stage

    out = p.read(0, WHOLE)
    assert BlockCentre.calls == np.prod(p.info(0).chunk_grid), "each block once, however many chunks it reaches"
    assert np.allclose(out, out[:1, :1]), "the same along y and z"
    along_x = out[0, 0]
    assert (np.diff(along_x) >= 0).all() and np.diff(along_x).max() < 4, "a ramp, not the staircase of block centres"
    np.testing.assert_allclose(along_x[:3], 4.0)  # where only the first block reaches
    np.testing.assert_allclose(along_x[-3:], 28.0)

    monkeypatch_overlap = BlockCentre.overlap
    try:
        BlockCentre.overlap = 0
        plain = Pipeline(open_source(SRC), [{"op": "block_centre_test"}])
        assert plain.levels[0].blocks is None
        steps = plain.read(0, WHOLE)[0, 0]
        np.testing.assert_array_equal(steps, np.repeat([4.0, 12.0, 20.0, 28.0], 8))
    finally:
        BlockCentre.overlap = monkeypatch_overlap


def test_an_op_with_overlap_is_a_stage_of_its_own_and_keeps_integer_dtypes():
    class Rounded(BlockCentre):
        name: ClassVar[str] = "block_centre_int_test"

        def output_dtype(self, in_dtype):
            return np.dtype("uint8")

    base._REGISTRY[Rounded.name] = Rounded
    try:
        p = Pipeline(open_source(SRC), [{"op": "cast", "dtype": "float32"}, {"op": "block_centre_int_test"}, {"op": "cast", "dtype": "float32"}])
        assert [s.cache_key().split(":")[0] for s in p.stages[0]] == ["raw", "cast", "block_centre_int_test", "cast"]
        out = p.read(0, WHOLE)[0, 0]
        assert out.dtype == np.float32 and (np.diff(out) >= 0).all() and out[0] == 4 and out[-1] == 28
        assert (out == np.round(out)).all(), "rounded after the mean, as the op's uint8 output asks"
    finally:
        del base._REGISTRY[Rounded.name]


def test_a_plan_says_what_a_read_would_compute_and_computes_nothing():
    p = Pipeline(open_source(SRC), [{"op": "block_centre_test"}])
    BlockCentre.calls = 0
    box = Box((0, 0, 0), (8, 8, 8))  # one chunk, which 8 blocks reach
    plan = p.plan(0, box)
    assert BlockCentre.calls == 0
    source, raw, blocks, out = plan
    assert "source" in source and source["reads"] == raw["compute"] == 12, "the blocks' inputs: 2 x 2 x 3 chunks"
    assert blocks["stage"].startswith("blocks:") and (blocks["chunks"], blocks["compute"]) == (8, 8)
    assert out["stage"] == p.levels[0].cache_key() and not out["keeps"] and out["compute"] == 1
    p.read(0, box)
    blocks, out = p.plan(0, box)
    assert (blocks["held"], blocks["compute"]) == (8, 0), "held now: the walk stops there"
    assert out["compute"] == 1, "a stage that keeps nothing computes again"


class OnCorners(BlockCentre):
    """Blocks on chunk corners, weighted by a ramp: as properties, read per instance, the way
    one op class serving many models sets them from each model's config."""

    name: ClassVar[str] = "block_corner_test"
    weighting: str = "ramp"

    @property
    def block_phase(self):
        return "corner"

    @property
    def blend(self):
        return self.weighting


def test_blocks_on_corners_put_each_chunk_in_eight_blended_by_a_ramp():
    from chunkmirage.ops.base import list_ops

    base._REGISTRY[OnCorners.name] = OnCorners
    try:
        assert (list_ops()[OnCorners.name]["block_phase"], list_ops()[OnCorners.name]["blend"]) == ("dynamic", "dynamic")
        p = Pipeline(open_source(SRC), [{"op": OnCorners.name}])
        blocks = p.levels[0].blocks
        assert blocks.grid == (3, 3, 5), "one more block along each axis"
        assert blocks.chunk_box((0, 0, 0)) == Box((-8, -8, -8), (8, 8, 8)), "a block is 2 x 2 x 2 chunks, on a corner"
        inner = Box((0, 0, 8), (8, 8, 16))  # a chunk inside along x
        assert p.plan(0, inner)[-2]["chunks"] == 8, "8 blocks, where centred ones take 2 x 2 x 3"
        assert Pipeline(open_source(SRC), [{"op": "block_centre_test"}]).plan(0, inner)[-2]["chunks"] == 12

        BlockCentre.calls = 0
        along_x = p.read(0, WHOLE)[0, 0]
        assert BlockCentre.calls == 3 * 3 * 5, "each block once"
        # every block is its x centre, so the blend along x is the ramp-weighted mean of the
        # centres of the blocks reaching each voxel (the other axes' weights cancel)
        expected = []
        for x in range(32):
            w = {j: min(x - (8 * j - 8) + 0.5, 8 * j + 8 - x - 0.5) / 8 for j in range(5) if 8 * j - 8 <= x < 8 * j + 8}
            expected.append(sum(wj * 8 * j for j, wj in w.items()) / sum(w.values()))
        np.testing.assert_allclose(along_x, expected, rtol=1e-6)

        tent = Pipeline(open_source(SRC), [{"op": OnCorners.name, "weighting": "tent"}])
        assert tent.levels[0].cache_key() != p.levels[0].cache_key(), "the blend is part of the op's identity"
        with pytest.raises(ValueError, match="blend 'mean'"):
            Pipeline(open_source(SRC), [{"op": OnCorners.name, "weighting": "mean"}])
    finally:
        del base._REGISTRY[OnCorners.name]
