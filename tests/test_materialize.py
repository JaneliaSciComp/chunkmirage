"""A dataset written to a store as the server serves it, read back as a stored dataset."""

import json
from typing import ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from chunkmirage import Pipeline, PipelineSpec, materialize, open_source
from chunkmirage.cli import app
from chunkmirage.core import Box
from chunkmirage.ops import base
from chunkmirage.ops.base import Op

IMG = "synthetic://julia?shape=32,48,48&chunk=16,16,16&levels=2&seed=2&voxel_size=1"
# moved 24 voxels along x: the last third of every row reads past the image, zeros
SHIFTED = f"warp://{IMG}?affine=1,0,0,0,0,1,0,0,0,0,1,24"


def _whole(p, level):
    return p.read(level, Box((0,) * p.info(level).ndim, p.info(level).shape))


@pytest.mark.parametrize("fmt", ["zarr3", "zarr", "n5"])
def test_a_dataset_written_reads_back_as_it_was_served(tmp_path, fmt):
    p = Pipeline.from_spec(PipelineSpec(source=SHIFTED))
    counts = materialize(p, str(tmp_path / "out"), format=fmt)
    assert counts["written"] + counts["skipped"] == counts["chunks"] == 18 + 4
    assert counts["skipped"] > 0, "the chunks past the image are zeros, not written"
    back = open_source(str(tmp_path / "out"))
    assert len(back.levels) == p.num_levels
    for level in range(p.num_levels):
        got, want = back.levels[level].info, p.info(level)
        assert (got.shape, got.voxel_size, got.dtype) == (want.shape, want.voxel_size, want.dtype)
        np.testing.assert_array_equal(back.levels[level].read(Box((0, 0, 0), got.shape)), _whole(p, level))


def test_a_box_of_one_level_from_the_command_line(tmp_path):
    out = tmp_path / "box.zarr"
    result = CliRunner().invoke(app, ["materialize", IMG, str(out), "--level", "1", "--box", "0,0,0:16,16,16", "--extent", "level"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout.strip().splitlines()[-1])["chunks"] == 1  # level 1's voxels 0-8: one chunk
    back = open_source(str(out)).levels
    p = Pipeline.from_spec(PipelineSpec(source=IMG))
    got = back[1].read(Box((0, 0, 0), back[1].info.shape))
    np.testing.assert_array_equal(got[:16, :16, :16], _whole(p, 1)[:16, :16, :16])
    assert not got[16:].any() and not got[:, 16:].any() and not back[0].read(Box((0, 0, 0), (16, 16, 16))).any()


class Count(Op):
    """Each block as it is, counted: how often the dataset's chunks are computed."""

    name: ClassVar[str] = "count_test"
    calls: ClassVar[int] = 0

    def apply(self, block):
        Count.calls += 1
        return block


@pytest.mark.parametrize("fmt", ["zarr3", "zarr"])
def test_a_box_is_written_as_exactly_the_box_where_it_is(tmp_path, monkeypatch, fmt):
    monkeypatch.setitem(base._REGISTRY, Count.name, Count)
    p = Pipeline.from_spec(PipelineSpec(source=IMG, ops=[{"op": "count_test"}]))  # its last stage keeps nothing
    box = Box((3, 5, 7), (29, 40, 45))  # off the chunk edges along every axis
    Count.calls = 0
    materialize(p, str(tmp_path / "box"), format=fmt, box=box)
    # each of the dataset's chunks the box reaches computed once, though the box's own
    # chunks each stitch up to eight of them
    reached = sum(len(list(p.info(lv).chunks_covering(_level(p, lv, box)))) for lv in range(p.num_levels))
    assert Count.calls == reached
    back = open_source(str(tmp_path / "box")).levels
    got, want = back[0].info, p.info(0)
    assert got.shape == box.shape and got.chunk_shape == want.chunk_shape
    assert got.translation == tuple(t + s * v for t, s, v in zip(want.translation, box.start, want.voxel_size))
    np.testing.assert_array_equal(back[0].read(Box((0, 0, 0), box.shape)), p.read(0, box))
    lv1 = _level(p, 1, box)  # level 1: the same region on its voxels, rounded out to whole ones
    assert back[1].info.shape == lv1.shape
    np.testing.assert_array_equal(back[1].read(Box((0, 0, 0), lv1.shape)), p.read(1, lv1))


def _level(p, level, box):
    from chunkmirage.materialize import level_box

    return level_box(p, level, box)
