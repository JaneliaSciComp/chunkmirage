"""A dataset written to a store as the server serves it, read back as a stored dataset."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from chunkmirage import Pipeline, PipelineSpec, materialize, open_source
from chunkmirage.cli import app
from chunkmirage.core import Box

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
    result = CliRunner().invoke(app, ["materialize", IMG, str(out), "--level", "1", "--box", "0,0,0:16,16,16"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout.strip().splitlines()[-1])["chunks"] == 1  # level 1's voxels 0-8: one chunk
    back = open_source(str(out)).levels
    p = Pipeline.from_spec(PipelineSpec(source=IMG))
    got = back[1].read(Box((0, 0, 0), back[1].info.shape))
    np.testing.assert_array_equal(got[:16, :16, :16], _whole(p, 1)[:16, :16, :16])
    assert not got[16:].any() and not got[:, 16:].any() and not back[0].read(Box((0, 0, 0), (16, 16, 16))).any()
