"""chunksizes:// on a sharded precomputed volume tensorstore writes: random chunks come out
near their raw size, empty ones near nothing, each minishard the mean of its chunks (its
bytes from the shard index alone), and no minishard level where a minishard is not one block
of chunks; and a spec's ``levels`` keeps the finest level alone."""

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import open_source
from chunkmirage.core import Box
from chunkmirage.pipeline import Pipeline


def write(path, preshift=3, shard_bits=3):
    """x, y, z grids of 12, 8, 4 chunks (unequal Morton bits, 9 in all); minishards of 2³
    chunks and shards of 4³ by default; random chunks where x < 48, the rest zeros."""
    data = np.zeros((32, 64, 96), np.uint8)  # z, y, x
    data[:, :, :48] = np.random.default_rng(0).integers(0, 256, (32, 64, 48), np.uint8)
    store = ts.open({
        "driver": "neuroglancer_precomputed",
        "kvstore": {"driver": "file", "path": str(path)},
        "create": True,
        "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 1},
        "scale_metadata": {
            "size": [96, 64, 32], "resolution": [8, 8, 8], "chunk_size": [8, 8, 8], "encoding": "raw",
            "sharding": {"@type": "neuroglancer_uint64_sharded_v1", "hash": "identity", "preshift_bits": preshift,
                         "minishard_bits": 3, "shard_bits": shard_bits, "minishard_index_encoding": "gzip",
                         "data_encoding": "gzip"},
        },
    }).result()
    store[..., 0].write(data.transpose(2, 1, 0)).result()  # x, y, z, c
    return str(path)


@pytest.fixture
def volume(tmp_path):
    return write(tmp_path / "vol")


def test_sizes_from_the_indexes(volume):
    src = open_source(f"chunksizes://precomputed://{volume}?range=0,2")
    s0, per_chunk, per_mini = src.levels
    assert per_chunk.info.shape == (4, 8, 12) and per_chunk.info.voxel_size == (64, 64, 64)
    assert per_chunk.info.translation == (32, 32, 32)  # voxel 0 spans chunk 0
    assert per_mini.info.shape == (2, 4, 6) and per_mini.info.voxel_size == (128, 128, 128)
    a = per_chunk.read(Box((0, 0, 0), (4, 8, 12))).astype(float)
    assert (a[..., :6] > 120).all()  # random: about raw size, half of range's 0..2
    assert (a[..., 6:] < 10).all()   # zeros: almost nothing
    b = per_mini.read(Box((0, 0, 0), (2, 4, 6))).astype(float)
    means = a.reshape(2, 2, 4, 2, 6, 2).mean(axis=(1, 3, 5))
    assert np.abs(b - means).max() <= 1  # the shard index's gaps are the chunks' bytes
    assert s0.read(Box((0, 0, 0), (1, 1, 96)))[0, 0, 48:].max() == 0 < s0.read(Box((0, 0, 0), (1, 1, 48))).max()  # as stored


def test_levels_keeps_the_finest_alone(volume):
    url = f"chunksizes://precomputed://{volume}"
    assert Pipeline.from_spec({"source": url}).num_levels == 3
    assert Pipeline.from_spec({"source": url, "levels": [0]}).num_levels == 1
    with pytest.raises(ValueError, match="finest first"):
        Pipeline.from_spec({"source": url, "levels": [1, 0]})


@pytest.mark.parametrize("preshift, shard_bits", [(0, 6), (3, 1)])
def test_no_minishard_level_unless_a_minishard_is_one_block(tmp_path, volume, preshift, shard_bits):
    # one chunk a minishard (preshift 0), or blocks far apart sharing one (too few shard bits)
    src = open_source(f"chunksizes://precomputed://{write(tmp_path / 'other', preshift, shard_bits)}?range=0,2")
    exact = open_source(f"chunksizes://precomputed://{volume}?range=0,2")
    assert len(src.levels) == 2  # the finest, and one voxel per chunk: still right
    box = Box((0, 0, 0), (4, 8, 12))
    assert np.abs(src.levels[1].read(box).astype(int) - exact.levels[1].read(box)).max() <= 2
