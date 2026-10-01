import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.cache import LRUCache
from chunkmirage.core import Box
from chunkmirage.ops import Cast, Threshold, Uniform

scipy = pytest.importorskip("scipy")


def test_open_multiscale_zarr2(zarr2_path, volume):
    ms = open_source(zarr2_path)
    assert len(ms) == 2
    assert ms[0].info.shape == volume.shape
    assert ms[0].info.voxel_size == (8, 8, 8)
    assert ms[1].info.voxel_size == (16, 16, 16)
    assert ms[0].info.units == ("nanometer",) * 3
    np.testing.assert_array_equal(ms[0].read(Box((0, 0, 0), (5, 6, 7))), volume[:5, :6, :7])


def test_open_n5_and_zarr3(n5_path, zarr3_path, volume):
    n5 = open_source(n5_path)
    assert len(n5) == 1
    assert n5[0].info.voxel_size == (4.0, 4.0, 4.0)
    np.testing.assert_array_equal(
        n5[0].read(Box((3, 4, 5), (10, 20, 30))), volume[3:10, 4:20, 5:30]
    )
    z3 = open_source(zarr3_path)
    np.testing.assert_array_equal(z3[0].read(Box((0, 0, 0), volume.shape)), volume)


def test_threshold_matches_numpy(zarr2_path, volume):
    p = Pipeline(open_source(zarr2_path), [Threshold(low=100, high=200)])
    info = p.info(0)
    assert info.dtype == np.uint8
    for idx in [(0, 0, 0), (2, 3, 2), (1, 1, 1)]:
        box = info.chunk_box(idx)
        expected = ((volume >= 100) & (volume < 200)).astype(np.uint8)[box.slices()]
        np.testing.assert_array_equal(p.chunk(0, idx), expected)


def test_halo_op_matches_global(zarr2_path, volume):
    """Neighbourhood filters computed per-chunk with halo must equal the whole-volume filter."""
    from scipy.ndimage import uniform_filter

    p = Pipeline(open_source(zarr2_path), [Uniform(size=5)], chunk_shape=(16, 16, 16))
    full = p.read(0, Box((0, 0, 0), volume.shape))
    expected = uniform_filter(volume.astype(np.float32), 5, mode="nearest")
    # interior must match exactly; borders differ only by boundary handling of the padded read,
    # which we make consistent by using 0-padding in read_padded -> compare interior.
    h = 3
    np.testing.assert_allclose(
        full[h:-h, h:-h, h:-h], expected[h:-h, h:-h, h:-h], rtol=1e-5, atol=1e-4
    )


def test_stage_cache_keys_isolate_downstream_edits(zarr2_path):
    cache = LRUCache()
    ms = open_source(zarr2_path)
    p1 = Pipeline(ms, [Cast(dtype="float32"), Threshold(low=100)], cache=cache)
    p1.chunk(0, (0, 0, 0))
    raw_entries = len(cache)
    assert raw_entries == 1  # raw stage cached; Cast/Threshold are cache=False
    # Editing the threshold reuses the raw chunk.
    p2 = Pipeline(ms, [Cast(dtype="float32"), Threshold(low=150)], cache=cache)
    misses_before = cache.misses
    p2.chunk(0, (0, 0, 0))
    assert cache.misses == misses_before  # raw chunk was a hit
    assert p1.digest() != p2.digest()


def test_rechunk_reads_span_source_chunks(zarr2_path, volume):
    p = Pipeline(open_source(zarr2_path), [], chunk_shape=(24, 24, 24), cache_source=False)
    idx = (1, 1, 2)
    box = p.info(0).chunk_box(idx)
    np.testing.assert_array_equal(p.chunk(0, idx), volume[box.slices()])
    # edge chunk is clipped
    last = tuple(g - 1 for g in p.info(0).chunk_grid)
    assert p.chunk(0, last).shape == p.info(0).chunk_box(last).shape


def test_sources_share_one_tensorstore_context(zarr2_path, monkeypatch):
    # one cache pool for every source in the process, so two opens of a store (and its
    # levels, and other images) share what is decoded rather than each keeping its own
    from chunkmirage.sources import tensorstore_source

    seen = []
    real = tensorstore_source.ts.open

    def recording(spec, *args, context=None, **kw):
        seen.append(context)
        return real(spec, *args, context=context, **kw)

    monkeypatch.setattr(tensorstore_source.ts, "open", recording)
    open_source(zarr2_path, cache_bytes=1 << 20)
    open_source(zarr2_path, cache_bytes=1 << 20)
    assert len(seen) >= 2 and all(c is seen[0] for c in seen)
    open_source(zarr2_path, cache_bytes=2 << 20)
    assert seen[-1] is not seen[0]  # another budget, another pool
