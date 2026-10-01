"""GeoTIFF sources and the terrain ops: a tiled GeoTIFF with an overview reads with its
levels, its geo origin and its no-data, and slope and hillshade measure in each level's own
pixel spacing."""

import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")

PIXEL, LEFT, TOP = 2.0, 1000.0, 5000.0  # metres; the top-left corner of the raster
SLOPE = np.tan(np.radians(30))  # rises 30 degrees eastward


def _cog(path, z):
    """``z`` as GDAL writes a COG: 16 x 16 tiles, a half-size overview, the georeferencing
    and no-data tags."""
    geo = [
        (33550, 12, 3, (PIXEL, PIXEL, 0.0), True),  # ModelPixelScale
        (33922, 12, 6, (0.0, 0.0, 0.0, LEFT, TOP, 0.0), True),  # ModelTiepoint
        (42113, 2, 0, "-9999", True),  # GDAL_NODATA
    ]
    with tifffile.TiffWriter(path) as w:
        w.write(z, tile=(16, 16), compression="zlib", predictor=True, extratags=geo)
        w.write(z[::2, ::2].copy(), tile=(16, 16), compression="zlib", subfiletype=1)


def test_a_cog_reads_with_its_levels_origin_and_no_data(tmp_path):
    x = (np.arange(40) + 0.5) * PIXEL
    z = np.tile(x * SLOPE, (36, 1)).astype(np.float32)
    z[0, 0] = -9999
    _cog(tmp_path / "dem.tif", z)
    ms = open_source(str(tmp_path / "dem.tif"))
    assert len(ms) == 2
    i0, i1 = ms[0].info, ms[1].info
    assert i0.shape == (36, 40) and i0.chunk_shape == (16, 16) and i0.units == ("m", "m")
    assert i0.voxel_size == (PIXEL, PIXEL) and i1.voxel_size == (2 * PIXEL, 2 * PIXEL)
    # pixel centres; y counts down the image, minus the northing
    assert i0.translation == pytest.approx((-TOP + PIXEL / 2, LEFT + PIXEL / 2))
    data = ms[0].read(Box((0, 0), (36, 40)))
    assert np.isnan(data[0, 0])
    np.testing.assert_array_equal(data.ravel()[1:], z.ravel()[1:])


def test_slope_and_hillshade_use_each_levels_pixel_spacing(tmp_path):
    x = (np.arange(64) + 0.5) * PIXEL
    _cog(tmp_path / "plane.tif", np.tile(x * SLOPE, (48, 1)).astype(np.float32))
    p = Pipeline(open_source(str(tmp_path / "plane.tif")), [{"op": "slope"}], chunk_shape=(16, 16))
    for level in range(p.num_levels):  # 2 m, then 4 m pixels: 30 degrees both
        s = p.read(level, Box((0, 0), p.info(level).shape))
        np.testing.assert_allclose(s[:, 1:-1], 30.0, atol=1e-3)

    def shade(azimuth, altitude):
        spec = {"op": "hillshade", "azimuth": azimuth, "altitude": altitude}
        q = Pipeline(open_source(str(tmp_path / "plane.tif")), [spec], chunk_shape=(16, 16))
        return int(np.median(q.read(0, Box((0, 0), (48, 64)))))

    # rising eastward, the slope faces west: lit by a western sun, dark under an eastern one
    assert shade(270, 60) == 255
    assert shade(90, 30) == 1
    assert shade(0, 90) == round(1 + 254 * np.cos(np.radians(30)))
