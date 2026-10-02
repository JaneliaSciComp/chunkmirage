"""Remote sensing's indices of two bands, and their change between two dates: burn severity
(dNBR) from a stack of before and after images."""

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box


def write(path, data):
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {"shape": list(data.shape), "chunks": [1, 8, 8], "dtype": "<u2"}}
    ts.open(spec, create=True).result().write(data).result()
    return str(path)


def test_burn_severity_is_the_index_before_minus_after(tmp_path):
    rng = np.random.default_rng(0)
    bands = [rng.integers(1001, 6000, (1, 16, 16)).astype(np.uint16) for _ in range(4)]
    bands[0][0, 0, 0] = 0  # no data in one band, one pixel
    paths = [write(tmp_path / f"b{i}", b) for i, b in enumerate(bands)]
    spec = {"op": "normalized_difference", "pair": [0, 1], "minus": [2, 3], "offset": -1000, "nodata": 0}
    p = Pipeline(open_source("stack://" + "|".join(paths)), [spec])
    assert p.info(0).shape == (1, 16, 16) and p.info(0).dtype == np.float32 and p.info(0).kind == "image"
    got = p.read(0, Box((0, 0, 0), (1, 16, 16)))
    r = [b.astype(np.float64) - 1000 for b in bands]
    want = (r[0] - r[1]) / (r[0] + r[1]) - (r[2] - r[3]) / (r[2] + r[3])
    assert np.isnan(got[0, 0, 0])
    np.testing.assert_allclose(got.ravel()[1:], want.ravel()[1:], rtol=1e-5, atol=1e-6)


def test_it_needs_the_channels_it_names(tmp_path):
    a = write(tmp_path / "a", np.ones((1, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="channels"):
        Pipeline(open_source(f"stack://{a}|{a}"), [{"op": "normalized_difference", "minus": [2, 3]}])
