"""Zarr metadata tensorstore would reject is read (worldarray tests the ways into a bucket)."""

import json

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import open_source
from chunkmirage.core import Box


def test_a_zarr_compressor_with_members_tensorstore_rejects_is_read(tmp_path):
    path = tmp_path / "zstd.zarr"
    meta = {"shape": [4, 6], "chunks": [4, 6], "dtype": "<u2", "compressor": {"id": "zstd", "level": 1}}
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}
    data = np.arange(24, dtype=np.uint16).reshape(4, 6)
    ts.open({**spec, "metadata": meta}, create=True).result().write(data).result()
    zarray = json.loads((path / ".zarray").read_text())
    zarray["compressor"]["checksum"] = False  # as numcodecs >= 0.13 writes it
    (path / ".zarray").write_text(json.dumps(zarray))
    with pytest.raises(ValueError, match="extra members"):
        ts.open(spec, read=True).result()
    src = open_source(str(path))[0]
    np.testing.assert_array_equal(src.read(Box((0, 0), (4, 6))), data)
