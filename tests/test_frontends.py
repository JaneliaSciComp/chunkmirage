import gzip
import struct

import numcodecs
import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.ops import Threshold


@pytest.fixture(scope="module")
def client(zarr2_path):
    pipe = Pipeline(open_source(zarr2_path), [Threshold(low=128)])
    plain = Pipeline(open_source(zarr2_path), [])
    app = create_app({"thr": pipe, "raw": plain})
    return TestClient(app)


def expected(volume):
    return (volume >= 128).astype(np.uint8)


def test_index_and_api(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert set(body["datasets"]) == {"thr", "raw"}
    assert set(body["formats"]) == {"n5", "zarr", "zarr3", "precomputed"}
    assert "threshold" in client.get("/api/ops").json()
    ng = client.get("/api/datasets/thr/neuroglancer?format=n5").json()
    assert ng["source"].startswith("n5://http://testserver/thr/@")
    assert ng["url"].startswith("https://neuroglancer-demo.appspot.com/#!")


def test_n5(client, volume):
    root = client.get("/thr/n5/attributes.json").json()
    assert root["scales"] == [[1, 1, 1], [2, 2, 2]]
    assert root["axes"] == ["x", "y", "z"]
    attrs = client.get("/thr/n5/s0/attributes.json").json()
    assert attrs["dimensions"] == list(volume.shape[::-1])
    assert attrs["dataType"] == "uint8"
    assert attrs["transform"]["axes"] == ["z", "y", "x"]  # COSEM transform lists are C order
    bs = attrs["blockSize"]  # x, y, z
    # chunk index (x=1, y=2, z=0)
    r = client.get("/thr/n5/s0/1/2/0")
    assert r.status_code == 200
    _mode, ndim = struct.unpack(">HH", r.content[:4])
    dims = struct.unpack(f">{ndim}I", r.content[4 : 4 + 4 * ndim])
    data = np.frombuffer(gzip.decompress(r.content[4 + 4 * ndim :]), dtype=">u1").reshape(
        dims[::-1]
    )
    exp = expected(volume)[0 : bs[2], 2 * bs[1] : 3 * bs[1], bs[0] : 2 * bs[0]]
    np.testing.assert_array_equal(data, exp)
    # out-of-range chunk
    assert client.get("/thr/n5/s0/99/0/0").status_code == 404
    # versioned URL works too
    assert client.get("/thr/@abc123/n5/s0/attributes.json").status_code == 200


def test_zarr2(client, volume):
    zattrs = client.get("/thr/zarr/.zattrs").json()
    assert zattrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"] == [
        16,
        16,
        16,
    ]
    zarray = client.get("/thr/zarr/s0/.zarray").json()
    assert zarray["shape"] == list(volume.shape)
    chunks = zarray["chunks"]
    codec = numcodecs.get_codec(zarray["compressor"])
    idx = (2, 3, 2)  # z, y, x  (edge chunk in z: 40/16 -> last index 2, partial)
    r = client.get("/thr/zarr/s0/" + "/".join(map(str, idx)))
    assert r.status_code == 200
    data = np.frombuffer(codec.decode(r.content), dtype=zarray["dtype"]).reshape(chunks)
    box = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunks))
    exp = expected(volume)[box]
    np.testing.assert_array_equal(data[tuple(slice(0, s) for s in exp.shape)], exp)
    assert (data[exp.shape[0] :] == 0).all()  # zero padded
    # dot separator accepted as well
    assert client.get("/thr/zarr/s0/0.0.0").status_code == 200
    # consolidated metadata
    assert "s1/.zarray" in client.get("/thr/zarr/.zmetadata").json()["metadata"]


def test_zarr3(client, volume):
    root = client.get("/thr/zarr3/zarr.json").json()
    assert root["node_type"] == "group"
    assert root["attributes"]["ome"]["version"] == "0.5"
    meta = client.get("/thr/zarr3/s1/zarr.json").json()
    assert meta["data_type"] == "uint8"
    assert meta["dimension_names"] == ["z", "y", "x"]
    cs = meta["chunk_grid"]["configuration"]["chunk_shape"]
    r = client.get("/thr/zarr3/s1/c/0/0/0")
    assert r.status_code == 200
    data = np.frombuffer(gzip.decompress(r.content), dtype=np.uint8).reshape(cs)
    exp = expected(volume[::2, ::2, ::2])[: cs[0], : cs[1], : cs[2]]
    np.testing.assert_array_equal(data[: exp.shape[0], : exp.shape[1], : exp.shape[2]], exp)


def test_precomputed(client, volume):
    info = client.get("/raw/precomputed/info").json()
    assert info["data_type"] == "uint8"
    assert info["num_channels"] == 1
    s0 = info["scales"][0]
    assert s0["size"] == list(volume.shape[::-1])
    assert s0["resolution"] == [8, 8, 8]
    cx, cy, _cz = s0["chunk_sizes"][0]
    # last chunk in z (40 voxels, chunk 16 -> 32..40)
    key = f"s0/{cx}-{2 * cx}_0-{cy}_32-40"
    r = client.get(f"/raw/precomputed/{key}", headers={"accept-encoding": "identity"})
    assert r.status_code == 200, r.text
    data = np.frombuffer(r.content, dtype=np.uint8).reshape(8, cy, cx)
    np.testing.assert_array_equal(data, volume[32:40, 0:cy, cx : 2 * cx])
    # misaligned request
    assert client.get("/raw/precomputed/s0/1-2_0-1_0-1").status_code == 404


def test_live_edit_changes_output(client, volume, zarr2_path):
    r = client.put(
        "/api/datasets/thr", json={"source": zarr2_path, "ops": [{"op": "threshold", "low": 200}]}
    )
    assert r.status_code == 200, r.text
    r = client.get("/thr/precomputed/s0/0-32_0-16_0-16", headers={"accept-encoding": "identity"})
    data = np.frombuffer(r.content, dtype=np.uint8).reshape(16, 16, 32)
    np.testing.assert_array_equal(data, (volume[:16, :16, :32] >= 200).astype(np.uint8))
    stats = client.get("/api/cache").json()
    assert stats["entries"] > 0
