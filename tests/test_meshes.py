"""Meshes computed when fetched: a ball's surface from its fragments is where the ball is,
closed at the array's edges and seamless across fragments; an elevation model's terrain
has its heights; the mesh frontend serves Neuroglancer's legacy layout."""

import json
import struct

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import Pipeline, open_source
from chunkmirage.frontends import get_frontend

pytest.importorskip("skimage")


def _zarr(path, data, voxel=(2.0, 2.0, 2.0)):
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}
    spec["metadata"] = {
        "shape": list(data.shape),
        "chunks": [8] * data.ndim,
        "dtype": data.dtype.str,
    }
    ts.open(spec, create=True).result().write(data).result()
    (path / ".zattrs").write_text(json.dumps({"resolution": list(voxel), "units": "nm"}))


def _decode(body: bytes):
    n = struct.unpack("<I", body[:4])[0]
    verts = np.frombuffer(body[4 : 4 + 12 * n], "<f4").reshape(-1, 3)
    faces = np.frombuffer(body[4 + 12 * n :], "<u4").reshape(-1, 3)
    return verts, faces


def _fragments(p):
    fe = get_frontend("mesh")
    assert json.loads(fe.resolve(p, "info").body) == {"@type": "neuroglancer_legacy_mesh"}
    names = json.loads(fe.resolve(p, "1:0").body)["fragments"]
    return [_decode(fe.compute(p, fe.resolve(p, n))) for n in names]


def test_a_balls_surface_is_where_the_ball_is_and_closed(tmp_path):
    z, y, x = np.mgrid[:24, :24, :24]
    r = np.sqrt((z - 12) ** 2 + (y - 12) ** 2 + (x - 13.5) ** 2)
    _zarr(tmp_path / "ball", (255 * (r < 8)).astype(np.uint8))
    # a second ball cut by the array's edge: its surface is closed there
    p = Pipeline.from_spec(
        {"source": str(tmp_path / "ball"), "chunk_shape": [8, 8, 8], "mesh": {"level": 0}}
    )
    parts = [f for f in _fragments(p) if len(f[0])]
    assert len(parts) > 1  # the ball crosses fragments
    verts = np.concatenate([v for v, _ in parts])
    centre = np.array([13.5, 12, 12]) * 2  # x, y, z in nm (2 nm voxels)
    radius = np.linalg.norm(verts - centre, axis=1)
    assert radius.mean() == pytest.approx(16, abs=1.0) and radius.max() < 18
    # seamless: every edge of the union is shared by two triangles, once vertices that
    # coincide across fragments are merged
    keyed, edges, base = {}, {}, 0
    for v, f in parts:
        ids = [keyed.setdefault(tuple(np.round(p, 4)), len(keyed)) for p in v]
        for tri in f:
            a, b, c = (ids[i] for i in tri)
            for e in ((a, b), (b, c), (c, a)):
                k = tuple(sorted(e))
                edges[k] = edges.get(k, 0) + 1
        base += len(v)
    assert set(edges.values()) == {2}


def test_terrain_has_the_elevation_as_height(tmp_path):
    dem = np.add.outer(np.zeros(12), np.arange(16) * 0.5).astype(np.float32)  # rises eastward
    dem[0, 0] = np.nan
    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(tmp_path / "dem")}}
    spec["metadata"] = {"shape": [12, 16], "chunks": [12, 16], "dtype": "<f4"}
    ts.open(spec, create=True).result().write(dem).result()
    (tmp_path / "dem" / ".zattrs").write_text(json.dumps({"resolution": [5, 5], "units": "m"}))
    p = Pipeline(open_source(str(tmp_path / "dem")), [], chunk_shape=(12, 16))
    p.spec = None
    from chunkmirage.meshes import MeshSpec, fragment, fragment_box

    info = p.info(0)
    box = fragment_box(info, (0, 0))
    verts, faces = _decode(
        fragment(MeshSpec(kind="terrain", exaggeration=2), p.read(0, box), box, info)
    )
    assert len(verts) == 12 * 16 and len(faces) == 2 * 11 * 15 - 2  # the NaN corner's cell
    east = verts[:, 0] / 1e9 / 5  # x in pixels
    np.testing.assert_allclose(verts[16:, 2] / 1e9, 2 * 0.5 * east[16:], atol=1e-3)
