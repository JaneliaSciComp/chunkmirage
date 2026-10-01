"""Meshes computed when fetched: a ball's surface from its fragments is where the ball is,
closed at the array's edges and seamless across fragments; an elevation model's terrain
has its heights; the mesh frontend serves Neuroglancer's legacy layout."""

import json
import struct

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import Pipeline, open_source
from chunkmirage.core import Box
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


def test_multiresolution_meshes_list_nodes_near_the_surface_and_serve_them_by_range():
    pytest.importorskip("DracoPy")
    import DracoPy
    from starlette.testclient import TestClient

    from chunkmirage import meshes
    from chunkmirage.cache import LRUCache
    from chunkmirage.server import DatasetRegistry, create_app

    src = "synthetic://mandelbulb?shape=128,128,128&levels=3&voxel_size=4&unit=nm"
    p = Pipeline.from_spec({"source": src, "chunk_shape": [16, 16, 16], "mesh": {"threshold": 255, "level": 2, "lods": 3}})
    infos = [p.info(i) for i in range(p.num_levels)]
    spec = p.spec.mesh
    assert meshes.lod_levels(infos, spec) == [0, 1, 2]
    fe = get_frontend("mesh")
    assert json.loads(fe.resolve(p, "info").body)["@type"] == "neuroglancer_multilod_draco"
    index = fe.resolve(p, "1.index").body
    shape = struct.unpack("<3f", index[:12])
    n_lods = struct.unpack("<I", index[24:28])[0]
    scales = struct.unpack(f"<{n_lods}f", index[28 : 28 + 4 * n_lods])
    counts = struct.unpack(f"<{n_lods}I", index[28 + 16 * n_lods : 28 + 20 * n_lods])
    assert n_lods == 3 and shape == (64.0, 64.0, 64.0) and scales == (4.0, 8.0, 16.0)
    assert len(index) == 28 + 20 * n_lods + 16 * sum(counts)
    # nodes only near the surface: the finest level's fewer than its grid's 8³, and every one's
    # parent listed
    assert 0 < counts[2] <= 8 and counts[0] < 8**3
    mask = p.read(2, Box((0, 0, 0), infos[2].shape)) >= 255
    nodes = meshes.multires_nodes(spec, meshes.surface_band(mask), infos, (16, 16, 16))
    # the band done in parts, each with a border, joins to the whole level's
    parts = np.zeros(mask.shape, bool)
    for lo in np.ndindex(2, 2, 2):
        a = np.array(lo) * 16
        b = np.maximum(a - meshes.BAND, 0), np.minimum(a + 16 + meshes.BAND, 32)
        core = Box(tuple(a - b[0]), tuple(a - b[0] + 16))
        parts[tuple(slice(x, x + 16) for x in a)] = meshes.surface_band(mask[tuple(slice(*x) for x in zip(*b))], core)
    assert np.array_equal(parts, meshes.surface_band(mask))
    parents = {tuple(n) for n in nodes[1]}
    assert all(tuple(c // 2) in parents for c in nodes[0])
    # a node's fragment: integers across the node, no triangle across its octants, padded
    level, node = 1, tuple(nodes[1][len(nodes[1]) // 2])
    box = meshes.node_box(infos[level], node, (16, 16, 16))
    v, f = meshes.multires_fragment(spec, p.read(level, box), box, infos[level], (16, 16, 16))
    assert len(f) and v.max() <= 2**16 - 1
    octant = (v[f] >= 2**15).astype(int)  # (triangles, corners, axes): which half per corner
    mid = v[f] == 2**15  # corners on a midplane belong to both halves
    assert ((octant.min(1) == octant.max(1)) | mid.any(1)).all()
    # served: the k-th node's bytes by range, 206, decodable, the rest zeros
    app = create_app(DatasetRegistry(LRUCache(1 << 26)))
    app.state.registry.add("bulb", {"source": src, "chunk_shape": [16, 16, 16], "mesh": {"threshold": 255, "level": 2, "lods": 3}})
    size = meshes.FRAGMENT_BYTES
    k = counts[0] // 2
    r = TestClient(app).get("/bulb/mesh/1", headers={"Range": f"bytes={k * size}-{(k + 1) * size - 1}"})
    assert r.status_code == 206 and r.headers["content-range"] == f"bytes {k * size}-{(k + 1) * size - 1}/{sum(counts) * size}"
    assert len(r.content) == size
    if r.content.strip(b"\0"):
        mesh = DracoPy.decode(r.content)
        assert np.asarray(mesh.points).max() <= 2**16 - 1
