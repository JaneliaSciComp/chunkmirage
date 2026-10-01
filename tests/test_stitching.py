"""Stitching: interest points, matching, RANSAC, the global fit and fusion, and stitch://
on a BigStitcher project of OME-Zarr tiles cut from one volume at known offsets."""

import json

import numpy as np
import pytest

from chunkmirage import stitching as S
from chunkmirage.core import Box

SHIFT = np.array([0.0, 0.0, 6.0])  # tile 1's stage position is this far off, in voxels (x)


def blobs(shape=(24, 96, 200), n=400, seed=1) -> np.ndarray:
    """Gaussian blobs of a few voxels, at random: something to find interest points in."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    out = np.zeros(shape, np.float32)
    for c in rng.uniform([2, 2, 2], np.array(shape) - 2, (n, 3)):
        d2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        near = d2 < 25
        out[near] += np.exp(-d2[near] / 4.0)
    return (np.clip(out, 0, 1) * 200 + 10).astype(np.uint8)


def tiles_of(vol: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    """Two tiles overlapping by 50 voxels in x; the second's stage position SHIFT off."""
    a, b = vol[:, :, :120], vol[:, :, 70:]
    stage = lambda x: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, x]]  # noqa: E731
    tiles = [
        {"name": "a", "shape": [list(a.shape)], "stage": stage(0.0)},
        {"name": "b", "shape": [list(b.shape)], "stage": stage(70.0 + SHIFT[2])},
    ]
    return tiles, [a, b]


def find(tiles, data, p):
    points = []
    for i, j, lo, hi in S.overlaps(tiles, p.margin):
        both = []
        for t in (i, j):
            start, stop = S.region(tiles[t], 0, lo, hi)
            block = data[t][tuple(slice(s, e) for s, e in zip(start, stop))]
            both.append(S.points_in(tiles[t], 0, block, start, [1, 1, 1], lo, hi, p))
        points.append(both)
    return points


def test_interest_points_are_found_to_a_fraction_of_a_voxel():
    vol = np.zeros((15, 30, 30), np.float32)
    z, y, x = np.meshgrid(*[np.arange(15), np.arange(30), np.arange(30)], indexing="ij")
    centre = np.array([7.3, 14.6, 15.2])
    vol += np.exp(-((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2) / 8)
    pts = S.detect(vol, [1, 1, 1], 1.5, 0.01, 0.0, 1.0)
    assert len(pts) == 1
    assert np.abs(pts[0] - centre).max() < 0.25


def test_ransac_keeps_the_agreeing_matches():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 100, (60, 3))
    q = p + [1.0, -2.0, 3.0] + rng.normal(0, 0.2, p.shape)
    bad = rng.random(60) < 0.3
    q[bad] = rng.uniform(0, 100, (bad.sum(), 3))
    model, inliers = S.ransac(p, q, "translation", 2.0, 0.1, 5, 500)
    assert (inliers == ~bad).all()
    assert np.allclose(model[:, 3], [1, -2, 3], atol=0.1)
    assert S.ransac(p, q, "translation", 2.0, 0.9, 5, 500)[0] is None  # too few agree


@pytest.mark.parametrize("model", ["rigid", "affine"])
def test_fits_recover_a_motion(model):
    rng = np.random.default_rng(2)
    t = np.deg2rad(10)
    a = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    if model == "affine":
        a = a @ np.diag([1.0, 1.1, 0.95])
    p = rng.uniform(0, 50, (30, 3))
    q = p @ a.T + [4, 5, 6]
    found = S.fit(model, p, q)
    assert np.allclose(found[:, :3], a, atol=1e-6) and np.allclose(found[:, 3], [4, 5, 6], atol=1e-6)
    batch = S._fit_many(model, p[None, : S.MIN_MATCHES[model]], q[None, : S.MIN_MATCHES[model]])
    assert np.allclose(batch[0], found, atol=1e-5)


def test_tiles_are_stitched_and_fused_back_into_the_volume():
    vol = blobs()
    tiles, data = tiles_of(vol)
    p = S.StitchParams(sigma=1.5, threshold=0.02, margin=10)
    found = S.register(tiles, find(tiles, data, p), p)
    pair = found["pairs"][0]
    assert pair["kept"] and pair["inliers"] >= 20
    # the correction undoes the stage position's error, to a fraction of a voxel
    assert np.allclose(np.array(found["corrections"][1])[:, 3], -SHIFT, atol=0.2)
    grid = S.grids(tiles, found["placements"])[0]
    # a fraction of a voxel more on some sides: the tiles' fitted placements spill over
    assert all(0 <= g - v <= 1 for g, v in zip(grid["shape"], vol.shape))
    blocks = [(d, [0, 0, 0]) for d in data]
    fused = S.fuse(tiles, found["placements"], 0, grid, [0, 0, 0], grid["shape"], blocks, 20.0, "uint8")
    assert same(fused, grid, vol)  # seams and all


def same(fused, grid, vol) -> bool:
    """The fused voxels nearest the volume's match it, to interpolation's blur."""
    at = np.rint(-np.asarray(grid["origin"]) / grid["voxel"]).astype(int)  # vol's voxel 0
    part = fused[tuple(slice(a, a + n) for a, n in zip(at, vol.shape))]
    return part.shape == vol.shape and np.abs(part.astype(int) - vol).mean() < 1.5


def project(tmp_path, vol: np.ndarray) -> str:
    """A BigStitcher project of two OME-Zarr tiles (t, c, z, y, x), one channel, the second
    tile's stage position SHIFT off and a "Stitching Transform" on top that corrects it."""
    import tensorstore as ts

    tiles, data = tiles_of(vol)
    zgroups, setups, regs = [], [], []
    for k, d in enumerate(data):
        root = tmp_path / "dataset.ome.zarr" / f"s{k}-t0.zarr"
        root.mkdir(parents=True)
        (root / ".zgroup").write_text('{"zarr_format": 2}')
        axes = [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"}] + [
            {"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"
        ]
        ds = [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1, 1, 1]}]}]
        (root / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "axes": axes, "datasets": ds}]}))
        arr = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(root / "0")}},
                      create=True, dtype=ts.uint8, shape=(1, 1, *d.shape), chunk_layout=ts.ChunkLayout(chunk_shape=[1, 1, 16, 64, 64])).result()
        arr[0, 0].write(d).result()
        zgroups.append(f'<zgroup setup="{k}" tp="0" path="s{k}-t0.zarr" indicies="0 0" />')
        z, y, x = d.shape
        setups.append(f"<ViewSetup><id>{k}</id><name>{k}</name><size>{x} {y} {z}</size>"
                      f"<attributes><channel>0</channel><tile>{k}</tile></attributes></ViewSetup>")
        x0 = tiles[k]["stage"][2][3]
        fix = -SHIFT[2] if k else 0.0
        regs.append(
            f'<ViewRegistration timepoint="0" setup="{k}">'
            f"<ViewTransform type=\"affine\"><Name>Stitching Transform</Name><affine>1 0 0 {fix} 0 1 0 0 0 0 1 0</affine></ViewTransform>"
            f"<ViewTransform type=\"affine\"><Name>Translation to Regular Grid</Name><affine>1 0 0 {x0} 0 1 0 0 0 0 1 0</affine></ViewTransform>"
            "</ViewRegistration>"
        )
    xml = (
        '<SpimData version="0.2"><SequenceDescription>'
        f'<ImageLoader format="bdv.multimg.zarr" version="3.0"><zarr type="relative">dataset.ome.zarr</zarr><zgroups>{"".join(zgroups)}</zgroups></ImageLoader>'
        f'<ViewSetups>{"".join(setups)}</ViewSetups></SequenceDescription>'
        f'<ViewRegistrations>{"".join(regs)}</ViewRegistrations></SpimData>'
    )
    (tmp_path / "dataset.xml").write_text(xml)
    return str(tmp_path / "dataset.xml")


def test_bigstitcher_projects_are_read_without_their_stitching(tmp_path):
    xml = project(tmp_path, blobs())
    tiles = S.tiles_from_bdv(open(xml).read(), str(tmp_path), 0)
    assert [t["shape"][0] for t in tiles] == [[24, 96, 120], [24, 96, 130]]
    assert np.allclose(np.array(tiles[1]["stage"])[:, 3], [0, 0, 70 + SHIFT[2]])  # x y z -> z y x
    assert np.allclose(np.array(tiles[1]["reference"])[:, 3], [0, 0, 70])
    assert tiles[0]["url"].endswith("dataset.ome.zarr/s0-t0.zarr")
    with pytest.raises(ValueError, match="no tiles of channel 3"):
        S.tiles_from_bdv(open(xml).read(), str(tmp_path), 3)


def test_stitch_source_serves_the_fused_tiles(tmp_path):
    from chunkmirage.sources.registry import open_source

    vol = blobs()
    ms = open_source(f"stitch://{project(tmp_path, vol)}?level=0&sigma=1.5&threshold=0.02&margin=10")
    info = ms[0].info
    assert info.axes == ("z", "y", "x") and info.units[-1] == "micrometer"
    fused = ms[0].read(Box((0, 0, 0), info.shape))
    assert same(fused, ms[0].grid, vol)
    part = ms[0].read(Box((4, 10, 100), (8, 50, 140)))  # across the seam
    assert np.array_equal(part, fused[4:8, 10:50, 100:140])
    with pytest.raises(ValueError, match="unknown stitch:// parameters"):
        open_source(f"stitch://{tmp_path / 'dataset.xml'}?colour=red")
