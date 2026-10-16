"""warp:// sources: an image twisted by a procedural swirl, and the swirl itself as a layer."""

from urllib.parse import quote

import numpy as np
import pytest
import tensorstore as ts
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, gpu, open_source
from chunkmirage.core import Box
from chunkmirage.sources.warp import sampled_field
from chunkmirage.transforms import SwirlField, Swirls


def _read_all(src):
    return src.read(Box((0,) * src.info.ndim, src.info.shape))


def test_swirl_turns_points_about_the_axis_by_the_falloff_angle():
    f = SwirlField(centre=[0, 10, 20], radius=5, angle=60, plane=(1, 2))
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 30, size=(500, 3))
    moved = pts + f.sample(pts)
    np.testing.assert_allclose(moved[:, 0], pts[:, 0])  # the axis direction does not move
    before, after = pts[:, 1:] - [10, 20], moved[:, 1:] - [10, 20]
    np.testing.assert_allclose(np.linalg.norm(after, axis=1), np.linalg.norm(before, axis=1))
    turned = np.arctan2(after[:, 1], after[:, 0]) - np.arctan2(before[:, 1], before[:, 0])
    turned = (turned + np.pi) % (2 * np.pi) - np.pi
    expected = np.deg2rad(60) * np.exp(-np.sum(before**2, axis=1) / 25)
    np.testing.assert_allclose(turned, expected, atol=1e-9)


def test_a_3d_swirl_turns_points_about_its_own_axis_fading_from_its_centre():
    centre, axis = np.array([5.0, -3.0, 8.0]), np.array([1.0, 2.0, -2.0]) / 3
    f = Swirls([centre], [axis * 7], [6.0], [-80.0])  # axes need not be unit length
    pts = np.random.default_rng(1).uniform(-15, 25, size=(500, 3))
    d = f.sample(pts)
    v, w = pts - centre, pts + d - centre
    np.testing.assert_allclose(d @ axis, 0, atol=1e-9)  # no motion along the axis
    np.testing.assert_allclose(np.linalg.norm(w, axis=1), np.linalg.norm(v, axis=1))
    vp, wp = v - np.outer(v @ axis, axis), w - np.outer(w @ axis, axis)
    turned = np.arctan2(np.cross(vp, wp) @ axis, np.einsum("ij,ij->i", vp, wp))
    expected = np.deg2rad(-80) * np.exp(-np.sum(v**2, axis=1) / 36)
    np.testing.assert_allclose(turned, expected, atol=1e-9)
    assert (np.abs(d) > 1e-3).sum(axis=0).all()  # a tilted axis moves z, y and x

    # Along z with a huge radius it is the planar swirl; far swirls add nothing.
    flat = SwirlField(centre, 1e9, 70, (1, 2)).sample(pts)
    np.testing.assert_allclose(Swirls([centre], [[1, 0, 0]], [1e9], [70]).sample(pts), flat)
    far = Swirls([centre, [1e4, 0, 0]], [axis, [0, 1, 0]], [6.0, 6.0], [-80.0, 90.0])
    np.testing.assert_array_equal(far.sample(pts), d)


def test_zero_angle_is_the_identity(zarr2_path, volume):
    lvl = open_source(f"warp://{zarr2_path}?field=swirl&angle=0").levels[0]
    assert lvl.info == open_source(zarr2_path).levels[0].info
    np.testing.assert_array_equal(_read_all(lvl), volume)


def test_a_uniform_half_turn_flips_the_plane(zarr2_path, volume):
    """With a huge radius the swirl is a rigid 180 degree turn about the volume centre,
    which maps voxel centres onto voxel centres: the result is exact."""
    src = open_source(f"warp://{zarr2_path}?field=swirl&angle=180&radius=1e9")
    np.testing.assert_array_equal(_read_all(src.levels[0]), volume[:, ::-1, ::-1])
    np.testing.assert_array_equal(
        src.levels[0].read(Box((3, 5, 7), (9, 21, 40))), volume[3:9, ::-1, ::-1][:, 5:21, 7:40]
    )
    xz = open_source(f"warp://{zarr2_path}?field=swirl&angle=180&radius=1e9&plane=z,x")
    np.testing.assert_array_equal(_read_all(xz.levels[0]), volume[::-1, :, ::-1])
    assert len(src.levels) == 2  # every level is warped, each on its own grid


def test_leading_channel_axes_pass_through(tmp_path):
    data = np.random.default_rng(3).integers(0, 255, size=(2, 9, 11, 13), dtype=np.uint8)
    arr = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(tmp_path / "c.zarr")}},
        create=True,
        dtype=ts.uint8,
        shape=data.shape,
    ).result()
    arr.write(data).result()
    lvl = open_source(f"warp://{tmp_path / 'c.zarr'}?field=swirl&angle=180&radius=1e9").levels[0]
    assert lvl.info.axes == ("c", "z", "y", "x")
    np.testing.assert_array_equal(_read_all(lvl), data[:, :, ::-1, ::-1])


def test_the_field_is_served_as_three_components(zarr2_path):
    url = f"warp://{zarr2_path}?field=swirl&angle=75&radius=120"
    lvl = open_source(url + "&show=field").levels[0]
    assert lvl.info.shape == (3, 40, 50, 70)
    assert lvl.info.axes == ("c", "z", "y", "x") and lvl.info.dtype == np.float32
    vectors = _read_all(lvl)
    z, y, x = np.meshgrid(*[np.arange(n) * 8.0 for n in (40, 50, 70)], indexing="ij")
    pts = np.stack([z, y, x], -1).reshape(-1, 3)
    centre = (np.array([40, 50, 70]) - 1) / 2 * 8
    expected = SwirlField(centre, 120, 75, (1, 2)).sample(pts).T.reshape(3, 40, 50, 70)
    np.testing.assert_allclose(vectors, expected, atol=1e-4)
    assert not vectors[0].any()  # dz: the default plane is (y, x)
    np.testing.assert_allclose(
        lvl.read(Box((1, 2, 3, 4), (3, 6, 9, 20))), expected[1:3, 2:6, 3:9, 4:20], atol=1e-4
    )

    # The precomputed frontend turns the component axis into channels (Neuroglancer: c^).
    client = TestClient(create_app({"field": Pipeline.from_spec({"source": url + "&show=field"})}))
    info = client.get("/field/precomputed/info").json()
    assert info["num_channels"] == 3 and info["data_type"] == "float32"
    r = client.get("/field/precomputed/s0/0-32_0-16_0-16", headers={"accept-encoding": "identity"})
    assert r.status_code == 200 and len(r.content) == 3 * 16 * 16 * 32 * 4


def test_parameters_are_part_of_the_cache_key(zarr2_path):
    a = Pipeline.from_spec({"source": f"warp://{zarr2_path}?field=swirl&angle=30"})
    b = Pipeline.from_spec({"source": f"warp://{zarr2_path}?field=swirl&angle=60"})
    a2 = Pipeline.from_spec({"source": f"warp://{zarr2_path}?field=swirl&angle=30"})
    assert a.digest() != b.digest() and a.digest() == a2.digest()


def test_the_image_url_may_carry_its_own_query():
    url = "warp://synthetic://blobs?shape=32,32,32&levels=1?field=swirl&angle=180&radius=1e9"
    lvl = open_source(url).levels[0]
    plain = open_source("synthetic://blobs?shape=32,32,32&levels=1").levels[0]
    np.testing.assert_array_equal(_read_all(lvl), _read_all(plain)[:, ::-1, ::-1])


def test_frames_twist_from_nothing_to_the_angle_along_t(zarr2_path, volume):
    base = f"warp://{zarr2_path}?field=swirl&angle=180&radius=1e9"
    lvl = open_source(base + "&frames=3").levels[0]
    assert lvl.info.axes == ("t", "z", "y", "x") and lvl.info.shape == (3, *volume.shape)
    assert lvl.info.chunk_shape[0] == 1 and lvl.info.units[0] == ""
    frames = _read_all(lvl)
    np.testing.assert_array_equal(frames[0], volume)  # frame 0: no twist
    np.testing.assert_array_equal(frames[2], volume[:, ::-1, ::-1])  # last: the full angle
    half = _read_all(open_source(f"warp://{zarr2_path}?field=swirl&angle=90&radius=1e9").levels[0])
    np.testing.assert_array_equal(frames[1], half)
    np.testing.assert_array_equal(
        lvl.read(Box((1, 3, 5, 7), (3, 9, 21, 40))), frames[1:3, 3:9, 5:21, 7:40]
    )
    field = open_source(base + "&frames=3&show=field").levels[0]
    assert field.info.axes == ("t", "c", "z", "y", "x") and field.info.shape[:2] == (3, 3)
    vectors = _read_all(field)
    assert not vectors[0].any() and vectors[2].any()


def test_frames_replace_a_single_time_point(tmp_path):
    data = np.random.default_rng(4).integers(0, 255, size=(1, 2, 6, 8, 10), dtype=np.uint8)
    path = tmp_path / "t.zarr"
    arr = ts.open(
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
        create=True,
        dtype=ts.uint8,
        shape=data.shape,
    ).result()
    arr.write(data).result()
    (path / ".zattrs").write_text('{"axis_names": ["t", "c", "z", "y", "x"]}')
    lvl = open_source(f"warp://{path}?field=swirl&angle=180&radius=1e9&frames=2").levels[0]
    assert lvl.info.axes == ("t", "c", "z", "y", "x") and lvl.info.shape == (2, 2, 6, 8, 10)
    np.testing.assert_array_equal(_read_all(lvl), [data[0], data[0][:, :, ::-1, ::-1]])

    many = tmp_path / "many.zarr"
    ts.open(
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(many)}},
        create=True,
        dtype=ts.uint8,
        shape=(3, 6, 8, 10),
    ).result()
    (many / ".zattrs").write_text('{"axis_names": ["t", "z", "y", "x"]}')
    with pytest.raises(ValueError, match="one time point"):
        open_source(f"warp://{many}?field=swirl&frames=2")


def test_output_chunks_can_be_chosen(zarr2_path, volume):
    lvl = open_source(f"warp://{zarr2_path}?field=swirl&angle=40&chunk=4,32,32").levels[0]
    assert lvl.info.chunk_shape == (4, 32, 32)
    same = open_source(f"warp://{zarr2_path}?field=swirl&angle=40").levels[0]
    np.testing.assert_array_equal(_read_all(lvl), _read_all(same))
    with pytest.raises(ValueError, match="chunk needs 3"):
        open_source(f"warp://{zarr2_path}?field=swirl&chunk=4,32")


def test_many_swirls_are_random_but_reproducible(zarr2_path, volume):
    url = f"warp://{zarr2_path}?field=swirls&count=5&radius=100&angle=120"
    seeded = [Pipeline.from_spec({"source": url + s}).digest() for s in ("", "&seed=0", "&seed=1")]
    assert seeded[0] == seeded[1] != seeded[2]
    lvl = open_source(url + "&frames=3").levels[0]
    frames = _read_all(lvl)
    np.testing.assert_array_equal(frames[0], volume)
    assert (frames[2] != volume).mean() > 0.2
    np.testing.assert_array_equal(frames[2], _read_all(open_source(url).levels[0]))
    vectors = _read_all(open_source(url + "&show=field").levels[0])
    assert all(np.abs(c).max() > 1 for c in vectors)  # z, y and x all move
    # Their centres fall in the box centre +- spread.
    tight = _read_all(open_source(url + "&centre=0,0,0&spread=1&show=field").levels[0])
    assert np.abs(tight[:, -1, -1, -1]).max() < 1e-6 < np.abs(tight[:, 0, 0, 0]).max()


def test_bad_urls_explain_themselves(zarr2_path):
    with pytest.raises(ValueError, match="field=swirl"):
        open_source(f"warp://{zarr2_path}?angle=3")
    with pytest.raises(ValueError, match="after the image URL"):  # warp parameters forgotten
        open_source("warp://synthetic://blobs?shape=32,32,32")
    with pytest.raises(ValueError, match="do not apply to field=swirls"):
        open_source(f"warp://{zarr2_path}?field=swirls&plane=y,x")
    with pytest.raises(ValueError, match="do not apply to field=swirl"):
        open_source(f"warp://{zarr2_path}?field=swirl&count=3")
    with pytest.raises(ValueError, match="unknown warp:// parameters"):
        open_source(f"warp://{zarr2_path}?field=swirl&spin=3")
    with pytest.raises(ValueError, match="plane"):
        open_source(f"warp://{zarr2_path}?field=swirl&plane=y,q")
    with pytest.raises(ValueError, match="frames"):
        open_source(f"warp://{zarr2_path}?field=swirl&frames=0")


# ------------------------------------------------ a field from any source, an affine, a target
IMG = "synthetic://julia?shape=32,48,48&chunk=16,16,16&levels=1&seed=2&voxel_size=1"  # textured (blobs are empty this small)


def test_a_field_read_from_a_source_reproduces_the_procedural_warp():
    swirl = f"warp://{IMG}?field=swirl&angle=40"
    # the swirl's own displacement, read back as a stored field would be (its URL encoded)
    field = quote(f"{swirl}&show=field", safe="")
    sampled = open_source(f"warp://{IMG}?field={field}")
    assert sampled.name.endswith("~warped") and sampled.levels[0].info.shape == (32, 48, 48)
    box = Box((0, 0, 0), (32, 48, 48))
    # exact at the field's voxel centres, so the mapping is the swirl's own
    np.testing.assert_allclose(sampled.levels[0].read(box).astype(float), open_source(swirl).levels[0].read(box).astype(float), atol=1)
    other = quote(f"warp://{IMG}?field=swirl&angle=41&show=field", safe="")
    assert open_source(f"warp://{IMG}?field={other}").levels[0].cache_key() != sampled.levels[0].cache_key()


def test_an_affine_after_the_field_moves_what_is_read():
    img = "synthetic://blobs?shape=16,32,32&chunk=16,16,16&levels=1&seed=2&voxel_size=1"
    # no displacement, then 8 units further along x: whole voxels, so the values are the image's
    shifted = open_source(f"warp://{img}?field=swirl&angle=0&affine=1,0,0,0,0,1,0,0,0,0,1,8")
    box = Box((0, 0, 0), (16, 32, 32))
    raw, out = open_source(img).levels[0].read(box), shifted.levels[0].read(box)
    np.testing.assert_array_equal(out[:, :, :24], raw[:, :, 8:])
    assert (out[:, :, 24:] == 0).all()  # beyond the image


def test_a_target_gives_the_output_grid_and_reads_the_level_that_suits_it():
    img = "synthetic://noise?shape=64,64,64&chunk=32,32,32&levels=2&seed=2&voxel_size=8"
    # another source on the grid of the image's own second level: 16 nm voxels over the same extent
    coarse = "synthetic://noise?shape=32,32,32&chunk=8,8,8&levels=1&seed=7&voxel_size=16"
    out = open_source(f"warp://{img}?field=swirl&angle=0&target={quote(coarse, safe='')}")
    want = open_source(coarse).levels[0].info
    got = out.levels[0].info
    assert len(out.levels) == 1 and (got.shape, got.voxel_size, got.translation, got.chunk_shape) == (want.shape, want.voxel_size, want.translation, want.chunk_shape)
    # with nothing displaced, the output's voxels are the image's second level's, read from it (not resampled from the first)
    box = Box((0, 0, 0), (32, 32, 32))
    np.testing.assert_array_equal(out.levels[0].read(box), open_source(img).levels[1].read(box))


def test_bad_fields_explain_themselves():
    img = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1&voxel_size=1"
    with pytest.raises(ValueError, match="components"):
        open_source(f"warp://{img}?field={quote(img, safe='')}")  # a 3-D image is not a (c, z, y, x) field
    with pytest.raises(ValueError, match="apply to field=swirl"):
        open_source(f"warp://{img}?field={quote(img, safe='')}&frames=3")
    with pytest.raises(ValueError, match="affine needs 12"):
        open_source(f"warp://{img}?field=swirl&affine=1,2,3")


# ------------------------------------------------ an affine alone, and several fields at once
def test_an_affine_or_a_target_alone_places_the_image_without_a_field():
    img = "synthetic://blobs?shape=16,32,32&chunk=16,16,16&levels=1&seed=2&voxel_size=1"
    shifted = open_source(f"warp://{img}?affine=1,0,0,0,0,1,0,0,0,0,1,8")
    box = Box((0, 0, 0), (16, 32, 32))
    raw, out = open_source(img).levels[0].read(box), shifted.levels[0].read(box)
    np.testing.assert_array_equal(out[:, :, :24], raw[:, :, 8:])
    assert shifted.name.endswith("~warped")
    # a target alone: the image on another grid, read from the level that suits it
    img2 = "synthetic://noise?shape=64,64,64&chunk=32,32,32&levels=2&seed=2&voxel_size=8"
    coarse = "synthetic://noise?shape=32,32,32&chunk=8,8,8&levels=1&seed=7&voxel_size=16"
    placed = open_source(f"warp://{img2}?target={quote(coarse, safe='')}")
    box2 = Box((0, 0, 0), (32, 32, 32))
    np.testing.assert_array_equal(placed.levels[0].read(box2), open_source(img2).levels[1].read(box2))
    with pytest.raises(ValueError, match="needs a field"):
        open_source(f"warp://{img}?interpolation=nearest")
    with pytest.raises(ValueError, match="show=field needs a field"):
        open_source(f"warp://{img}?affine=1,0,0,0,0,1,0,0,0,0,1,8&show=field")
    with pytest.raises(ValueError, match="a warp with no field"):
        open_source(f"warp://{img}?affine=1,0,0,0,0,1,0,0,0,0,1,8&angle=3")


# two passes' fields: the first pass's, and the second's, fitted on the image the first moved
PASS1 = f"warp://{IMG}?field=swirl&angle=60&centre=16,20,24&radius=16&show=field"
PASS2 = f"warp://{IMG}?field=swirl&angle=-45&centre=16,28,20&radius=14&show=field"
BOX = Box((0, 0, 0), (32, 48, 48))


def test_several_fields_compose_before_one_resample():
    p1, p2 = quote(PASS1, safe=""), quote(PASS2, safe="")
    composed = open_source(f"warp://{IMG}?field={p2}|{p1}")  # the latest pass's field first
    nested = open_source(f"warp://warp://{IMG}?field={p1}?field={p2}")  # the warped image warped again
    c, n = composed.levels[0].read(BOX).astype(float), nested.levels[0].read(BOX).astype(float)
    # the same mapping (a grey level apart on average, of 254), the original interpolated once
    # instead of twice, so sharper: more gradient, away from the edges
    assert np.abs(c - n).mean() < 3, np.abs(c - n).mean()
    inner = (slice(4, -4),) * 3
    assert np.abs(np.diff(c[inner], axis=-1)).mean() > np.abs(np.diff(n[inner], axis=-1)).mean()
    # the order is the order given: the other way round is another image, under another key
    other = open_source(f"warp://{IMG}?field={p1}|{p2}")
    assert other.levels[0].cache_key() != composed.levels[0].cache_key()
    assert np.abs(other.levels[0].read(BOX).astype(float) - c).mean() > 3


def test_the_net_displacement_of_several_fields_is_served():
    p1, p2 = quote(PASS1, safe=""), quote(PASS2, safe="")
    net = open_source(f"warp://{IMG}?field={p2}|{p1}&show=field")
    got = net.levels[0].read(Box((0, 0, 0, 0), (3, *BOX.shape)))
    # T_1(T_2(x)) - x, each field sampled as the warp samples it: the second pass's at x, the
    # first's where that moved the point; not the sum of the two at x
    d1 = sampled_field(open_source(PASS1).levels[0], "1")
    d2 = sampled_field(open_source(PASS2).levels[0], "2")
    pts = np.indices(BOX.shape, dtype=float).reshape(3, -1).T  # voxels of 1 at the origin
    moved = pts + d2.sample(pts)
    want = (moved + d1.sample(moved) - pts).T.reshape(3, *BOX.shape)
    np.testing.assert_allclose(got, want, atol=1e-4)
    summed = (d1.sample(pts) + d2.sample(pts)).T.reshape(3, *BOX.shape)
    assert np.abs(want - summed).max() > 0.5


def test_the_gpu_takes_chains_of_affines_and_sampled_fields_only():
    # with CuPy (serve --device cuda) the resampler's own code runs on the GPU; a chain with
    # anything else stays on the CPU. No GPU here: what is checked is the choice.
    from chunkmirage.sources.scene import _on_gpu

    p1 = quote(PASS1, safe="")
    assert _on_gpu(open_source(f"warp://{IMG}?field={p1}&affine=1,0,0,0.5,0,1,0,0,0,0,1,0").levels[0].mapping)
    assert _on_gpu(open_source(f"warp://{IMG}?affine=1,0,0,0.5,0,1,0,0,0,0,1,0").levels[0].mapping)
    assert not _on_gpu(open_source(f"warp://{IMG}?field=swirl&angle=40").levels[0].mapping)
    gpu.set_device("auto")  # no CuPy, or no GPU: the CPU
    assert gpu.device() is None
