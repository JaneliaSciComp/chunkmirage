"""register:// sources: a moving image registered onto a fixed one, solved when opened."""

from urllib.parse import quote

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import open_source
from chunkmirage.core import Box
from chunkmirage.sources import register

FIXED = "synthetic://blobs+noise?shape=24,64,64&chunk=24,32,32&levels=2&voxel_size=64"
SWIRL = "field=swirl&angle=25&radius=1100"
MOVING = f"warp://{FIXED}?{SWIRL}"  # the fixed image, twisted by a known swirl
# Near the faces the swirl brought in the zeros from outside the fixed image, which no
# registration can match; judge the fit inside.
INNER = (slice(4, -4), slice(4, -4), slice(4, -4))


def _url(extra: str = "", moving: str = MOVING, fixed: str = FIXED) -> str:
    url = f"register://{moving}?fixed={quote(fixed, safe='')}&device=cpu&iterations=40"
    return url + (f"&{extra}" if extra else "")


def _read(url_or_src, level: int = 0) -> np.ndarray:
    src = open_source(url_or_src).levels[level] if isinstance(url_or_src, str) else url_or_src
    return src.read(Box((0,) * src.info.ndim, src.info.shape))


def _corr(a, b) -> float:
    return float(np.corrcoef(np.ravel(a).astype(float), np.ravel(b).astype(float))[0, 1])


def test_the_affine_alone_needs_no_solve():
    # moving = fixed, the affine shifts by two voxels in y (64 nm each): the output is the
    # fixed image moved by two voxels, computed with no GPU or torch involved.
    shift = "1,0,0,0,0,1,0,128,0,0,1,0"
    out = _read(_url(f"affine={shift}&iterations=0", moving=FIXED))
    fixed = _read(FIXED)
    np.testing.assert_array_equal(out[:, :-2], fixed[:, 2:])
    assert not out[:, -2:].any()  # sampled beyond the moving image


def test_registration_undoes_a_known_swirl():
    pytest.importorskip("torch")
    fixed, moving, registered = (_read(u)[INNER] for u in (FIXED, MOVING, _url()))
    before, after = 1 - _corr(fixed, moving), 1 - _corr(fixed, registered)
    assert after < 0.3 * before, (before, after)
    # Where the swirl moves things most, the field points back the other way.
    u = _read(_url("show=field"))
    true = _read(f"warp://{FIXED}?{SWIRL}&show=field")
    big = np.linalg.norm(true, axis=0) > np.percentile(np.linalg.norm(true, axis=0), 75)
    cos = (u * -true).sum(0)[big] / (np.linalg.norm(u, axis=0) * np.linalg.norm(true, axis=0))[big]
    assert np.median(cos) > 0.7


def test_every_level_comes_from_the_same_field():
    pytest.importorskip("torch")
    src = open_source(_url())
    assert len(src.levels) == 2
    fine, coarse = _read(src.levels[0]), _read(src.levels[1])
    # the synthetic pyramid is exact subsampling (s1[i] == s0[2i]), so is its registration
    assert _corr(fine[::2, ::2, ::2], coarse) > 0.99


def test_pair_field_and_frames():
    pytest.importorskip("torch")
    registered = _read(_url())
    pair = open_source(_url("show=pair")).levels[0]
    assert pair.info.axes == ("c", "z", "y", "x") and pair.info.shape == (2, 24, 64, 64)
    both = _read(pair)
    np.testing.assert_array_equal(both[0], _read(FIXED))
    np.testing.assert_array_equal(both[1], registered)
    field = open_source(_url("show=field")).levels[0]
    assert field.info.shape == (3, 24, 64, 64) and field.info.dtype == np.float32

    frames = open_source(_url("frames=3")).levels[0]
    assert frames.info.axes == ("t", "z", "y", "x") and frames.info.shape[0] == 3
    video = _read(frames)
    np.testing.assert_array_equal(video[0], _read(MOVING))  # no field yet: the affine alone
    np.testing.assert_allclose(video[-1], registered, atol=1)  # the solve's final field
    fixed = _read(FIXED)[INNER]
    assert 1 - _corr(fixed, video[1][INNER]) < 1 - _corr(fixed, video[0][INNER])


def _zarr(path, data, axes="zyx"):
    arr = ts.open(
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
        create=True,
        dtype=ts.uint8,
        shape=data.shape,
    ).result()
    arr.write(data).result()
    n = len(axes) - 3
    (path / ".zattrs").write_text(
        f'{{"axis_names": {list(axes)}, "resolution": {[1] * n + [64] * 3},'
        f' "units": {[""] * n + ["nm"] * 3}}}'.replace("'", '"')
    )
    return str(path)


def test_what_the_moving_image_does_not_cover_is_left_alone(tmp_path):
    # The moving image is the fixed one cut off at x = 40: the answer is no deformation at
    # all. Fitting what lies beyond the cut would drag the moving image's edge over it.
    pytest.importorskip("torch")
    cropped = _zarr(tmp_path / "cropped.zarr", _read(FIXED)[:, :, :40])
    size = np.linalg.norm(_read(_url("show=field", moving=cropped)), axis=0)
    # voxels are 64 nm; fitting the uncovered part too gives 110-190 nm and 11-24 nm
    assert size[:, :, 40:].mean() < 45 and size[:, :, :37].mean() < 7


def test_reopening_reuses_the_solve(monkeypatch):
    pytest.importorskip("torch")
    calls = []
    real = register.solve

    def counting(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(register, "solve", counting)
    monkeypatch.setattr(register, "_solved", type(register._solved)())
    for extra in ("", "", "show=pair", "chunk=8,32,32", "smooth=2"):
        open_source(_url(extra))
    assert len(calls) == 2  # once, then again for the new smoothness


def test_refine_fits_finer_levels_where_they_are_read(monkeypatch):
    pytest.importorskip("torch")
    # a taller volume than FIXED, so a block is big enough for a coarse copy of itself
    fixed_url = FIXED.replace("shape=24,64,64&chunk=24,32,32", "shape=32,64,64&chunk=32,32,32")
    moving_url = f"warp://{fixed_url}?{SWIRL}"
    calls = []
    real = register.solve

    def counting(fixed, *args, **kw):
        calls.append((kw.get("label", ""), len(fixed)))
        return real(fixed, *args, **kw)

    def blocks():
        return [c for c in calls if "block" in c[0]]

    def url(extra):
        return _url(extra, moving=moving_url, fixed=fixed_url)

    monkeypatch.setattr(register, "solve", counting)
    monkeypatch.setattr(register, "_blocks", type(register._blocks)(register.BLOCK_CACHE_BYTES))
    fixed = _read(fixed_url)[INNER]
    coarse = _corr(fixed, _read(url("levels=1"))[INNER])  # level 0 served through level 1's field
    whole = _corr(fixed, _read(url("levels=1,0"))[INNER])  # both levels solved up front
    calls.clear()
    s0 = open_source(url("levels=1&refine=1&halo=4&block=32,32,32")).levels[0]
    assert not blocks()  # opening fits no block
    s0.read(s0.info.chunk_box((0, 0, 0)))
    touched = blocks()
    assert 0 < len(touched) <= 8, touched  # its blocks, and the neighbours it interpolates into
    assert 2 in {
        n for _, n in touched
    }  # a coarse copy of the block, then the block (edges: too thin)
    s0.read(s0.info.chunk_box((0, 0, 0)))
    assert len(blocks()) == len(touched)  # remembered
    refined = _corr(fixed, _read(s0)[INNER])
    assert len(blocks()) > len(touched)  # the rest, once read
    assert coarse < refined and abs(refined - whole) < 2e-3, (coarse, refined, whole)
    u = _read(url("levels=1&refine=1&halo=4&block=32,32,32&show=field"))
    assert u.shape == (3, 32, 64, 64) and np.isfinite(u).all()


def test_refined_blocks_live_in_the_pipelines_cache(monkeypatch):
    # a served register:// pipeline keeps its blocks with its chunks: --cache-gb bounds them,
    # clearing the cache clears them, and the process-wide fallback stays empty
    from chunkmirage.cache import LRUCache
    from chunkmirage.pipeline import Pipeline

    real = register.solve

    def constant(fixed, moving, affine, settings, *, box, label="", **kw):
        if "block" not in label:
            return real(fixed, moving, affine, settings, box=box, label=label, **kw)
        from chunkmirage.registration import Grid, _control_shape

        lo, hi = (np.asarray(b, dtype=float) for b in box)
        shape = _control_shape(lo, hi, fixed[-1].voxel_size, settings.grid)
        return [Grid(np.zeros((*shape, 3), np.float32), lo, (hi - lo) / (np.asarray(shape) - 1))]

    monkeypatch.setattr(register, "solve", constant)
    fallback = type(register._blocks)(register.BLOCK_CACHE_BYTES)
    monkeypatch.setattr(register, "_blocks", fallback)
    cache = LRUCache()
    spec = {"source": _url("levels=1&refine=1&iterations=0,5&halo=8&block=32,32,32")}
    p = Pipeline.from_spec(spec, cache=cache)
    p.chunk(0, (0, 0, 0))
    blocks = [k for k in cache._data if isinstance(k[0], str) and k[0].startswith("register:")]
    assert blocks and len(fallback) == 0
    cache.invalidate()
    assert len(cache) == 0


def test_neighbouring_blocks_blend_across_their_overlap(monkeypatch):
    # every block fits a constant field, 64 nm (a voxel) times its x index: the field steps by
    # a whole block's difference from one block to the next, and blending spreads each step
    # over the overlap instead of one lattice cell
    from chunkmirage.registration import Grid, _control_shape

    real = register.solve

    def constant(fixed, moving, affine, settings, *, box, label="", **kw):
        if "block" not in label:
            return real(fixed, moving, affine, settings, box=box, label=label, **kw)
        bx = int(label.rsplit(",", 1)[1].strip(" )"))
        lo, hi = (np.asarray(b, dtype=float) for b in box)
        shape = _control_shape(lo, hi, fixed[-1].voxel_size, settings.grid)
        values = np.zeros((*shape, 3), np.float32)
        values[..., 2] = 64.0 * bx
        return [Grid(values, lo, (hi - lo) / (np.asarray(shape) - 1))]

    monkeypatch.setattr(register, "solve", constant)
    monkeypatch.setattr(register, "_blocks", type(register._blocks)(register.BLOCK_CACHE_BYTES))
    url = _url("levels=1&refine=1&iterations=0,5&halo=8&block=32,32,32&show=field")
    ux = _read(url)[2, 12, 12]  # x component along x, inside a block in z and y
    # blocks are 8 lattice points (32 voxels) wide; 3 to 5 points into one, only it counts
    for bx in range(2):
        np.testing.assert_allclose(ux[32 * bx + 14 : 32 * bx + 20], 64.0 * bx, atol=1e-3)
    steps = np.abs(np.diff(ux))
    assert steps.max() <= 0.3 * 64 / 4, (
        steps.max()
    )  # a quarter of the step per lattice cell at most


def test_halving_does_not_wrap_small_integers():
    from chunkmirage.registration import Level, _halve

    lvl = Level(np.full((4, 4, 4), 200, np.uint8), np.ones(3), np.zeros(3))
    assert (_halve(lvl).data == 200).all()  # not (200 + 200) % 256 / 2


def test_the_affine_is_found_when_asked(tmp_path):
    pytest.importorskip("torch")
    # the moving image is the fixed one turned half a turn about y and shifted 3 voxels in y
    fixed = _read(FIXED)
    path = _zarr(tmp_path / "turned.zarr", np.ascontiguousarray(fixed[::-1, :, ::-1]))
    shift = "1,0,0,0,0,1,0,192,0,0,1,0"
    turned = _read(_url(f"affine={shift}&iterations=0", moving=path))
    still = _zarr(tmp_path / "moving.zarr", turned)
    before = _corr(fixed[INNER], _read(_url("iterations=0", moving=still))[INNER])
    found = _read(_url("affine=auto&iterations=0", moving=still))
    after = _corr(fixed[INNER][:, 3:], found[INNER][:, 3:])
    assert before < 0.3 and after > 0.95, (before, after)


def test_pair_and_field_views_say_how_to_show_themselves():
    from chunkmirage.neuroglancer import layer_for
    from chunkmirage.pipeline import Pipeline

    pair = layer_for("pair", Pipeline(open_source(_url("show=pair&iterations=0")), []), "zarr3://u")
    assert "emitRGB(vec3(f, m, f))" in pair["shader"]  # fixed magenta, moving green
    assert list(pair["source"]["transform"]["outputDimensions"]) == ["c^", "z", "y", "x"]
    field = layer_for(
        "field", Pipeline(open_source(_url("show=field&iterations=0")), []), "zarr3://u"
    )
    assert "colormapJet" in field["shader"]
    plain = layer_for("image", Pipeline(open_source(_url("iterations=0")), []), "zarr3://u")
    assert plain == {"type": "image", "source": "zarr3://u", "name": "image"}


def test_a_mirror_image_needs_its_affine_given(tmp_path):
    pytest.importorskip("torch")
    # the moving image is the fixed one with z reversed, as a stack acquired the other way
    # round: no rotation maps one onto the other, so the search (rotations only) cannot find
    # it, and the mirror given as an affine registers it exactly
    fixed = _read(FIXED)
    mirrored = _zarr(tmp_path / "mirrored.zarr", np.ascontiguousarray(fixed[::-1]))
    found = _corr(fixed[INNER], _read(_url("affine=auto&iterations=0", moving=mirrored))[INNER])
    z = (fixed.shape[0] - 1) * 64.0  # z' = z_max - z, in physical units (64 per voxel)
    given = f"affine=-1,0,0,{z},0,1,0,0,0,0,1,0&iterations=0"
    mirror = _corr(fixed[INNER], _read(_url(given, moving=mirrored))[INNER])
    assert mirror > 0.999 and found < 0.97, (found, mirror)


def test_channels_are_matched_and_all_registered(tmp_path):
    fixed = _read(FIXED)
    data = np.stack([255 - fixed, fixed])[None]  # t, c: match on channel 1
    path = _zarr(tmp_path / "moving.zarr", data, axes="tczyx")
    out = open_source(_url("moving_channel=1&iterations=0", moving=path)).levels[0]
    assert out.info.axes == ("t", "c", "z", "y", "x") and out.info.shape == (1, 2, 24, 64, 64)
    np.testing.assert_array_equal(_read(out), data)
    with pytest.raises(ValueError, match="outside its 2"):
        open_source(_url("moving_channel=2&iterations=0", moving=path))
    with pytest.raises(ValueError, match="no c axis"):
        open_source(_url("fixed_channel=1&iterations=0", moving=path))


@pytest.mark.parametrize(
    "extra, message",
    [
        ("bogus=1", "unknown register:// parameters"),
        ("show=both", r"show\s+Input should be 'image', 'pair' or 'field'"),
        ("window=4", "positive odd"),
        ("levels=0,5", "levels 0..1"),
        ("iterations=1,2,3", "one per level"),
        ("affine=1,2,3", "4x4 or 3x4"),
        ("frames=0", r"frames\s+Input should be greater than or equal to 1"),
        ("levels=1&refine=2", "1 level\\(s\\) lie below level 1"),
        ("levels=1&refine=1&frames=2", "frames and refine"),
        ("levels=1&refine=1&iterations=1,2,3", "1 or 2 values"),
    ],
)
def test_bad_urls_explain_themselves(extra, message):
    with pytest.raises(ValueError, match=message):
        open_source(_url(extra + "&iterations=0" if "iterations" not in extra else extra))


def test_a_fixed_image_is_required():
    with pytest.raises(ValueError, match="fixed="):
        open_source(f"register://{FIXED}")


def test_default_levels_go_from_coarse_enough_to_fine_enough():
    def fake(*shapes):
        from types import SimpleNamespace

        levels = [SimpleNamespace(info=SimpleNamespace(shape=(1, *s))) for s in shapes]
        return SimpleNamespace(levels=levels)

    gut = fake((1231, 14124, 5452), (615, 7062, 2726), (307, 3531, 1363), (153, 1765, 681),
               (76, 882, 340), (38, 441, 170), (19, 220, 85))  # fmt: skip
    assert register._default_levels(gut, 1) == [6, 5, 4]
    assert register._default_levels(fake((8, 8, 8)), 1) == [0]  # small: the one level
    assert register._default_levels(fake((64, 64, 64), (32, 32, 32), (16, 16, 16)), 1) == [2, 1, 0]


def test_flat_windows_carry_no_signal():
    torch = pytest.importorskip("torch")
    from chunkmirage.registration import _lncc, _window_stats

    rng = np.random.default_rng(0)
    img = torch.as_tensor(rng.random((1, 1, 12, 12, 12)), dtype=torch.float32)
    stats = _window_stats(img, 5)
    np.testing.assert_allclose(_lncc(img, stats, img, 5).numpy(), 1, atol=1e-4)  # itself: perfect
    flat = torch.full_like(img, 0.5)
    assert not _lncc(img, stats, flat, 5).any()  # no contrast: nothing to gain from a warp
    assert (_lncc(img, stats, 2 * img + 1, 5) > 0.999).all()  # intensity scale does not matter
