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
