"""scene:// sources: images resampled through OME-Zarr 0.6 coordinate transformations.

The moving images are a smooth analytic function sampled on their own grid, so the value
a correct implementation must produce at every fixed-grid voxel is known in closed form.
"""

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.core import Box
from chunkmirage.ngff import (
    FieldOptions,
    LargeFieldChunks,
    Scene,
    UnsupportedTransform,
    parse_transform,
)
from chunkmirage.transforms import (
    Affine,
    Displacements,
    InverseDisplacements,
    Sequence,
    VectorField,
    simplify,
)

CONFORMANCE = Path(__file__).parent / "data" / "rfc5_conformance" / "cases"

AXES = {
    "z": {"name": "z", "type": "space", "unit": "micrometer"},
    "y": {"name": "y", "type": "space", "unit": "micrometer"},
    "x": {"name": "x", "type": "space", "unit": "micrometer"},
    "c": {"name": "c", "type": "channel"},
    "t": {"name": "t", "type": "time", "unit": "millisecond"},
}
K = 2 * np.pi / 40.0


def g(p):
    """Smooth test signal of physical position (..., 3)."""
    return 100 + 15 * (
        np.sin(K * p[..., 0] + 0.3) + np.sin(K * p[..., 1] + 1.1) + np.sin(K * p[..., 2] + 2.0)
    )


def centres(shape, spacing, origin=0.0):
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    return idx * np.asarray(spacing, dtype=float) + np.asarray(origin, dtype=float)


def _array(path: Path, data: np.ndarray, chunks=None):
    chunks = chunks or [min(s, 16) for s in data.shape]
    t = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(data.shape),
                "data_type": data.dtype.name,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
            },
        },
        create=True,
    ).result()
    t.write(data).result()


def _group(path: Path, ome: dict):
    path.mkdir(parents=True, exist_ok=True)
    body = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": {"version": "0.6", **ome}},
    }
    (path / "zarr.json").write_text(json.dumps(body))


def _image(root: Path, path: str, levels, scales, translations=None, names="zyx", **extra):
    datasets = []
    for i, (data, s) in enumerate(zip(levels, scales)):
        _array(root / path / f"s{i}", data)
        ct = {"type": "scale", "scale": list(map(float, s))}
        if translations is not None:
            tr = {"type": "translation", "translation": list(map(float, translations[i]))}
            ct = {"type": "sequence", "transformations": [ct, tr]}
        ct |= {"input": {"path": f"s{i}"}, "output": {"name": "physical"}}
        datasets.append({"path": f"s{i}", "coordinateTransformations": [ct]})
    systems = [{"name": "physical", "axes": [AXES[n] for n in names]}]
    ms = {"coordinateSystems": systems + extra.pop("systems", []), "datasets": datasets}
    if "transforms" in extra:
        ms["coordinateTransformations"] = extra.pop("transforms")
    _group(root / path, {"multiscales": [ms], **extra})


def _field(root: Path, path: str, vectors: np.ndarray, spacing, origin=None):
    """A 0.6 displacement field group; ``vectors`` has the vector axis first."""
    n = vectors.ndim - 1
    _array(root / path / "s0", vectors.astype(np.float32))
    ct = {
        "type": "sequence",
        "input": {"path": "s0"},
        "output": {"name": "physical"},
        "transformations": [
            {"type": "scale", "scale": [1.0, *map(float, spacing)]},
            {"type": "translation", "translation": [0.0, *map(float, origin or [0] * n)]},
        ],
    }
    axes = [{"name": "d", "type": "displacement", "discrete": True}] + [AXES[a] for a in "zyx"[-n:]]
    _group(
        root / path,
        {
            "multiscales": [
                {
                    "coordinateSystems": [{"name": "physical", "axes": axes}],
                    "datasets": [{"path": "s0", "coordinateTransformations": [ct]}],
                }
            ]
        },
    )


def _scene(root: Path, transforms, systems=()):
    _group(
        root,
        {"scene": {"coordinateTransformations": transforms, "coordinateSystems": list(systems)}},
    )


def ref(path, name="physical"):
    return {"path": path, "name": name}


# A fixed grid, and a moving image whose physical space covers it after the transform.
FIXED_SHAPE, FIXED_SPACING, FIXED_ORIGIN = (24, 26, 28), 2.0, 5.0
MOVING_SHAPE, MOVING_SPACING = (44, 46, 48), 1.5
ANG = np.deg2rad(5)
A = 0.95 * np.array([[1, 0, 0], [0, np.cos(ANG), -np.sin(ANG)], [0, np.sin(ANG), np.cos(ANG)]])
T = np.array([3.0, -2.0, 4.0])


def _fixed_and_moving(root: Path):
    fixed_pts = centres(FIXED_SHAPE, FIXED_SPACING, FIXED_ORIGIN)
    _image(
        root,
        "fixed",
        [np.zeros(FIXED_SHAPE, np.uint8)],
        [[FIXED_SPACING] * 3],
        [[FIXED_ORIGIN] * 3],
    )
    moving = g(centres(MOVING_SHAPE, MOVING_SPACING)).astype(np.float32)
    _image(root, "moving", [moving], [[MOVING_SPACING] * 3])
    return fixed_pts


def _inside_moving(p, margin=1.0):
    hi = (np.asarray(MOVING_SHAPE) - 1) * MOVING_SPACING
    return np.all((p >= margin) & (p <= hi - margin), axis=-1)


def _read_all(src):
    return src.read(Box((0,) * src.info.ndim, src.info.shape))


def test_affine_registration_matches_analytic(tmp_path):
    fixed_pts = _fixed_and_moving(tmp_path)
    affine = np.hstack([A, T[:, None]]).tolist()
    _scene(
        tmp_path,
        [{"type": "affine", "affine": affine, "input": ref("fixed"), "output": ref("moving")}],
    )
    src = open_source(f"scene://{tmp_path}?image=moving&target=fixed")
    lvl = src.levels[0]
    assert lvl.info.shape == FIXED_SHAPE
    assert lvl.info.voxel_size == (FIXED_SPACING,) * 3
    assert lvl.info.translation == (FIXED_ORIGIN,) * 3
    assert lvl.info.axes == ("z", "y", "x") and lvl.info.units == ("micrometer",) * 3
    out = _read_all(lvl)
    mapped = fixed_pts @ A.T + T
    inside = _inside_moving(mapped)
    assert inside.mean() > 0.5
    np.testing.assert_allclose(out[inside], g(mapped)[inside], atol=0.5)
    far = ~_inside_moving(mapped, margin=-2 * MOVING_SPACING)
    assert np.all(out[far] == 0)
    rechunked = open_source(f"scene://{tmp_path}?image=moving&target=fixed&chunk=8,8,8")
    assert rechunked.levels[0].info.chunk_shape == (8, 8, 8)
    np.testing.assert_array_equal(_read_all(rechunked.levels[0]), out)


def test_closed_form_inverse_is_used_for_the_reverse_direction(tmp_path):
    fixed_pts = _fixed_and_moving(tmp_path)
    inv = Affine.from_linear(A, T).inverse()
    _scene(
        tmp_path,
        [
            {
                "type": "affine",
                "affine": inv.matrix.tolist(),
                "input": ref("moving"),
                "output": ref("fixed"),
            }
        ],
    )
    out = _read_all(open_source(f"scene://{tmp_path}?image=moving&target=fixed").levels[0])
    mapped = fixed_pts @ A.T + T
    inside = _inside_moving(mapped)
    np.testing.assert_allclose(out[inside], g(mapped)[inside], atol=0.5)


def d_analytic(p):
    return 2.0 * np.stack(
        [np.sin(p[..., 1] / 12), np.sin(p[..., 2] / 14), np.sin(p[..., 0] / 10)], -1
    )


def test_displacement_field_then_affine_matches_analytic(tmp_path):
    """bigstream/BigWarp shape: fixed -> moving = affine(x + d(x)), d sampled coarsely."""
    fixed_pts = _fixed_and_moving(tmp_path)
    fshape, fspacing = (17, 17, 17), 4.0
    _field(
        tmp_path,
        "coordinateTransformations/dfield",
        np.moveaxis(d_analytic(centres(fshape, fspacing)), -1, 0),
        [fspacing] * 3,
    )
    chain = {
        "type": "sequence",
        "input": ref("fixed"),
        "output": ref("moving"),
        "transformations": [
            {
                "type": "displacements",
                "path": "coordinateTransformations/dfield",
                "interpolation": "linear",
            },
            {"type": "affine", "affine": np.hstack([A, T[:, None]]).tolist()},
        ],
    }
    _scene(tmp_path, [chain])
    out = _read_all(open_source(f"scene://{tmp_path}?image=moving&target=fixed").levels[0])
    mapped = (fixed_pts + d_analytic(fixed_pts)) @ A.T + T
    inside = _inside_moving(mapped)
    assert inside.mean() > 0.5
    np.testing.assert_allclose(out[inside], g(mapped)[inside], atol=1.0)
    tiny_cache = open_source(f"scene://{tmp_path}?image=moving&target=fixed&field_cache_gb=0.001")
    np.testing.assert_array_equal(_read_all(tiny_cache.levels[0]), out)


def test_displacements_only_invert_approximately_when_asked(tmp_path):
    """The field maps moving -> fixed; rendering moving onto fixed needs its inverse."""
    fixed_pts = _fixed_and_moving(tmp_path)
    fshape, fspacing = (18, 18, 18), 4.0
    _field(
        tmp_path,
        "coordinateTransformations/m2f",
        np.moveaxis(d_analytic(centres(fshape, fspacing)), -1, 0),
        [fspacing] * 3,
    )
    _scene(
        tmp_path,
        [
            {
                "type": "displacements",
                "path": "coordinateTransformations/m2f",
                "input": ref("moving"),
                "output": ref("fixed"),
            }
        ],
    )
    with pytest.raises(ValueError, match="inverse=approx"):
        open_source(f"scene://{tmp_path}?image=moving&target=fixed")
    out = _read_all(
        open_source(f"scene://{tmp_path}?image=moving&target=fixed&inverse=approx").levels[0]
    )
    x = fixed_pts.copy()  # solve x + d(x) = y analytically
    for _ in range(100):
        x = fixed_pts - d_analytic(x)
    inside = _inside_moving(x)
    np.testing.assert_allclose(out[inside], g(x)[inside], atol=1.0)


def test_channels_pass_through_by_dimension(tmp_path):
    rng = np.random.default_rng(1)
    moving = rng.integers(0, 255, size=(2, 30, 32), dtype=np.uint8)
    _image(tmp_path, "moving", [moving], [[1, 1, 1]], names="cyx")
    _image(tmp_path, "fixed", [np.zeros((3, 20, 22), np.uint8)], [[1, 1, 1]], names="cyx")
    by = {
        "type": "byDimension",
        "input": ref("fixed"),
        "output": ref("moving"),
        "transformations": [
            {
                "transformation": {"type": "scale", "scale": [1.0]},
                "inputAxes": [0],
                "outputAxes": [0],
            },
            {
                "transformation": {"type": "translation", "translation": [4.0, 6.0]},
                "inputAxes": [1, 2],
                "outputAxes": [1, 2],
            },
        ],
    }
    _scene(tmp_path, [by])
    lvl = open_source(f"scene://{tmp_path}?image=moving&target=fixed").levels[0]
    assert lvl.info.shape == (2, 20, 22)  # the moving image's channels, the fixed grid
    assert lvl.info.axes == ("c", "y", "x")
    np.testing.assert_array_equal(_read_all(lvl), moving[:, 4:24, 6:28])
    part = lvl.read(Box((1, 3, 5), (2, 9, 20)))
    np.testing.assert_array_equal(part, moving[1:2, 7:13, 11:26])


def test_labels_use_nearest_and_keep_ids(tmp_path):
    rng = np.random.default_rng(2)
    labels = rng.choice(np.array([0, 7, 1_000_003, 4_000_000_001], np.uint32), size=MOVING_SHAPE)
    _image(tmp_path, "labels", [labels], [[MOVING_SPACING] * 3], **{"image-label": {}})
    _image(
        tmp_path,
        "fixed",
        [np.zeros(FIXED_SHAPE, np.uint8)],
        [[FIXED_SPACING] * 3],
        [[FIXED_ORIGIN] * 3],
    )
    _scene(
        tmp_path,
        [
            {
                "type": "affine",
                "affine": np.hstack([A, T[:, None]]).tolist(),
                "input": ref("fixed"),
                "output": ref("labels"),
            }
        ],
    )
    out = _read_all(open_source(f"scene://{tmp_path}?image=labels&target=fixed").levels[0])
    assert out.dtype == np.uint32
    assert set(np.unique(out)) <= {0, 7, 1_000_003, 4_000_000_001}
    # spot-check against the nearest source voxel
    p = centres(FIXED_SHAPE, FIXED_SPACING, FIXED_ORIGIN) @ A.T + T
    idx = np.floor(p / MOVING_SPACING + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < MOVING_SHAPE), axis=-1)
    np.testing.assert_array_equal(out[ok], labels[tuple(idx[ok].T)])


def test_each_level_reads_the_matching_source_level(tmp_path):
    moving = g(centres(MOVING_SHAPE, MOVING_SPACING)).astype(np.float32)
    m1 = moving[::2, ::2, ::2]
    _image(
        tmp_path,
        "moving",
        [moving, m1],
        [[MOVING_SPACING] * 3, [2 * MOVING_SPACING] * 3],
        [[0] * 3, [MOVING_SPACING / 2] * 3],
    )
    f0 = np.zeros(FIXED_SHAPE, np.uint8)
    _image(
        tmp_path,
        "fixed",
        [f0, f0[::2, ::2, ::2]],
        [[FIXED_SPACING] * 3, [2 * FIXED_SPACING] * 3],
        [[FIXED_ORIGIN] * 3, [FIXED_ORIGIN + FIXED_SPACING / 2] * 3],
    )
    _scene(tmp_path, [{"type": "identity", "input": ref("fixed"), "output": ref("moving")}])
    src = open_source(f"scene://{tmp_path}?image=moving&target=fixed")
    assert len(src.levels) == 2
    assert src.levels[0].src.info.shape == MOVING_SHAPE
    assert src.levels[1].src.info.shape == m1.shape
    assert src.levels[1].info.shape == f0[::2, ::2, ::2].shape
    assert src.levels[1].cache_key() != src.levels[0].cache_key()


def test_explicit_grid_in_a_scene_coordinate_system(tmp_path):
    """Spec stitching example: tiles placed in 'world' by translations (walked inverted)."""
    tile = np.arange(10 * 12, dtype=np.uint16).reshape(10, 12) + 1
    _image(tmp_path, "tile_1", [tile], [[1, 1]], names="yx")
    world = {"name": "world", "axes": [AXES["y"], AXES["x"]]}
    _scene(
        tmp_path,
        [
            {
                "type": "translation",
                "translation": [0, 12],
                "input": ref("tile_1"),
                "output": {"name": "world"},
            }
        ],
        [world],
    )
    lvl = open_source(f"scene://{tmp_path}?image=tile_1&shape=10,30&levels=1").levels[0]
    out = _read_all(lvl)
    assert out.shape == (10, 30)
    np.testing.assert_array_equal(out[:, 12:24], tile)
    assert not out[:, :12].any() and not out[:, 24:].any()


def test_served_through_the_pipeline_and_zarr3_frontend(tmp_path):
    _fixed_and_moving(tmp_path)
    _scene(
        tmp_path,
        [
            {
                "type": "affine",
                "affine": np.hstack([A, T[:, None]]).tolist(),
                "input": ref("fixed"),
                "output": ref("moving"),
            }
        ],
    )
    url = f"scene://{tmp_path}?image=moving&target=fixed"
    pipe = Pipeline.from_spec({"source": url, "ops": [{"op": "cast", "dtype": "uint8"}]})
    client = TestClient(create_app({"warped": pipe}))
    meta = client.get("/warped/zarr3/s0/zarr.json").json()
    assert meta["shape"] == list(FIXED_SHAPE)
    assert client.get("/warped/zarr3/s0/c/0/0/0").status_code == 200
    direct = _read_all(open_source(url).levels[0]).astype(np.uint8)
    np.testing.assert_array_equal(pipe.read(0, Box((0, 0, 0), FIXED_SHAPE)), direct)


def test_plain_open_of_a_v06_image(tmp_path):
    """0.6 metadata without scene://: sequence translation read, other systems not applied."""
    data = np.zeros((4, 5, 6), np.uint8)
    other = {"name": "elsewhere", "axes": [AXES[a] for a in "zyx"]}
    shift = [
        {
            "type": "translation",
            "translation": [100, 100, 100],
            "input": {"name": "physical"},
            "output": {"name": "elsewhere"},
        }
    ]
    _image(tmp_path, "img", [data], [[2, 3, 4]], [[1, 1, 1]], systems=[other], transforms=shift)
    info = open_source(str(tmp_path / "img")).levels[0].info
    assert info.voxel_size == (2.0, 3.0, 4.0)
    assert info.translation == (1.0, 1.0, 1.0)
    assert info.units == ("micrometer",) * 3 and info.axes == ("z", "y", "x")
    moved = open_source(f"scene://{tmp_path / 'img'}?target=elsewhere").levels[0].info
    assert moved.translation == (1.0, 1.0, 1.0)  # same grid, now in "elsewhere"


def test_bad_urls_explain_themselves(tmp_path):
    _fixed_and_moving(tmp_path)
    _scene(tmp_path, [{"type": "identity", "input": ref("fixed"), "output": ref("moving")}])
    with pytest.raises(ValueError, match="unknown scene:// parameters"):
        open_source(f"scene://{tmp_path}?image=moving&bogus=1")
    with pytest.raises(ValueError, match="choose an image"):
        open_source(f"scene://{tmp_path}")
    with pytest.raises(UnsupportedTransform):
        parse_transform({"type": "warp9000"}, str(tmp_path))


def _warp_scene(tmp_path):
    """fixed -> moving through a displacement field, as in the bigstream layout."""
    _fixed_and_moving(tmp_path)
    vectors = np.moveaxis(d_analytic(centres((17, 17, 17), 4.0)), -1, 0)
    _field(tmp_path, "coordinateTransformations/dfield", vectors, [4.0] * 3)
    chain = {
        "type": "sequence",
        "input": ref("fixed"),
        "output": ref("moving"),
        "transformations": [
            {"type": "displacements", "path": "coordinateTransformations/dfield"},
            {"type": "affine", "affine": np.hstack([A, T[:, None]]).tolist()},
        ],
    }
    _scene(tmp_path, [chain])


def test_fields_in_huge_chunks_are_refused_unless_allowed(tmp_path):
    """A field stored in huge chunks costs one whole-chunk decode per output chunk; under a
    viewer's parallel requests that exhausted a 93 GB workstation. Refuse, and say why."""
    _warp_scene(tmp_path)
    small = FieldOptions(max_chunk_bytes=1000)  # the test field's chunks are ~49 kB
    with pytest.raises(LargeFieldChunks, match="large_field_chunks=1"):
        Scene(str(tmp_path), fields=small)
    allowed = FieldOptions(max_chunk_bytes=1000, allow_large_chunks=True)
    t = Scene(str(tmp_path), fields=allowed).transform(
        ("fixed", "physical"), ("moving", "physical")
    )
    t_default = Scene(str(tmp_path)).transform(("fixed", "physical"), ("moving", "physical"))
    pts = centres(FIXED_SHAPE, FIXED_SPACING, FIXED_ORIGIN).reshape(-1, 3)
    np.testing.assert_array_equal(t.apply(pts), t_default.apply(pts))
    url = f"scene://{tmp_path}?image=moving&target=fixed"
    np.testing.assert_array_equal(
        _read_all(open_source(url + "&large_field_chunks=1").levels[0]),
        _read_all(open_source(url).levels[0]),
    )


def test_a_chunk_that_would_read_too_much_is_refused(tmp_path, monkeypatch):
    import chunkmirage.sources.scene as scene_mod

    _warp_scene(tmp_path)
    lvl = open_source(f"scene://{tmp_path}?image=moving&target=fixed").levels[0]
    monkeypatch.setattr(scene_mod, "MAX_WINDOW_BYTES", 1000)
    with pytest.raises(MemoryError, match="smaller output chunks"):
        lvl.read(Box((0, 0, 0), (16, 16, 16)))


def test_the_output_grid_is_part_of_the_cache_key(tmp_path):
    """Grids that differ only in shape must not share cached chunks or URLs."""
    _image(tmp_path, "tile_1", [np.ones((10, 12), np.uint16)], [[1, 1]], names="yx")
    world = {"name": "world", "axes": [AXES["y"], AXES["x"]]}
    t = {"type": "translation", "translation": [0, 12], "input": ref("tile_1"), "output": world}
    _scene(tmp_path, [t | {"output": {"name": "world"}}], [world])
    url = f"scene://{tmp_path}?image=tile_1&levels=1"
    a, b = (Pipeline.from_spec({"source": f"{url}&shape={s}"}) for s in ("10,30", "10,40"))
    assert a.digest() != b.digest()
    # One value applies to every spatial axis; a wrong count is an error, not an IndexError.
    one = open_source(f"{url}&shape=10,30&voxel_size=2&chunk=8").levels[0].info
    assert one.voxel_size == (2.0, 2.0) and one.chunk_shape == (8, 8)
    with pytest.raises(ValueError, match="voxel_size needs 1 or 2 values"):
        open_source(f"{url}&shape=10,30&voxel_size=1,2,3")


def test_time_and_channel_keep_their_own_scale(tmp_path):
    data = np.zeros((3, 2, 8, 9), np.uint8)
    _image(tmp_path, "img", [data], [[2, 1, 1, 1]], names="tcyx")
    info = open_source(f"scene://{tmp_path}/img").levels[0].info
    assert info.axes == ("t", "c", "y", "x") and info.voxel_size[:2] == (2.0, 1.0)
    assert info.units[0] == "millisecond"


def test_references_ignore_slashes_and_sequences_compose_in_order():
    from chunkmirage.ngff import _ref
    from chunkmirage.sources.tensorstore_source import _ome_transforms

    assert _ref({"path": "moving/", "name": "physical"}) == ("moving", "physical")
    assert _ref({"path": "/", "name": "world"}) == (None, "world")
    seq = {
        "type": "sequence",
        "transformations": [
            {"type": "translation", "translation": [1, 2]},
            {"type": "scale", "scale": [2, 3]},
        ],
    }
    assert _ome_transforms([seq], 2) == ((2.0, 3.0), (2.0, 6.0))  # 2*(i+1), 3*(i+2)


def test_the_inverse_judges_the_points_it_returns():
    """Out of iterations, the last step's result is what is checked, not the one before."""
    halves = _field_1d(lambda x: 0.5 * x)  # x + x/2 = y: fixed-point error halves each step
    x = InverseDisplacements(halves, max_iter=1, tol=0.02).apply(np.array([[1.0]]))
    np.testing.assert_allclose(x, [[0.75]])  # error 0.125 < 10 tol, so kept


def _field_1d(values_fn, n=200):
    xs = np.arange(n, dtype=float)
    arr = values_fn(xs)[None, :]
    return VectorField(lambda sl: arr[sl], arr.shape, 0, Affine.identity(1), key="t")


def test_inverse_displacements_round_trip_and_flag_folds():
    smooth = _field_1d(lambda x: 0.5 * np.sin(x / 5))  # |d'| = 0.1 < 1: invertible
    y = np.linspace(20, 180, 50)[:, None]
    x = InverseDisplacements(smooth).apply(y)
    np.testing.assert_allclose(Displacements(smooth).apply(x), y, atol=1e-2)
    folding = _field_1d(lambda x: 20 * np.sin(x / 5))  # |d'| = 4: folds, no unique inverse
    assert np.isnan(InverseDisplacements(folding).apply(y)).any()


def test_simplify_folds_linear_runs():
    t = Sequence(
        [
            Affine.scale_translation([2, 2]),
            Affine.from_linear(np.eye(2), [1, -1]),
            Affine.identity(2),
        ]
    )
    s = simplify(t)
    assert isinstance(s, Affine)
    p = np.array([[1.0, 2.0], [3.0, -4.0]])
    np.testing.assert_allclose(s.apply(p), t.apply(p))


@pytest.mark.parametrize("case", sorted(p.name for p in CONFORMANCE.iterdir()))
def test_rfc5_conformance(case):
    conf = tomllib.loads((CONFORMANCE / case / "conformance.toml").read_text())
    src, dst = (None, conf["source"]["name"]), (None, conf["target"]["name"])
    if conf.get("should_error"):
        with pytest.raises((KeyError, ValueError)):
            Scene(str(CONFORMANCE / case)).transform(src, dst)
        return
    out = (
        Scene(str(CONFORMANCE / case))
        .transform(src, dst)
        .apply(np.asarray(conf["source"]["coordinates"], float))
    )
    np.testing.assert_allclose(
        out,
        conf["target"]["coordinates"],
        atol=conf.get("absolute_tolerance", 1e-6),
        rtol=conf.get("relative_tolerance", 1e-3),
    )
