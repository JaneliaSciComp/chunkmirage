"""Reading voxel size / translation / axis order from the metadata conventions found in the wild."""

import json

import numpy as np
import pytest
import tensorstore as ts

from chunkmirage import open_source
from chunkmirage.core import Box
from chunkmirage.sources.tensorstore_source import TensorStoreSource

DATA = np.arange(10 * 20 * 30, dtype=np.uint8).reshape(10, 20, 30)  # (z, y, x)


def _n5(path, attrs):
    """Write DATA as a real N5 would store it (dimensions x-first), then merge ``attrs``."""
    t = ts.open(
        {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "dimensions": list(DATA.shape[::-1]),
                "blockSize": [8, 8, 8],
                "dataType": "uint8",
                "compression": {"type": "raw"},
            },
        },
        create=True,
    ).result()
    t.write(DATA.T).result()
    meta = json.loads((path / "attributes.json").read_text())
    meta.update(attrs)
    (path / "attributes.json").write_text(json.dumps(meta))


def _zarr2(path, arr, attrs=None):
    t = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {"shape": list(arr.shape), "chunks": [5, 5, 5], "dtype": "|u1"},
        },
        create=True,
    ).result()
    t.write(arr).result()
    if attrs:
        (path / ".zattrs").write_text(json.dumps(attrs))


def _write_json(path, name, body):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(body))


def _zgroup(path, attrs):
    _write_json(path, ".zgroup", {"zarr_format": 2})
    _write_json(path, ".zattrs", attrs)


def _ome(paths, scales, translations=None):
    datasets = []
    for i, (p, s) in enumerate(zip(paths, scales)):
        ct = [{"type": "scale", "scale": s}]
        if translations:
            ct.append({"type": "translation", "translation": translations[i]})
        datasets.append({"path": p, "coordinateTransformations": ct})
    axes = [{"name": n, "type": "space", "unit": "nanometer"} for n in "zyx"]
    return {"multiscales": [{"version": "0.4", "axes": axes, "datasets": datasets}]}


def test_n5_is_transposed_to_c_order_with_resolution_and_offset(tmp_path):
    _n5(tmp_path / "s0", {"resolution": [6, 5, 4], "offset": [60, 50, 40]})  # x, y, z
    info = (s := TensorStoreSource.from_path(str(tmp_path / "s0"))).info
    assert info.shape == DATA.shape
    assert info.chunk_shape == (8, 8, 8)
    assert info.voxel_size == (4.0, 5.0, 6.0)
    assert info.translation == (40.0, 50.0, 60.0)
    np.testing.assert_array_equal(s.read(Box((1, 2, 3), (6, 17, 29))), DATA[1:6, 2:17, 3:29])


def test_n5_cosem_transform_is_c_order(tmp_path):
    # As on OpenOrganelle: pixelResolution x-first, transform z-first.
    _n5(
        tmp_path / "s0",
        {
            "pixelResolution": {"dimensions": [4.0, 4.0, 5.24], "unit": "nm"},
            "transform": {
                "axes": ["z", "y", "x"],
                "scale": [5.24, 4.0, 4.0],
                "translate": [1.0, 2.0, 3.0],
                "units": ["nm", "nm", "nm"],
            },
        },
    )
    info = TensorStoreSource.from_path(str(tmp_path / "s0")).info
    assert info.voxel_size == (5.24, 4.0, 4.0)
    assert info.translation == (1.0, 2.0, 3.0)
    assert info.axes == ("z", "y", "x")


def test_n5_group_metadata_and_downsampling_factors(tmp_path):
    root = tmp_path / "vol.n5"
    _write_json(
        root, "attributes.json", {"pixelResolution": {"dimensions": [4, 4, 8], "unit": "nm"}}
    )
    _n5(root / "s0", {})
    _n5(root / "s1", {"downsamplingFactors": [2, 2, 1]})
    ms = open_source(str(root))
    assert [lvl.info.voxel_size for lvl in ms] == [(8.0, 4.0, 4.0), (8.0, 8.0, 8.0)]


def test_n5_group_multiscales_transform_matched_by_path(tmp_path):
    root = tmp_path / "vol.n5"
    ds = [
        {
            "path": "s1",
            "transform": {
                "axes": list("zyx"),
                "scale": [16, 8, 8],
                "translate": [4, 2, 2],
                "units": ["nm"] * 3,
            },
        },
        {
            "path": "s0",
            "transform": {
                "axes": list("zyx"),
                "scale": [8, 4, 4],
                "translate": [0, 0, 0],
                "units": ["nm"] * 3,
            },
        },
    ]
    _write_json(root, "attributes.json", {"n5": "2.0.0", "multiscales": [{"datasets": ds}]})
    _n5(root / "s0", {})
    _n5(root / "s1", {})
    ms = open_source(str(root))
    # Level order follows the datasets list; each level gets its own entry's metadata.
    assert [lvl.info.voxel_size for lvl in ms] == [(16.0, 8.0, 8.0), (8.0, 4.0, 4.0)]
    assert ms[0].info.translation == (4.0, 2.0, 2.0)


def test_ome_levels_are_matched_by_path(tmp_path):
    root = tmp_path / "ome.zarr"
    _zgroup(root, _ome(["0", "1"], [[8, 8, 8], [16, 16, 16]], [[0, 0, 0], [4, 4, 4]]))
    _zarr2(root / "0", DATA)
    _zarr2(root / "1", np.ascontiguousarray(DATA[::2, ::2, ::2]))
    ms = open_source(str(root))
    assert [lvl.info.voxel_size for lvl in ms] == [(8.0,) * 3, (16.0,) * 3]
    assert ms[1].info.translation == (4.0, 4.0, 4.0)
    assert ms[0].info.units == ("nanometer",) * 3
    direct = TensorStoreSource.from_path(str(root / "1"))
    assert direct.info.voxel_size == (16.0, 16.0, 16.0)


def test_ome_unlisted_array_does_not_borrow_another_levels_scale(tmp_path):
    root = tmp_path / "ome.zarr"
    _zgroup(root, _ome(["s0"], [[8, 8, 8]]))
    _zarr2(root / "s0", DATA)
    _zarr2(root / "s1", DATA, {"resolution": [2, 3, 4]})
    assert TensorStoreSource.from_path(str(root / "s1")).info.voxel_size == (2.0, 3.0, 4.0)


def test_zarr_legacy_array_attrs(tmp_path):
    _zgroup(tmp_path / "legacy.zarr", {})
    _zarr2(
        tmp_path / "legacy.zarr" / "raw", DATA, {"resolution": [8, 4, 4], "offset": [80, 40, 40]}
    )
    info = TensorStoreSource.from_path(str(tmp_path / "legacy.zarr" / "raw")).info
    assert info.voxel_size == (8.0, 4.0, 4.0)
    assert info.translation == (80.0, 40.0, 40.0)


def test_zarr3_ome_group(tmp_path):
    root = tmp_path / "v3.zarr"
    ome = _ome(["0"], [[2, 3, 4]])["multiscales"]
    attrs = {"ome": {"version": "0.5", "multiscales": ome}}
    _write_json(root, "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": attrs})
    ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(root / "0")}},
        create=True,
        dtype=ts.uint8,
        shape=DATA.shape,
    ).result().write(DATA).result()
    ms = open_source(str(root))
    assert len(ms) == 1
    assert ms[0].info.voxel_size == (2.0, 3.0, 4.0)


def test_precomputed_voxel_offset_becomes_translation(tmp_path):
    t = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {"driver": "file", "path": str(tmp_path / "vol")},
            "multiscale_metadata": {"type": "image", "data_type": "uint8", "num_channels": 2},
            "scale_metadata": {
                "size": [30, 20, 10],
                "resolution": [4, 5, 6],
                "voxel_offset": [3, 2, 1],
                "chunk_size": [16, 16, 8],
                "encoding": "raw",
            },
        },
        create=True,
    ).result()
    t.write(np.stack([DATA.T, 255 - DATA.T], axis=-1)).result()  # (x, y, z, c)
    s = open_source(str(tmp_path / "vol"))[0]
    assert s.info.shape == (2, *DATA.shape)
    assert s.info.voxel_size == (1.0, 6.0, 5.0, 4.0)
    assert s.info.translation == (0.0, 6.0, 10.0, 12.0)
    np.testing.assert_array_equal(s.read(Box((1, 0, 0, 0), (2, 10, 20, 30)))[0], 255 - DATA)


def test_hdf5_resolution_and_offset(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "v.h5", "w") as f:
        d = f.create_dataset("raw", data=DATA)
        d.attrs["resolution"] = [8, 4, 4]
        d.attrs["offset"] = [80, 40, 40]
    info = open_source(f"{tmp_path / 'v.h5'}::/raw")[0].info
    assert info.voxel_size == (8.0, 4.0, 4.0)
    assert info.translation == (80.0, 40.0, 40.0)
