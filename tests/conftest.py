import numpy as np
import pytest
import tensorstore as ts

SHAPE = (40, 50, 70)
CHUNKS = (16, 16, 32)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def volume(rng):
    return rng.integers(0, 255, size=SHAPE, dtype=np.uint8)


def _write(spec, data):
    arr = ts.open(
        spec, create=True, delete_existing=True, dtype=ts.uint8, shape=data.shape
    ).result()
    arr[...] = data
    return arr


@pytest.fixture(scope="session")
def zarr2_path(tmp_path_factory, volume):
    """OME-NGFF style multiscale zarr v2 group with s0 (full) and s1 (2x downsampled)."""
    root = tmp_path_factory.mktemp("data") / "vol.zarr"
    group = root / "em"
    group.mkdir(parents=True)
    (group / ".zgroup").write_text('{"zarr_format": 2}')
    import json

    (group / ".zattrs").write_text(
        json.dumps(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
                        "datasets": [
                            {
                                "path": "s0",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [8, 8, 8]}
                                ],
                            },
                            {
                                "path": "s1",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [16, 16, 16]}
                                ],
                            },
                        ],
                    }
                ]
            }
        )
    )
    _write(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(group / "s0")},
            "metadata": {"chunks": list(CHUNKS), "compressor": {"id": "gzip"}, "dtype": "|u1"},
        },
        volume,
    )
    s1 = volume[::2, ::2, ::2]
    _write(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(group / "s1")},
            "metadata": {"chunks": list(CHUNKS), "compressor": {"id": "gzip"}, "dtype": "|u1"},
        },
        s1,
    )
    return str(group)


@pytest.fixture(scope="session")
def n5_path(tmp_path_factory, volume):
    """N5 array written like a Java/n5-zarr writer: dimensions and blockSize x-first."""
    root = tmp_path_factory.mktemp("data") / "vol.n5" / "raw"
    _write(
        {
            "driver": "n5",
            "kvstore": {"driver": "file", "path": str(root / "s0")},
            "metadata": {
                "blockSize": list(CHUNKS[::-1]),
                "compression": {"type": "gzip"},
                "pixelResolution": {"dimensions": [4.0, 4.0, 4.0], "unit": "nm"},
            },
        },
        np.ascontiguousarray(volume.T),
    )
    return str(root)


@pytest.fixture(scope="session")
def zarr3_path(tmp_path_factory, volume):
    root = tmp_path_factory.mktemp("data") / "vol3.zarr" / "s0"
    _write(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(root)},
            "metadata": {
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(CHUNKS)}}
            },
        },
        volume,
    )
    return str(root)
