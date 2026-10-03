"""What an array's values are (``ArrayInfo.kind``), declared by the ops that make them and
read by whatever shows them: a segmentation layer for labels and masks, an image otherwise,
the dtype guessed from only where nothing says."""

import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.core import ArrayInfo
from chunkmirage.neuroglancer import layer_for

pytest.importorskip("scipy")

BLOBS = "synthetic://blobs?shape=32,32,32&chunk=16,16,16&levels=1"


@pytest.mark.parametrize(
    "ops, kind",
    [
        ([], None),
        ([{"op": "threshold", "low": 100}], "mask"),
        ([{"op": "threshold", "low": 100}, {"op": "morphology"}], "mask"),
        ([{"op": "threshold", "low": 100}, {"op": "label"}], "label"),
        ([{"op": "threshold", "low": 100}, {"op": "label"}, {"op": "cast", "dtype": "uint64"}], "label"),
        ([{"op": "threshold", "low": 100}, {"op": "gaussian"}], None),  # a mask blurred is not one
        ([{"op": "spots", "threshold": 10}], "label"),
        ([{"op": "dog"}], "image"),
        ([{"op": "gradient"}], "image"),
    ],
)
def test_ops_say_what_their_values_are(ops, kind):
    assert Pipeline(open_source(BLOBS), ops).info(0).kind == kind


def test_contacts_are_a_mask():
    a = BLOBS
    p = Pipeline(open_source(f"stack://{a}|{a}"), [{"op": "contacts", "radius": 1, "a_low": 100, "b_low": 100}])
    assert p.info(0).kind == "mask"


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        ArrayInfo((1,), "uint8", (1,), (1.0,), ("",), ("x",), kind="blob")


def test_the_viewer_and_precomputed_follow_the_kind():
    def layer(ops):
        return layer_for("d", Pipeline(open_source(BLOBS), ops), "zarr://x")

    mask = layer([{"op": "threshold", "low": 100}, {"op": "morphology"}])
    assert mask["type"] == "segmentation" and mask["segments"] == ["1"]  # its one segment
    assert layer([{"op": "threshold", "low": 100, "value": 7}])["segments"] == ["7"]
    labels = layer([{"op": "threshold", "low": 100}, {"op": "label"}])
    assert labels["type"] == "segmentation" and "segments" not in labels
    assert layer([{"op": "gaussian"}])["type"] == "image"
    assert layer([{"op": "threshold", "low": 100}, {"op": "label"}, {"op": "scale", "factor": 2}])["type"] == "image"

    pipes = {
        "mask": Pipeline(open_source(BLOBS), [{"op": "threshold", "low": 100}]),
        "raw": Pipeline(open_source(BLOBS), []),
        "ids": Pipeline(open_source(BLOBS), [{"op": "cast", "dtype": "uint32"}]),  # kind unknown: by dtype
    }
    client = TestClient(create_app(pipes))
    assert client.get("/mask/precomputed/info").json()["type"] == "segmentation"
    assert client.get("/raw/precomputed/info").json()["type"] == "image"
    assert client.get("/ids/precomputed/info").json()["type"] == "segmentation"
    assert client.get("/api/datasets/mask").json()["levels"][0]["kind"] == "mask"


def _zarr(path, data):
    import tensorstore as ts

    spec = {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}}
    meta = {"shape": list(data.shape), "chunks": [8] * data.ndim, "dtype": data.dtype.str}
    ts.open({**spec, "metadata": meta}, create=True).result().write(data).result()
    return str(path)


def test_stored_wide_integers_are_labels_and_resampled_as_such(tmp_path):
    """A uint32 volume read from storage is labels unless told otherwise: an op at another
    voxel size reads it resampled by nearest voxel, so no ids are invented between two."""
    import numpy as np

    from chunkmirage.ops.base import Op

    ids = np.zeros((16, 16, 16), np.uint32)
    ids[:, :, 5:] = 1000
    ids[:, :, 11:] = 2000
    labels = _zarr(tmp_path / "labels.zarr", ids)
    image = _zarr(tmp_path / "image.zarr", ids.astype(np.uint8))
    assert open_source(labels)[0].info.kind == "label"
    assert open_source(image)[0].info.kind is None
    told = open_source(labels, kind="image")
    assert told[0].info.kind == "image" and told[0].cache_key() != open_source(labels)[0].cache_key()

    class AtCoarser(Op):
        name = "test_at_coarser"

        def input_voxel_size(self):
            return (1.5, 1.5, 1.5)

        def apply(self, block):
            return block

    out = Pipeline(open_source(labels), [AtCoarser()]).levels[0]
    values = set(np.unique(out.read(_whole(out))))
    assert values <= {0, 1000, 2000}
    blurred = Pipeline(open_source(labels, kind="image"), [AtCoarser()]).levels[0]
    assert not set(np.unique(blurred.read(_whole(blurred)))) <= {0, 1000, 2000}


def _whole(src):
    from chunkmirage.core import Box

    return Box((0,) * src.info.ndim, src.info.shape)
