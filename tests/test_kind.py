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
