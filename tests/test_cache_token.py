"""An op whose output depends on state outside its parameters (weights replaced in place)
says so through cache_token; a refresh then gives its dataset new stage keys and links."""

import numpy as np
from starlette.testclient import TestClient

from chunkmirage import create_app, open_source
from chunkmirage.ops.base import Op
from chunkmirage.pipeline import Pipeline

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"
WEIGHTS = {"version": 1, "gain": 2.0}  # stands in for a file updated in place


class _Weighted(Op):
    name = "test_weighted"
    cache = True

    def cache_token(self):
        return str(WEIGHTS["version"])

    def apply(self, block):
        return np.full_like(block, WEIGHTS["gain"])


def test_a_new_token_changes_the_digest_and_refresh_serves_the_new_output():
    WEIGHTS.update(version=1, gain=2.0)
    app = create_app({"d": Pipeline(open_source(SRC), [_Weighted()])})
    registry, c = app.state.registry, TestClient(app)
    before = c.get("/api/datasets/d").json()
    first = registry.get("d").chunk(0, (0, 0, 0)).copy()
    r = c.post("/api/datasets/d/refresh").json()
    assert r["changed"] is False and r["digest"] == before["digest"]  # nothing changed outside

    WEIGHTS.update(version=2, gain=1.0)
    version = registry.version
    assert registry.get("d").digest() == before["digest"]  # built pipelines keep their keys
    r = c.post("/api/datasets/d/refresh").json()
    assert r["changed"] is True and r["digest"] != before["digest"]
    assert r["sources"]["zarr3"] != before["sources"]["zarr3"]  # viewers see a new URL
    assert registry.version == version + 1  # and an event
    second = registry.get("d").chunk(0, (0, 0, 0))
    assert not np.array_equal(first, second)  # computed anew, not the cached chunk
    assert c.post("/api/datasets/nope/refresh").status_code == 404


def test_ops_without_a_token_keep_their_digests():
    import hashlib
    import json

    from chunkmirage.ops.base import op_from_spec

    op = op_from_spec({"op": "threshold", "low": 1})
    old = json.dumps({"op": op.name, **op.model_dump(mode="json")}, sort_keys=True, default=str)
    assert op.digest() == hashlib.sha1(old.encode()).hexdigest()[:12]  # links stay valid
