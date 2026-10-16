"""dataset:// sources: another dataset of the server read in-process through its stage
cache, what reads a dataset rebuilt when it is edited, and cycles refused."""

import numpy as np
import pytest
from starlette.testclient import TestClient

from chunkmirage import create_app, open_source
from chunkmirage.core import Box
from chunkmirage.server import DatasetRegistry

SRC = "synthetic://blobs?shape=32,32,32&chunk=16,16,16&levels=2&seed=3"
WHOLE = Box((0, 0, 0), (32, 32, 32))


def test_a_dataset_reads_another_through_its_stages_and_follows_its_edits():
    reg = DatasetRegistry()
    a = reg.add("a", {"source": SRC, "ops": [{"op": "threshold", "low": 100}]})
    b = reg.add("b", {"source": "dataset://a", "ops": [{"op": "label"}]})
    assert reg.upstreams("b") == {"a"} and reg.dependents("a") == ["b"] and reg.upstreams("a") == set()
    assert b.source.cached and b.num_levels == a.num_levels and b.info(0).shape == a.info(0).shape
    labels = b.read(0, WHOLE)
    np.testing.assert_array_equal(labels > 0, a.read(0, WHOLE) > 0)
    # the chunks come through a's stages: a's raw chunks are cached once, b keeps no copy
    raw = [k for k in reg.cache._data if k[0].startswith("raw:")]
    assert len(raw) == np.prod(a.levels[0].info.chunk_grid) and len({k[0] for k in raw}) == 1

    seen: list[str] = []
    reg.subscribe(lambda name, p: seen.append(name))
    a2 = reg.add("a", {"source": SRC, "ops": [{"op": "threshold", "low": 150}]})
    b2 = reg.get("b")
    assert b2 is not b and b2.digest() != b.digest(), "b reads a, so a's edit rebuilds b under new keys"
    assert seen == ["a", "b"]
    np.testing.assert_array_equal(b2.read(0, WHOLE) > 0, a2.read(0, WHOLE) > 0)
    assert reg.refresh("a") == []  # nothing outside a's parameters changed: nothing rebuilt
    assert reg.remove("a") and reg.get("b") is b2  # b serves what it last read of a


def test_a_chain_of_datasets_is_rebuilt_in_order():
    reg = DatasetRegistry()
    reg.add("a", {"source": SRC})
    reg.add("b", {"source": "dataset://a", "ops": [{"op": "threshold", "low": 100}]})
    c = reg.add("c", {"source": "dataset://b", "ops": [{"op": "label"}]})
    assert reg.upstreams("c") == {"a", "b"} and reg.upstreams("c", transitive=False) == {"b"}
    seen: list[str] = []
    reg.subscribe(lambda name, p: seen.append(name))
    reg.add("a", {"source": SRC.replace("seed=3", "seed=4")})
    assert seen == ["a", "b", "c"] and reg.get("c") is not c


def test_cycles_and_unknown_names_are_refused():
    reg = DatasetRegistry()
    with pytest.raises(ValueError, match="cycle"):
        reg.add("self", {"source": "dataset://self"})
    assert reg.get("self") is None
    a = reg.add("a", {"source": SRC})
    reg.add("b", {"source": "dataset://a"})
    with pytest.raises(ValueError, match="cycle: a -> b"):
        reg.add("a", {"source": "dataset://b"})  # b reads a: a may not read b
    assert reg.get("a") is a  # the refused edit left a as it was
    with pytest.raises(ValueError, match="no dataset named 'nothing'"):
        reg.add("c", {"source": "dataset://nothing"})
    with pytest.raises(ValueError, match="DatasetRegistry"):
        open_source("dataset://a")  # on its own there is no registry to resolve against


def test_the_api_shows_what_a_dataset_reads_and_rebuilds_it_on_an_edit():
    app = create_app({"a": {"source": SRC, "ops": [{"op": "threshold", "low": 100}]}, "b": {"source": "dataset://a"}})
    c = TestClient(app)
    d = c.get("/api/datasets/b").json()
    assert d["upstream"] == ["a"] and c.get("/api/datasets/a").json()["upstream"] == []
    r = c.put("/api/datasets/a", json={"source": SRC, "ops": [{"op": "threshold", "low": 200}]})
    assert r.status_code == 200
    assert c.get("/api/datasets/b").json()["digest"] != d["digest"]
    assert c.get("/b/zarr3/s0/c/0/0/0").status_code == 200
    r = c.post("/api/datasets", json={"name": "loop", "spec": {"source": "dataset://loop"}})
    assert r.status_code == 400 and "cycle" in r.json()["error"]
