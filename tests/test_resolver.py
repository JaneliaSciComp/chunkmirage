"""Datasets built on first request by a resolver, from nothing but their name."""

import threading
import time

from starlette.testclient import TestClient

from chunkmirage import create_app
from chunkmirage.cli import load_object
from chunkmirage.server import DatasetRegistry

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"
calls: list[str] = []


def threshold_names(name):
    """thr-<low> is the source thresholded at low; any other name is unknown."""
    calls.append(name)
    if name == "bad":
        raise ValueError("not a threshold")
    if not name.startswith("thr-"):
        return None
    time.sleep(0.05)  # long enough for concurrent requests to overlap
    return {"source": SRC, "ops": [{"op": "threshold", "low": int(name[4:])}]}


def test_an_unregistered_name_is_built_once_and_then_served_like_any_dataset():
    calls.clear()
    app = create_app({}, resolver=threshold_names)
    c = TestClient(app)
    assert c.get("/thr-100/zarr3/s0/zarr.json").status_code == 200
    assert app.state.registry.names() == ["thr-100"]
    assert c.get("/thr-100/zarr3/s0/c/0/0/0").status_code == 200
    assert calls == ["thr-100"]  # registered now, not resolved again
    d = c.get("/api/datasets/thr-100").json()
    assert d["spec"]["ops"] == [{"op": "threshold", "low": 100}]
    assert c.get(f"/thr-100/@{d['digest']}/zarr3/s0/zarr.json").status_code == 200
    assert c.get("/api/datasets/thr-7/neuroglancer").json()["source"].startswith("zarr3://")
    assert c.get("/other/zarr3/zarr.json").status_code == 404
    r = c.get("/bad/zarr3/zarr.json")
    assert r.status_code == 400 and "not a threshold" in r.json()["error"]
    assert c.get("/thr-100/nosuchformat/x").status_code == 404 and calls.count("thr-100") == 1


def test_concurrent_requests_for_a_new_name_build_it_once():
    calls.clear()
    registry = DatasetRegistry(resolver=threshold_names)
    found = []
    threads = [threading.Thread(target=lambda: found.append(registry.resolve("thr-5"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == ["thr-5"] and len({id(p) for p in found}) == 1


def test_the_cli_loads_a_resolver_by_name():
    import os.path

    assert load_object("os.path:join") is os.path.join
