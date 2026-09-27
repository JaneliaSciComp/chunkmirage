"""Change notifications, control page, combined viewer state, and the python-neuroglancer viewer."""

import json

import pytest
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.server import DatasetRegistry


@pytest.fixture
def registry(zarr2_path):
    reg = DatasetRegistry()
    reg.add("raw", Pipeline(open_source(zarr2_path), []))
    reg.add("thr", {"source": zarr2_path, "ops": [{"op": "threshold", "low": 100}]})
    return reg


def test_registry_version_and_subscribe(registry, zarr2_path):
    seen = []
    unsub = registry.subscribe(lambda name, p: seen.append((name, p is not None)))
    v0 = registry.version
    registry.add("thr", {"source": zarr2_path, "ops": [{"op": "threshold", "low": 120}]})
    assert registry.version == v0 + 1
    assert seen == [("thr", True)]
    registry.remove("thr")
    assert seen[-1] == ("thr", False)
    unsub()
    registry.add("thr", {"source": zarr2_path, "ops": []})
    assert len(seen) == 2  # unsubscribed


def test_ui_and_combined_state(registry):
    c = TestClient(create_app(registry))
    r = c.get("/ui")
    assert r.status_code == 200 and "chunkmirage" in r.text and "EventSource" in r.text
    body = c.get("/api/neuroglancer?format=precomputed").json()
    names = [layer["name"] for layer in body["state"]["layers"]]
    assert names == ["raw", "thr"]
    types = {layer["name"]: layer["type"] for layer in body["state"]["layers"]}
    assert types == {"raw": "image", "thr": "segmentation"}
    assert body["url"].startswith("https://neuroglancer-demo.appspot.com/#!")
    assert "@" in body["sources"]["thr"]  # digest embedded for cache busting


def test_events_stream_reports_digests(registry, zarr2_path):
    """TestClient/ASGITransport buffer whole responses, so drive the generator directly."""
    import asyncio

    from chunkmirage.server import event_stream

    async def run():
        frames = []
        calls = {"n": 0}

        async def disconnected():
            calls["n"] += 1
            return calls["n"] > 3  # three polls, then hang up

        links = lambda name, p: {"zarr3": f"zarr3://x/{name}/@{p.digest()}/zarr3"}  # noqa: E731
        gen = event_stream(registry, links, disconnected, poll_seconds=0.01)
        async for frame in gen:
            frames.append(frame)
            if len(frames) == 1:  # edit between polls -> a second change event
                registry.add("thr", {"source": zarr2_path, "ops": [{"op": "threshold", "low": 5}]})
        return frames

    frames = asyncio.run(run())
    changes = [f for f in frames if f.startswith("event: change")]
    assert len(changes) == 2
    first = json.loads(changes[0].split("data: ", 1)[1])
    second = json.loads(changes[1].split("data: ", 1)[1])
    assert set(first["datasets"]) == {"raw", "thr"}
    assert first["datasets"]["thr"]["digest"] != second["datasets"]["thr"]["digest"]
    assert second["version"] == first["version"] + 1


def test_events_endpoint_headers(registry):
    """Only the headers: the body is infinite, so don't read it through a buffering client."""
    app = create_app(registry)
    route = next(r for r in app.routes if getattr(r, "path", "") == "/api/events")
    assert route.methods == {"GET", "HEAD"}


def test_private_network_access_headers(registry):
    c = TestClient(create_app(registry))
    r = c.get("/thr/zarr3/zarr.json", headers={"Origin": "https://neuroglancer-demo.appspot.com"})
    assert r.headers["access-control-allow-origin"] == "*"
    pre = c.options(
        "/thr/zarr3/zarr.json",
        headers={
            "Origin": "https://neuroglancer-demo.appspot.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-private-network"] == "true"


def test_get_dataset_reports_source_dtype(registry):
    c = TestClient(create_app(registry))
    assert c.get("/api/datasets/thr").json()["source_dtype"] == "uint8"


def test_python_viewer_tracks_edits(registry, zarr2_path):
    pytest.importorskip("neuroglancer")
    from chunkmirage.viewer import Viewer

    v = Viewer(registry, "http://localhost:9999", format="zarr3")
    try:
        state = v.viewer.state
        thr = state.layers["thr"]
        src_before = thr.source[0].url
        assert src_before.startswith("zarr3://http://localhost:9999/thr/@")
        assert thr.type == "segmentation"
        # Edit through the registry (what a REST PUT does) -> the layer URL changes.
        v.set_ops("thr", [{"op": "threshold", "low": 200}])
        src_after = v.viewer.state.layers["thr"].source[0].url
        assert src_after != src_before
        assert v.url.startswith("http://127.0.0.1:")
        assert registry.viewer_url == v.url
        assert TestClient(create_app(registry)).get("/").json()["viewer_url"] == v.url
        registry.remove("raw")
        assert "raw" not in [layer.name for layer in v.viewer.state.layers]
    finally:
        v.close()


def test_viewer_dimensions_put_space_first_and_time_last(tmp_path):
    from chunkmirage.core import ArrayInfo
    from chunkmirage.neuroglancer import dimensions, global_dimensions

    info = ArrayInfo(
        shape=(5, 2, 10, 20, 30),
        dtype="uint16",
        chunk_shape=(1, 1, 8, 8, 8),
        voxel_size=(2.0, 1.0, 1.0, 0.5, 0.5),
        units=("millisecond", "", "micrometer", "micrometer", "micrometer"),
        axes=("t", "c", "z", "y", "x"),
    )
    assert dimensions(info) == {
        "t": [0.002, "s"],
        "c'": [1.0, ""],
        "z": [1e-6, "m"],
        "y": [5e-7, "m"],
        "x": [5e-7, "m"],
    }
    dims, position, display = global_dimensions(info)
    assert list(dims) == ["z", "y", "x", "t"]
    assert position == [5.0, 10.0, 15.0, 0.5] and display == ["x", "y", "z"]


def test_python_viewer_renames_dimensions_across_edits(registry):
    pytest.importorskip("neuroglancer")
    from chunkmirage.viewer import Viewer

    v = Viewer(registry, "http://localhost:9999", format="zarr3")
    try:
        v.set_dimensions("raw")
        assert list(v.viewer.state.display_dimensions) == ["x", "y", "z"]
        v.rename_dimensions("thr", {"x": "xx"})
        v.set_ops("thr", [{"op": "threshold", "low": 100}])
        src = v.viewer.state.layers["thr"].source[0]
        assert "/thr/@" in src.url
        assert list(src.transform.output_dimensions.names) == ["z", "y", "xx"]
        link = v.hosted_link()
        assert link.startswith("https://neuroglancer-demo.appspot.com/#!") and "%22xx%22" in link
    finally:
        v.close()
    with pytest.raises(ValueError, match="zarr format"):
        w = Viewer(registry, "http://localhost:9999", format="n5")
        try:
            w.rename_dimensions("thr", {"x": "xx"})
        finally:
            w.close()
