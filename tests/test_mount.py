"""The app mounted under a prefix in another app, and routes from other packages."""

import logging

import anyio
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage import server as server_module

SRC = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1"
AUTH = {"Authorization": "Bearer s3cret"}


async def _threads(request):
    return JSONResponse({"tokens": anyio.to_thread.current_default_thread_limiter().total_tokens})


def test_mounted_under_a_prefix_keeps_its_token_threads_and_links():
    inner = create_app(
        {"d": Pipeline(open_source(SRC), [])},
        token="s3cret",
        threads=7,
        extra_routes=[Route("/api/probe/threads", _threads)],
    )
    with TestClient(Starlette(routes=[Mount("/cm", inner)])) as c:
        assert c.get("/cm/api/ops").status_code == 401  # the token guard sees /api/ under /cm
        assert c.get("/cm/api/probe/threads").status_code == 401  # extra /api/ routes too
        assert c.get("/cm/d/zarr3/s0/c/0/0/0").status_code == 200  # the data stays open
        # the threadpool was sized by the chunk request, though a mounted app has no lifespan
        assert c.get("/cm/api/probe/threads", headers=AUTH).json()["tokens"] >= 7 + 1024
        src = c.get("/cm/api/datasets/d", headers=AUTH).json()["sources"]["zarr3"]
        assert src.startswith("zarr3://http://testserver/cm/d/@")  # links keep the prefix
        assert c.get(src.removeprefix("zarr3://http://testserver") + "/zarr.json").status_code == 200
        ui = c.get("/cm/ui").text
        assert "'/api/" not in ui and "`/api/" not in ui  # the page asks relative to /cm/ui


class _EntryPoint:
    def __init__(self, name, load):
        self.name, self.load = name, load


def test_route_plugins_are_added_and_a_broken_one_is_skipped(monkeypatch, caplog):
    def good(registry):
        return [Route("/api/myplugin/names", lambda r: JSONResponse(registry.names()))]

    def broken():
        raise ImportError("no such module")

    eps = [_EntryPoint("broken", broken), _EntryPoint("good", lambda: good)]
    monkeypatch.setattr(server_module, "entry_points", lambda group: eps)
    with caplog.at_level(logging.WARNING, logger="chunkmirage"):
        app = create_app({"d": Pipeline(open_source(SRC), [])})
    assert TestClient(app).get("/api/myplugin/names").json() == ["d"]
    assert "broken" in caplog.text and "no such module" in caplog.text
    off = create_app({"d": Pipeline(open_source(SRC), [])}, route_plugins=False)
    assert TestClient(off).get("/api/myplugin/names").status_code == 404
