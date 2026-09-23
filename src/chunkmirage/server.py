"""Starlette ASGI app exposing pipelines through every frontend plus a small control API.

URL layout (all CORS-open)::

    /                                   index: datasets, formats, links
    /ui                                 built-in control page (live parameter editing)
    /api/ops                            registered ops + JSON schemas
    /api/datasets                       GET list | POST {"name":..., "spec": PipelineSpec}
    /api/datasets/{name}                GET spec | PUT PipelineSpec (live edit) | DELETE
    /api/datasets/{name}/neuroglancer   ?format=... -> {"url":..., "source":...}
    /api/neuroglancer                   ?format=... -> one viewer state with all datasets
    /api/events                         Server-Sent Events: fires whenever a pipeline changes
    /api/cache                          GET stats | DELETE clear
    /{name}/{format}/{path}             the spoofed dataset
    /{name}/@{digest}/{format}/{path}   same, with a cache-busting token in the path

Chunk computation runs in the default threadpool; numpy, numcodecs and tensorstore all
release the GIL for the heavy parts, so a single process serves many chunks concurrently.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from chunkmirage.cache import LRUCache
from chunkmirage.frontends import FRONTENDS, ChunkRequest, Frontend, Metadata, get_frontend
from chunkmirage.neuroglancer import source_url, viewer_link, viewer_state
from chunkmirage.ops.base import list_ops
from chunkmirage.pipeline import Pipeline, PipelineSpec

log = logging.getLogger("chunkmirage")

ChangeCallback = Callable[[str, "Pipeline | None"], None]


class DatasetRegistry:
    """Named pipelines sharing one cache. Thread-safe for live edits.

    ``version`` increments on every add/remove; ``subscribe`` registers a callback invoked
    (outside the lock, in the editing thread) with ``(name, pipeline_or_None)``.
    """

    def __init__(self, cache: LRUCache | None = None, source_cache_bytes: int = 0):
        self.cache = cache or LRUCache()
        self.source_cache_bytes = source_cache_bytes
        self._pipelines: dict[str, Pipeline] = {}
        self._lock = threading.RLock()
        self._callbacks: list[ChangeCallback] = []
        self.version = 0
        #: URL of an attached python-neuroglancer viewer, if any (set by chunkmirage.viewer.Viewer)
        self.viewer_url: str | None = None

    def add(self, name: str, pipeline: Pipeline | PipelineSpec | dict) -> Pipeline:
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline.from_spec(
                pipeline, cache=self.cache, source_cache_bytes=self.source_cache_bytes
            )
        with self._lock:
            self._pipelines[name] = pipeline
            self.version += 1
        self._notify(name, pipeline)
        return pipeline

    def get(self, name: str) -> Pipeline | None:
        with self._lock:
            return self._pipelines.get(name)

    def remove(self, name: str) -> bool:
        with self._lock:
            removed = self._pipelines.pop(name, None) is not None
            if removed:
                self.version += 1
        if removed:
            self._notify(name, None)
        return removed

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._pipelines)

    def items(self) -> list[tuple[str, Pipeline]]:
        with self._lock:
            return sorted(self._pipelines.items())

    def subscribe(self, callback: ChangeCallback) -> Callable[[], None]:
        """Register ``callback(name, pipeline)``; returns an unsubscribe function."""
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def _notify(self, name: str, pipeline: Pipeline | None) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(name, pipeline)
            except Exception:  # noqa: BLE001 - one bad subscriber must not break edits
                log.exception("registry subscriber failed")


def _public_url(request: Request, override: str | None) -> str:
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")


def _level_info(p: Pipeline) -> list[dict]:
    return [
        {
            "shape": list(p.info(i).shape),
            "chunk_shape": list(p.info(i).chunk_shape),
            "dtype": p.info(i).dtype.name,
            "voxel_size": list(p.info(i).voxel_size),
            "units": list(p.info(i).units),
            "axes": list(p.info(i).axes),
        }
        for i in range(p.num_levels)
    ]


def create_app(
    datasets: Mapping[str, Pipeline | PipelineSpec | dict] | DatasetRegistry | None = None,
    *,
    frontends: Mapping[str, Frontend] | None = None,
    public_url: str | None = None,
    cache: LRUCache | None = None,
    allow_edit: bool = True,
) -> Starlette:
    if isinstance(datasets, DatasetRegistry):
        registry = datasets
    else:
        if cache is None and datasets:
            # Report/clear the cache the pipelines actually use, if they were built with one.
            pipes = [p for p in datasets.values() if isinstance(p, Pipeline)]
            cache = pipes[0].cache if pipes else None
        registry = DatasetRegistry(cache)
        for name, p in (datasets or {}).items():
            registry.add(name, p)
    fronts: dict[str, Frontend] = (
        dict(frontends) if frontends else {n: get_frontend(n) for n in FRONTENDS}
    )
    ui_html = (files("chunkmirage") / "static" / "ui.html").read_text(encoding="utf-8")

    def links(request: Request, name: str, pipeline: Pipeline) -> dict:
        base = _public_url(request, public_url)
        d = pipeline.digest()
        return {
            fmt: source_url(base, name, fmt, fe.neuroglancer_scheme, d)
            for fmt, fe in fronts.items()
        }

    def dataset_summary(request: Request, name: str, p: Pipeline) -> dict:
        return {
            "name": name,
            "spec": p.spec.model_dump() if p.spec else None,
            "digest": p.digest(),
            "source_dtype": p.source.levels[0].info.dtype.name,
            "levels": _level_info(p),
            "sources": links(request, name, p),
        }

    def combined_state(request: Request, fmt: str, viewer: str) -> dict:
        pipes = dict(registry.items())
        srcs = {name: links(request, name, p)[fmt] for name, p in pipes.items()}
        state = viewer_state(pipes, srcs)
        return {"state": state, "url": viewer_link(pipes, srcs, viewer), "sources": srcs}

    async def index(request: Request):
        out = {name: dataset_summary(request, name, p) for name, p in registry.items()}
        return JSONResponse(
            {
                "datasets": out,
                "formats": sorted(fronts),
                "version": registry.version,
                "viewer_url": registry.viewer_url,
                "cache": registry.cache.stats(),
            }
        )

    async def ui(request: Request):
        return HTMLResponse(ui_html)

    async def ops(request: Request):
        return JSONResponse(list_ops())

    async def list_datasets(request: Request):
        return JSONResponse({"datasets": registry.names()})

    async def create_dataset(request: Request):
        if not allow_edit:
            return JSONResponse({"error": "editing disabled"}, 403)
        body = await request.json()
        name = body.get("name")
        spec = body.get("spec", body)
        if not name:
            return JSONResponse({"error": "missing 'name'"}, 400)
        try:
            p = await run_in_threadpool(registry.add, name, PipelineSpec.model_validate(spec))
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, 400)
        return JSONResponse(dataset_summary(request, name, p), 201)

    async def get_dataset(request: Request):
        name = request.path_params["name"]
        p = registry.get(name)
        if p is None:
            return JSONResponse({"error": "not found"}, 404)
        return JSONResponse(dataset_summary(request, name, p))

    async def put_dataset(request: Request):
        if not allow_edit:
            return JSONResponse({"error": "editing disabled"}, 403)
        name = request.path_params["name"]
        try:
            spec = PipelineSpec.model_validate(await request.json())
            p = await run_in_threadpool(registry.add, name, spec)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, 400)
        return JSONResponse(dataset_summary(request, name, p))

    async def delete_dataset(request: Request):
        if not allow_edit:
            return JSONResponse({"error": "editing disabled"}, 403)
        ok = registry.remove(request.path_params["name"])
        return JSONResponse({"removed": ok}, 200 if ok else 404)

    async def neuroglancer_one(request: Request):
        name = request.path_params["name"]
        p = registry.get(name)
        if p is None:
            return JSONResponse({"error": "not found"}, 404)
        fmt = request.query_params.get("format", "zarr3")
        if fmt not in fronts:
            return JSONResponse({"error": f"unknown format {fmt}"}, 400)
        viewer = request.query_params.get("viewer", "https://neuroglancer-demo.appspot.com")
        src = links(request, name, p)[fmt]
        return JSONResponse({"source": src, "url": viewer_link({name: p}, {name: src}, viewer)})

    async def neuroglancer_all(request: Request):
        fmt = request.query_params.get("format", "zarr3")
        if fmt not in fronts:
            return JSONResponse({"error": f"unknown format {fmt}"}, 400)
        viewer = request.query_params.get("viewer", "https://neuroglancer-demo.appspot.com")
        return JSONResponse(combined_state(request, fmt, viewer))

    async def events(request: Request):
        """SSE stream: an initial ``change`` event, then one per registry edit."""
        gen = event_stream(
            registry, lambda name, p: links(request, name, p), request.is_disconnected
        )
        return StreamingResponse(
            gen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def cache_stats(request: Request):
        return JSONResponse(registry.cache.stats())

    async def cache_clear(request: Request):
        return JSONResponse({"cleared": registry.cache.invalidate()})

    async def serve(request: Request):
        name = request.path_params["name"]
        fmt = request.path_params["format"]
        rest = request.path_params.get("path", "")
        p = registry.get(name)
        fe = fronts.get(fmt)
        if p is None or fe is None:
            return Response("not found", 404)
        try:
            resolved = fe.resolve(p, rest)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, 400)
        if resolved is None:
            return Response("not found", 404)
        if isinstance(resolved, Metadata):
            return Response(
                resolved.body,
                media_type=resolved.content_type,
                headers={"Cache-Control": "no-cache"},
            )
        assert isinstance(resolved, ChunkRequest)
        try:
            body = await run_in_threadpool(_compute_and_encode, p, fe, resolved)
        except Exception as e:  # noqa: BLE001
            log.exception("chunk %s/%s/%s failed", name, fmt, rest)
            return JSONResponse({"error": str(e)}, 500)
        headers = {"Cache-Control": "no-cache"}
        if (
            fe.name == "precomputed"
            and "gzip" in request.headers.get("accept-encoding", "")
            and len(body) > 1024
        ):
            body = await run_in_threadpool(gzip.compress, body, 3)
            headers["Content-Encoding"] = "gzip"
        return Response(body, media_type="application/octet-stream", headers=headers)

    routes = [
        Route("/", index),
        Route("/ui", ui),
        Route("/api/ops", ops),
        Route("/api/datasets", list_datasets, methods=["GET"]),
        Route("/api/datasets", create_dataset, methods=["POST"]),
        Route("/api/datasets/{name}", get_dataset, methods=["GET"]),
        Route("/api/datasets/{name}", put_dataset, methods=["PUT"]),
        Route("/api/datasets/{name}", delete_dataset, methods=["DELETE"]),
        Route("/api/datasets/{name}/neuroglancer", neuroglancer_one),
        Route("/api/neuroglancer", neuroglancer_all),
        Route("/api/events", events),
        Route("/api/cache", cache_stats, methods=["GET"]),
        Route("/api/cache", cache_clear, methods=["DELETE"]),
        Route("/{name}/@{digest}/{format}", serve),
        Route("/{name}/@{digest}/{format}/{path:path}", serve),
        Route("/{name}/{format}", serve),
        Route("/{name}/{format}/{path:path}", serve),
    ]
    middleware = [
        # allow_private_network: Chrome's Private Network Access preflight, sent when a public
        # https page (e.g. the hosted Neuroglancer) fetches from a 10.x/192.168.x address.
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_private_network=True,
        ),
    ]
    app = Starlette(routes=routes, middleware=middleware)
    app.state.registry = registry
    app.state.frontends = fronts
    return app


def event_payload(registry: DatasetRegistry, links_fn: Callable[[str, Pipeline], dict]) -> dict:
    return {
        "version": registry.version,
        "datasets": {
            name: {"digest": p.digest(), "sources": links_fn(name, p)}
            for name, p in registry.items()
        },
    }


async def event_stream(
    registry: DatasetRegistry,
    links_fn: Callable[[str, Pipeline], dict],
    is_disconnected: Callable[[], Awaitable[bool]],
    *,
    poll_seconds: float = 0.25,
    keepalive_seconds: float = 15.0,
):
    """Yield SSE frames: a ``change`` event whenever ``registry.version`` moves, else keepalives."""
    last_version = -1
    last_beat = time.monotonic()
    while not await is_disconnected():
        v = registry.version
        if v != last_version:
            last_version = v
            yield f"event: change\ndata: {json.dumps(event_payload(registry, links_fn))}\n\n"
            last_beat = time.monotonic()
        elif time.monotonic() - last_beat > keepalive_seconds:
            yield ": keepalive\n\n"
            last_beat = time.monotonic()
        await asyncio.sleep(poll_seconds)


def _compute_and_encode(p: Pipeline, fe: Frontend, req: ChunkRequest) -> bytes:
    block = p.chunk(req.level, req.index)
    return fe.encode(p.info(req.level), req.index, block)


__all__: list[Any] = ["DatasetRegistry", "create_app", "event_stream", "event_payload"]
