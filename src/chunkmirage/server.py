"""Starlette ASGI app exposing pipelines through every frontend plus a small control API.

URL layout (all CORS-open)::

    /                                   index: datasets, formats, links
    /api/ops                            registered ops + JSON schemas
    /api/datasets                       GET list | POST {"name":..., "spec": PipelineSpec}
    /api/datasets/{name}                GET spec | PUT PipelineSpec (live edit) | DELETE
    /api/datasets/{name}/neuroglancer   ?format=n5|zarr|zarr3|precomputed -> {"url":..., "source":...}
    /api/cache                          GET stats | DELETE clear
    /{name}/{format}/{path}             the spoofed dataset
    /{name}/@{digest}/{format}/{path}   same, with a cache-busting token in the path

Chunk computation runs in the default threadpool; numpy, numcodecs and tensorstore all
release the GIL for the heavy parts, so a single process serves many chunks concurrently.
"""

from __future__ import annotations

import gzip
import logging
import threading
from collections.abc import Mapping

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from chunkmirage.cache import LRUCache
from chunkmirage.frontends import FRONTENDS, ChunkRequest, Frontend, Metadata, get_frontend
from chunkmirage.neuroglancer import source_url, viewer_link
from chunkmirage.ops.base import list_ops
from chunkmirage.pipeline import Pipeline, PipelineSpec

log = logging.getLogger("chunkmirage")


class DatasetRegistry:
    """Named pipelines sharing one cache. Thread-safe for live edits."""

    def __init__(self, cache: LRUCache | None = None, source_cache_bytes: int = 0):
        self.cache = cache or LRUCache()
        self.source_cache_bytes = source_cache_bytes
        self._pipelines: dict[str, Pipeline] = {}
        self._lock = threading.RLock()

    def add(self, name: str, pipeline: Pipeline | PipelineSpec | dict) -> Pipeline:
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline.from_spec(
                pipeline, cache=self.cache, source_cache_bytes=self.source_cache_bytes
            )
        with self._lock:
            self._pipelines[name] = pipeline
        return pipeline

    def get(self, name: str) -> Pipeline | None:
        with self._lock:
            return self._pipelines.get(name)

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._pipelines.pop(name, None) is not None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._pipelines)


def _public_url(request: Request, override: str | None) -> str:
    if override:
        return override.rstrip("/")
    return str(request.base_url).rstrip("/")


def create_app(
    datasets: Mapping[str, Pipeline | PipelineSpec | dict] | DatasetRegistry | None = None,
    *,
    frontends: Mapping[str, Frontend] | None = None,
    public_url: str | None = None,
    cache: LRUCache | None = None,
    allow_edit: bool = True,
) -> Starlette:
    registry = datasets if isinstance(datasets, DatasetRegistry) else DatasetRegistry(cache)
    if datasets and not isinstance(datasets, DatasetRegistry):
        for name, p in datasets.items():
            registry.add(name, p)
    fronts: dict[str, Frontend] = (
        dict(frontends) if frontends else {n: get_frontend(n) for n in FRONTENDS}
    )

    def links(request: Request, name: str, pipeline: Pipeline) -> dict:
        base = _public_url(request, public_url)
        d = pipeline.digest()
        return {
            fmt: source_url(base, name, fmt, fe.neuroglancer_scheme, d)
            for fmt, fe in fronts.items()
        }

    async def index(request: Request):
        out = {}
        for name in registry.names():
            p = registry.get(name)
            if p is None:
                continue
            out[name] = {
                "spec": p.spec.model_dump() if p.spec else None,
                "levels": p.num_levels,
                "shape": list(p.info(0).shape),
                "dtype": p.info(0).dtype.name,
                "sources": links(request, name, p),
            }
        return JSONResponse(
            {"datasets": out, "formats": sorted(fronts), "cache": registry.cache.stats()}
        )

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
        return JSONResponse(
            {"name": name, "digest": p.digest(), "sources": links(request, name, p)}, 201
        )

    async def get_dataset(request: Request):
        name = request.path_params["name"]
        p = registry.get(name)
        if p is None:
            return JSONResponse({"error": "not found"}, 404)
        return JSONResponse(
            {
                "name": name,
                "spec": p.spec.model_dump() if p.spec else None,
                "digest": p.digest(),
                "levels": [
                    {
                        "shape": list(p.info(i).shape),
                        "chunk_shape": list(p.info(i).chunk_shape),
                        "dtype": p.info(i).dtype.name,
                        "voxel_size": list(p.info(i).voxel_size),
                        "units": list(p.info(i).units),
                        "axes": list(p.info(i).axes),
                    }
                    for i in range(p.num_levels)
                ],
                "sources": links(request, name, p),
            }
        )

    async def put_dataset(request: Request):
        if not allow_edit:
            return JSONResponse({"error": "editing disabled"}, 403)
        name = request.path_params["name"]
        try:
            spec = PipelineSpec.model_validate(await request.json())
            p = await run_in_threadpool(registry.add, name, spec)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"error": str(e)}, 400)
        return JSONResponse(
            {"name": name, "digest": p.digest(), "sources": links(request, name, p)}
        )

    async def delete_dataset(request: Request):
        if not allow_edit:
            return JSONResponse({"error": "editing disabled"}, 403)
        ok = registry.remove(request.path_params["name"])
        return JSONResponse({"removed": ok}, 200 if ok else 404)

    async def neuroglancer(request: Request):
        name = request.path_params["name"]
        p = registry.get(name)
        if p is None:
            return JSONResponse({"error": "not found"}, 404)
        fmt = request.query_params.get("format", "zarr3")
        if fmt not in fronts:
            return JSONResponse({"error": f"unknown format {fmt}"}, 400)
        viewer = request.query_params.get("viewer", "https://neuroglancer-demo.appspot.com")
        src = links(request, name, p)[fmt]
        return JSONResponse({"source": src, "url": viewer_link(p, name, src, viewer)})

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
        except Exception as e:
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
        Route("/api/ops", ops),
        Route("/api/datasets", list_datasets, methods=["GET"]),
        Route("/api/datasets", create_dataset, methods=["POST"]),
        Route("/api/datasets/{name}", get_dataset, methods=["GET"]),
        Route("/api/datasets/{name}", put_dataset, methods=["PUT"]),
        Route("/api/datasets/{name}", delete_dataset, methods=["DELETE"]),
        Route("/api/datasets/{name}/neuroglancer", neuroglancer),
        Route("/api/cache", cache_stats, methods=["GET"]),
        Route("/api/cache", cache_clear, methods=["DELETE"]),
        Route("/{name}/@{digest}/{format}", serve),
        Route("/{name}/@{digest}/{format}/{path:path}", serve),
        Route("/{name}/{format}", serve),
        Route("/{name}/{format}/{path:path}", serve),
    ]
    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ]
    app = Starlette(routes=routes, middleware=middleware)
    app.state.registry = registry
    app.state.frontends = fronts
    return app


def _compute_and_encode(p: Pipeline, fe: Frontend, req: ChunkRequest) -> bytes:
    block = p.chunk(req.level, req.index)
    return fe.encode(p.info(req.level), req.index, block)
