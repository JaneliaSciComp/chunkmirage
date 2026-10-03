"""Starlette ASGI app exposing pipelines through every frontend plus a small control API.

URL layout (all CORS-open; with a token, ``/api/*`` needs it)::

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

Other packages add routes of their own (``extra_routes``, the ``chunkmirage.routes`` entry
point). Every path is relative to where the app is mounted, so it works as a sub-app too.

Chunk computation runs in the default threadpool; numpy, numcodecs and tensorstore all
release the GIL for the heavy parts, so a single process serves many chunks concurrently.
"""

from __future__ import annotations

import asyncio
import gzip
import hmac
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from importlib.metadata import entry_points
from importlib.resources import files
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import BaseRoute, Route

from chunkmirage import demand
from chunkmirage.cache import LRUCache
from chunkmirage.frontends import FRONTENDS, ChunkRequest, Frontend, Metadata, get_frontend
from chunkmirage.neuroglancer import source_url, viewer_link, viewer_state
from chunkmirage.ops.base import list_ops
from chunkmirage.pipeline import Pipeline, PipelineSpec

log = logging.getLogger("chunkmirage")

ChangeCallback = Callable[[str, "Pipeline | None"], None]
#: Builds the pipeline for a dataset name nobody registered, or returns None if it has none
Resolver = Callable[[str], "Pipeline | PipelineSpec | dict | None"]


class DatasetRegistry:
    """Named pipelines sharing one cache. Thread-safe for live edits.

    ``version`` increments on every add/remove; ``subscribe`` registers a callback invoked
    (outside the lock, in the editing thread) with ``(name, pipeline_or_None)``.

    ``resolver``, if given, is asked for a name requested but not registered (``resolve``):
    what it returns is built and registered under that name, once however many requests
    ask at the same time, and served from then on like any other dataset.
    """

    def __init__(
        self,
        cache: LRUCache | None = None,
        source_cache_bytes: int = 0,
        resolver: Resolver | None = None,
    ):
        self.cache = cache or LRUCache()
        self.source_cache_bytes = source_cache_bytes
        self.resolver = resolver
        self._pipelines: dict[str, Pipeline] = {}
        self._lock = threading.RLock()
        self._resolving: dict[str, threading.Lock] = {}
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

    def resolve(self, name: str) -> Pipeline | None:
        """The pipeline named ``name``: the registered one, else the one the resolver builds
        for it (then registered), else None. Exceptions from the resolver propagate."""
        p = self.get(name)
        if p is not None or self.resolver is None:
            return p
        with self._lock:
            lock = self._resolving.setdefault(name, threading.Lock())
        try:
            with lock:  # one build per name; later callers find it registered
                p = self.get(name)
                if p is None:
                    found = self.resolver(name)
                    if found is not None:
                        p = self.add(name, found)
        finally:
            with self._lock:
                self._resolving.pop(name, None)
        return p

    def refresh(self, name: str | None = None) -> list[str]:
        """Rebuild dataset ``name`` (every one if None) so its ops' ``cache_token`` is read
        again, after the state outside their parameters changed. Returns the names whose
        digest changed: they get new links, and subscribers and ``/api/events`` hear of it."""
        names = self.names() if name is None else [name]
        changed = []
        for n in names:
            old = self.get(n)
            if old is None:
                if name is not None:
                    raise KeyError(n)
                continue
            new = old.rebuilt()
            if new.digest() == old.digest():
                continue
            with self._lock:
                if self._pipelines.get(n) is not old:  # edited meanwhile: the edit stands
                    continue
                self._pipelines[n] = new
                self.version += 1
            self._notify(n, new)
            changed.append(n)
        return changed

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


def _route_path(scope) -> str:
    """The request's path relative to where the app is mounted: ``scope["path"]`` holds the
    whole path and ``root_path`` the mount's prefix (and a proxy's)."""
    path, root = scope["path"], scope.get("root_path", "")
    if root and path.startswith(root) and path[len(root) : len(root) + 1] in ("", "/"):
        return path[len(root) :] or "/"
    return path


def _public_url(request: Request, override: str | None) -> str:
    if override:
        return override.rstrip("/")
    # ``root_path`` carries a Mount's prefix; ``request.base_url`` leaves it out by design
    url = request.url
    return f"{url.scheme}://{url.netloc}{request.scope.get('root_path', '')}".rstrip("/")


def _level_info(p: Pipeline) -> list[dict]:
    return [
        {
            "shape": list(p.info(i).shape),
            "chunk_shape": list(p.info(i).chunk_shape),
            "dtype": p.info(i).dtype.name,
            "voxel_size": list(p.info(i).voxel_size),
            "units": list(p.info(i).units),
            "axes": list(p.info(i).axes),
            "kind": p.info(i).kind,
        }
        for i in range(p.num_levels)
    ]


def set_compute_threads(n: int) -> None:
    """Size the threadpool that computes chunks (default 40). numpy/scipy/tensorstore release
    the GIL, so this is the server's parallelism for chunk work; pure-Python ops serialise.
    Call it from the event loop that serves: the pool belongs to it (an app mounted in
    another shares the host's pool)."""
    import anyio

    # ``n`` compute at once (demand.Slots); threads of requests waiting on queued work (their
    # slot given up meanwhile) or for a slot must not run the pool out
    limiter = anyio.to_thread.current_default_thread_limiter()
    if limiter.total_tokens < int(n) + 1024:
        limiter.total_tokens = int(n) + 1024


def plugin_routes(registry: DatasetRegistry) -> list[BaseRoute]:
    """Routes from the ``chunkmirage.routes`` entry point: each names a function taking the
    app's registry and returning Starlette routes. A plugin that fails to load is skipped,
    with a warning."""
    routes: list[BaseRoute] = []
    for ep in entry_points(group="chunkmirage.routes"):
        try:
            routes.extend(ep.load()(registry))
        except Exception:  # noqa: BLE001 - a broken plugin must not take the server down
            log.warning("route plugin %r failed to load; skipped", ep.name, exc_info=True)
    return routes


def create_app(
    datasets: Mapping[str, Pipeline | PipelineSpec | dict] | DatasetRegistry | None = None,
    *,
    frontends: Mapping[str, Frontend] | None = None,
    public_url: str | None = None,
    cache: LRUCache | None = None,
    allow_edit: bool = True,
    threads: int | None = None,
    token: str | None = None,
    extra_routes: Sequence[BaseRoute] = (),
    route_plugins: bool = True,
    resolver: Resolver | None = None,
) -> Starlette:
    """The ASGI app serving ``datasets`` (a mapping of names to pipelines or specs, or a
    :class:`DatasetRegistry`) through every frontend, with the control API.

    * ``public_url``: base of the links the app hands out (default: the request's own, with
      the prefix the app is mounted under).
    * ``allow_edit``: whether ``/api/datasets`` may create, replace and remove datasets.
    * ``threads``: chunk requests computing at once (default 40).
    * ``token``: required on ``/api/*`` (header or ``?token=``); the datasets stay open.
    * ``extra_routes``: Starlette routes of your own, matched after the built-in ones and
      before the datasets (``/{name}/...``); under ``/api/`` they share the token.
    * ``route_plugins``: also add the routes of installed ``chunkmirage.routes`` plugins.
    * ``resolver``: builds datasets requested by a name nobody registered
      (``DatasetRegistry.resolve``); the registry's own if not given.

    It can be mounted in another Starlette app (``Mount("/prefix", app)``): paths, the token
    and links are then relative to the prefix.
    """
    # chunk requests computing at once (the rest wait their turn, or give theirs up while they
    # wait on queued work)
    slots = demand.Slots(int(threads) if threads else 40)
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
    if resolver is not None:
        registry.resolver = resolver
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
            "ops_info": [
                {
                    "op": op.name,
                    "halo": list(op.halo_for(p.info(0).ndim)),
                    "cached": op.cached,
                    "doc": (op.__doc__ or "").strip(),
                }
                for op in p.ops
            ],
            "levels": _level_info(p),
            "reads": p.input_read,
            "sources": links(request, name, p),
        }

    async def lookup(name: str) -> Pipeline | Response | None:
        """The dataset ``name``, resolved if need be; a response saying why it could not be."""
        p = registry.get(name)
        if p is not None or registry.resolver is None:
            return p
        try:
            return await run_in_threadpool(registry.resolve, name)
        except ValueError as e:
            log.warning("could not resolve dataset %r: %s", name, e)
            return JSONResponse({"error": f"{name}: {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            log.exception("resolving dataset %r failed", name)
            return JSONResponse({"error": f"{name}: {e}"}, 500)

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
        p = await lookup(name)
        if isinstance(p, Response):
            return p
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

    async def refresh_dataset(request: Request):
        name = request.path_params["name"]
        try:
            changed = await run_in_threadpool(registry.refresh, name)
        except KeyError:
            return JSONResponse({"error": "not found"}, 404)
        p = registry.get(name)
        return JSONResponse({"changed": bool(changed), "digest": p.digest(), "sources": links(request, name, p)})

    async def neuroglancer_one(request: Request):
        name = request.path_params["name"]
        p = await lookup(name)
        if isinstance(p, Response):
            return p
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

    async def queue_stats(request: Request):
        queues = {name: q.stats() for name, q in demand.queues.items()}
        return JSONResponse({"requests": slots.stats(), **queues})

    async def serve(request: Request):
        name = request.path_params["name"]
        fmt = request.path_params["format"]
        rest = request.path_params.get("path", "")
        fe = fronts.get(fmt)
        p = await lookup(name) if fe is not None else None
        if isinstance(p, Response):
            return p
        if p is None or fe is None:
            return Response("not found", 404)
        set_compute_threads(slots.n)  # here, not at startup: a mounted app gets no lifespan
        # A group or level asked for as a directory (a trailing /) or as a page (HTML first in
        # Accept, as browsers and Java's HTTP client send) gets a listing: that is how Fiji's
        # N5 viewer finds the levels. Metadata clients (Neuroglancer, tensorstore, zarr-python)
        # ask for */* and get the metadata, as before.
        directory = request.url.path.endswith("/")
        page = request.headers.get("accept", "").lstrip().startswith("text/html")
        if directory or page:
            entries = fe.listing(p, rest)
            if entries is not None:
                return HTMLResponse(_listing_html(request.url.path, entries))
            if directory:
                return Response("not found", 404)
        try:
            resolved = None
            span = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers.get("range", "").strip())
            if span:  # files read in parts (multi-resolution mesh fragments)
                resolved = fe.resolve_range(p, rest, int(span.group(1)), int(span.group(2)) + 1)
            if resolved is None:
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
        # The request claims the work it needs; if its client disconnects, the claim is
        # cancelled and work nobody else wants is dropped: a turn it waits for, and queued
        # expensive work (``demand``).
        claim = demand.Claim()
        watch = asyncio.create_task(_watch_disconnect(request, claim))
        try:
            body = await run_in_threadpool(
                demand.claimed, claim, _compute_and_encode, p, fe, resolved, slots=slots
            )
        except demand.Cancelled:
            return Response(status_code=499)  # the client stopped waiting
        except asyncio.CancelledError:
            claim.cancel()  # the server cancelled the request (its client went): its work too
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("chunk %s/%s/%s failed", name, fmt, rest)
            return JSONResponse({"error": str(e)}, 500)
        finally:
            watch.cancel()
        headers = {"Cache-Control": "no-cache"}
        if (
            fe.name in ("precomputed", "mesh")  # a padded mesh fragment is mostly zeros
            and "gzip" in request.headers.get("accept-encoding", "")
            and len(body) > 1024
        ):
            body = await run_in_threadpool(gzip.compress, body, 3)
            headers["Content-Encoding"] = "gzip"
        if resolved.part is not None:
            a, b = resolved.part
            headers["Content-Range"] = f"bytes {a}-{b - 1}/{resolved.total}"
            return Response(body, 206, media_type="application/octet-stream", headers=headers)
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
        Route("/api/datasets/{name}/refresh", refresh_dataset, methods=["POST"]),
        Route("/api/datasets/{name}/neuroglancer", neuroglancer_one),
        Route("/api/neuroglancer", neuroglancer_all),
        Route("/api/events", events),
        Route("/api/cache", cache_stats, methods=["GET"]),
        Route("/api/cache", cache_clear, methods=["DELETE"]),
        Route("/api/queue", queue_stats, methods=["GET"]),
        *extra_routes,
        *(plugin_routes(registry) if route_plugins else ()),
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
            expose_headers=["Content-Range"],  # range reads (mesh fragments) check it
            allow_private_network=True,
        ),
    ]
    if token:  # inside CORS, so a browser's preflight is answered without one
        middleware.append(Middleware(_TokenGuard, token=token))

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


def _listing_html(path: str, entries: list[str]) -> str:
    """A directory listing as Python's http.server writes one, which HTTP clients that browse
    a store (n5-universe's, so Fiji's) parse: one link per entry, directories ending in /."""
    from html import escape

    items = "".join(f'<li><a href="{escape(e)}">{escape(e)}</a></li>' for e in entries)
    title = f"Directory listing for {escape(path)}"
    return (
        f'<!DOCTYPE HTML><html><head><meta charset="utf-8"><title>{title}</title></head>'
        f"<body><h1>{title}</h1><hr><ul>{items}</ul><hr></body></html>\n"
    )


class _TokenGuard:
    """``/api/*`` only with ``Authorization: Bearer <token>``, or ``?token=`` (for the event
    stream, which a browser's EventSource cannot give a header). Datasets and the index stay
    open: viewers send no headers."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and _route_path(scope).startswith("/api/"):
            request = Request(scope)
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            given = given or request.query_params.get("token", "")
            if not hmac.compare_digest(given.encode(), self.token):
                response = JSONResponse({"error": "this server's /api needs its token"}, 401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


async def _watch_disconnect(request: Request, claim: demand.Claim) -> None:
    """Cancel ``claim`` when the client disconnects (it aborted the request)."""
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)
    claim.cancel()


def _compute_and_encode(p: Pipeline, fe: Frontend, req: ChunkRequest) -> bytes:
    return fe.compute(p, req)


__all__: list[Any] = [
    "DatasetRegistry",
    "create_app",
    "event_stream",
    "event_payload",
    "plugin_routes",
    "Resolver",
    "set_compute_threads",
]
