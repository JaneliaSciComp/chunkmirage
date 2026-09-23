from __future__ import annotations

import json

import typer

app = typer.Typer(
    help="chunkmirage: spoof chunked array formats over HTTP with live processing.",
    no_args_is_help=True,
)


def _parse_op(text: str) -> dict:
    """``threshold:low=120,high=200`` or JSON ``{"op":"threshold","low":120}``."""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    name, _, params = text.partition(":")
    spec: dict = {"op": name}
    for kv in filter(None, params.split(",")):
        k, _, v = kv.partition("=")
        try:
            spec[k] = json.loads(v)
        except json.JSONDecodeError:
            spec[k] = v
    return spec


def build_registry(
    source: str,
    name: str,
    ops: list[str],
    chunk: str | None,
    *,
    raw: bool = False,
    cache_gb: float = 2.0,
    source_cache_gb: float = 0.5,
):
    """Build the registry the CLI serves (separated from `serve` so it can be tested)."""
    from chunkmirage.cache import LRUCache
    from chunkmirage.pipeline import PipelineSpec
    from chunkmirage.server import DatasetRegistry

    chunk_shape = [int(c) for c in chunk.split(",")] if chunk else None
    registry = DatasetRegistry(
        LRUCache(int(cache_gb * 1024**3)), source_cache_bytes=int(source_cache_gb * 1024**3)
    )
    if raw:
        registry.add(f"{name}-raw", PipelineSpec(source=source, ops=[], chunk_shape=chunk_shape))
    registry.add(
        name, PipelineSpec(source=source, ops=[_parse_op(o) for o in ops], chunk_shape=chunk_shape)
    )
    return registry


@app.command()
def serve(
    source: str = typer.Argument(
        ...,
        help="zarr/n5/precomputed path or URL (file, s3://, gs://, http(s)://), or file.h5::/dataset",
    ),
    name: str = typer.Option("data", help="dataset name in the served URL"),
    op: list[str] = typer.Option(
        [], "--op", "-o", help="op spec, e.g. threshold:low=120 (repeatable)"
    ),
    raw: bool = typer.Option(
        True,
        "--raw/--no-raw",
        help="also serve the unprocessed source as '<name>-raw' so it shows as a layer (shares the cache)",
    ),
    chunk: str | None = typer.Option(
        None, help="output chunk shape, e.g. 64,64,64 (default: source chunks)"
    ),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    https: bool = typer.Option(
        False,
        "--https",
        help="serve https with a self-signed certificate (required for the hosted appspot "
        "viewer to reach a server that is not on localhost)",
    ),
    cert: str | None = typer.Option(
        None, help="certificate file for --https (default: auto-generated)"
    ),
    key: str | None = typer.Option(
        None, help="private key file for --https (default: auto-generated)"
    ),
    public_url: str | None = typer.Option(
        None,
        help="URL clients use to reach this server. Default: this machine's network address "
        "(http(s)://<lan-ip>:<port>) when binding 0.0.0.0, else http(s)://localhost:<port>",
    ),
    cache_gb: float = typer.Option(2.0, help="in-process chunk cache size"),
    source_cache_gb: float = typer.Option(
        0.5, help="tensorstore raw-byte cache for remote sources"
    ),
    viewer: str = typer.Option("https://neuroglancer-demo.appspot.com"),
    format: str = typer.Option("zarr3", help="format used for the printed neuroglancer link"),
    workers: int = typer.Option(1, help="uvicorn worker processes (caches are per-process)"),
    python_viewer: bool = typer.Option(
        False,
        "--python-viewer",
        help="also start a python-neuroglancer viewer whose layers follow live edits",
    ),
    ng_client: str = typer.Option(
        "bundled", help="client build for --python-viewer: 'bundled', 'appspot', or a URL"
    ),
    viewer_host: str | None = typer.Option(
        None, help="bind address for --python-viewer (default: same as --host)"
    ),
    viewer_port: int = typer.Option(0, help="port for --python-viewer (default: random free port)"),
):
    """Serve SOURCE through a pipeline of ops as n5 / zarr / zarr3 / precomputed."""
    import uvicorn

    from chunkmirage.netutil import ensure_self_signed_cert, public_host_for
    from chunkmirage.neuroglancer import source_url, viewer_link
    from chunkmirage.server import create_app

    registry = build_registry(
        source, name, op, chunk, raw=raw, cache_gb=cache_gb, source_cache_gb=source_cache_gb
    )
    public_host = public_host_for(host)
    ssl: dict = {}
    if https:
        if cert and key:
            certfile, keyfile = cert, key
        else:
            certfile, keyfile = ensure_self_signed_cert(hosts=[public_host])
        ssl = {"ssl_certfile": certfile, "ssl_keyfile": keyfile}
    scheme_http = "https" if https else "http"
    base = (public_url or f"{scheme_http}://{public_host}:{port}").rstrip("/")
    public_url = base
    scheme = {"n5": "n5", "zarr": "zarr2", "zarr3": "zarr3", "precomputed": "precomputed"}[format]
    pipes = dict(registry.items())
    srcs = {n: source_url(base, n, format, scheme, p.digest()) for n, p in pipes.items()}
    for n, s in srcs.items():
        typer.echo(f"source [{n}]: {s}")
    typer.echo(f"neuroglancer: {viewer_link(pipes, srcs, viewer)}")
    typer.echo(f"control UI:   {base}/ui")
    typer.echo(f"control API:  {base}/api/datasets/{name}")
    if https and not (cert and key):
        typer.echo(
            f"https:        self-signed certificate ({ssl['ssl_certfile']}). Each browser must "
            f"trust it once: open {base}/ and accept the warning, then load the viewer link."
        )
    if python_viewer:
        from chunkmirage.viewer import Viewer

        v = Viewer(
            registry,
            base,
            format=format,
            client=ng_client,
            bind_address=viewer_host or host,
            port=viewer_port,
            public_host=public_host,
        )
        typer.echo(f"python viewer: {v.url}   (layers follow live edits; camera preserved)")
    application = create_app(registry, public_url=public_url)
    uvicorn.run(
        application,
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        log_level="info",
        **ssl,
    )


@app.command()
def ops():
    """List registered ops and their parameters."""
    from chunkmirage.ops.base import list_ops

    for name, meta in list_ops().items():
        props = meta["schema"].get("properties", {})
        params = ", ".join(f"{k}={v.get('default')!r}" for k, v in props.items())
        typer.echo(f"{name:14s} halo={meta['halo']!s:4s} cache={meta['cache']!s:5s} {params}")
        if meta["doc"]:
            typer.echo(f"{'':14s} {meta['doc'].splitlines()[0]}")


@app.command()
def inspect(source: str):
    """Print what chunkmirage sees when opening SOURCE."""
    from chunkmirage.sources import open_source

    ms = open_source(source)
    for i, lvl in enumerate(ms.levels):
        info = lvl.info
        typer.echo(
            f"s{i}: shape={info.shape} chunks={info.chunk_shape} dtype={info.dtype} "
            f"voxel_size={info.voxel_size} units={info.units} axes={info.axes}"
        )


if __name__ == "__main__":
    app()
