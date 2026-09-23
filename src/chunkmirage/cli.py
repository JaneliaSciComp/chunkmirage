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
    chunk: str | None = typer.Option(
        None, help="output chunk shape, e.g. 64,64,64 (default: source chunks)"
    ),
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    public_url: str | None = typer.Option(
        None, help="URL clients use to reach this server (behind a proxy/tunnel)"
    ),
    cache_gb: float = typer.Option(2.0, help="in-process chunk cache size"),
    source_cache_gb: float = typer.Option(
        0.5, help="tensorstore raw-byte cache for remote sources"
    ),
    viewer: str = typer.Option("https://neuroglancer-demo.appspot.com"),
    format: str = typer.Option("zarr3", help="format used for the printed neuroglancer link"),
    workers: int = typer.Option(1, help="uvicorn worker processes (caches are per-process)"),
):
    """Serve SOURCE through a pipeline of ops as n5 / zarr / zarr3 / precomputed."""
    import uvicorn

    from chunkmirage.cache import LRUCache
    from chunkmirage.neuroglancer import source_url, viewer_link
    from chunkmirage.pipeline import PipelineSpec
    from chunkmirage.server import DatasetRegistry, create_app

    spec = PipelineSpec(
        source=source,
        ops=[_parse_op(o) for o in op],
        chunk_shape=[int(c) for c in chunk.split(",")] if chunk else None,
    )
    registry = DatasetRegistry(
        LRUCache(int(cache_gb * 1024**3)), source_cache_bytes=int(source_cache_gb * 1024**3)
    )
    pipeline = registry.add(name, spec)
    base = (public_url or f"http://localhost:{port}").rstrip("/")
    scheme = {"n5": "n5", "zarr": "zarr2", "zarr3": "zarr3", "precomputed": "precomputed"}[format]
    src = source_url(base, name, format, scheme, pipeline.digest())
    typer.echo(f"source:      {src}")
    typer.echo(f"neuroglancer: {viewer_link(pipeline, name, src, viewer)}")
    typer.echo(f"control API:  {base}/api/datasets/{name}")
    application = create_app(registry, public_url=public_url)
    uvicorn.run(
        application,
        host=host,
        port=port,
        workers=workers if workers > 1 else None,
        log_level="info",
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
