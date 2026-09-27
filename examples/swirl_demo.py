"""Twist a volume live: procedural swirls, their displacement field as an RGB layer, and the
volume warped through them, each computed chunk by chunk as Neuroglancer asks (nothing is
written, and the swirls themselves are never stored: they are a formula).

    uv run python examples/swirl_demo.py [IMAGE] [--channel N] [--animate]

By default ``--count`` 3-D swirls sit at random places (``--seed``), each turning about its
own randomly tilted axis and fading with distance from its centre; ``--field swirl`` is a
single twist about an axis along z instead.

IMAGE is anything chunkmirage reads: a zarr/n5/precomputed path or URL, or synthetic://...
(default: synthetic blobs and shells). A python-neuroglancer viewer opens with two panels:

* left: ``original`` (red) and ``swirled`` (green), blended additively, so they are yellow
  where they agree and split into red and green where the swirl moved things;
* right: ``swirl_field``, the displacement as RGB = (z, y, x) components; grey is none.

The swirl grows along a ``t`` axis of ``--frames`` frames, from none at the first to
``--angle`` at the last. ``--animate`` plays it back and forth with Neuroglancer's own
playback (or click ``t`` in the top bar). Every frame is its own set of chunks, so nothing
is invalidated: the first pass computes frames as they come on screen, later passes come
from the browser's and the server's caches. Changing a parameter instead (PUT a new URL
to ``/api/datasets/{name}``) refetches the chunks on screen.
"""

from __future__ import annotations

import argparse

import numpy as np

DEFAULT = "synthetic://blobs+shells?shape=512,512,512&chunk=64,64,64&levels=4"
SPATIAL = ("z", "y", "x")

IMAGE_SHADER = """#uicontrol invlerp normalized
#uicontrol vec3 colour color(default="{colour}")
void main() {{ emitRGB(colour * normalized()); }}
"""
FIELD_SHADER = """#uicontrol float gain slider(min=0.0, max={max_gain:.6g}, default={gain:.6g})
void main() {{
  vec3 d = vec3(getDataValue(0), getDataValue(1), getDataValue(2));
  emitRGB(clamp(vec3(0.5) + gain * d, 0.0, 1.0));
}}
"""


def warp_url(image: str, extra: str, field: bool = False) -> str:
    return f"warp://{image}?{extra}" + ("&show=field" if field else "")


def coarse_read(pipeline, channel: int | None = None) -> np.ndarray:
    """The whole coarsest level, for picking display ranges: one channel of an image, or
    every channel (all displacement components) when ``channel`` is None, at the last
    time point (the most twisted frame)."""
    from chunkmirage.core import Box

    lvl = pipeline.num_levels - 1
    info = pipeline.info(lvl)
    n_lead = next(i for i, a in enumerate(info.axes) if a in SPATIAL)
    lead = []
    for a, n in zip(info.axes[:n_lead], info.shape[:n_lead]):
        if a == "c":
            lead.append((0, n) if channel is None else (channel, channel + 1))
        else:
            lead.append((n - 1, n))
    start = tuple(lo for lo, _ in lead) + (0,) * (info.ndim - n_lead)
    stop = tuple(hi for _, hi in lead) + info.shape[n_lead:]
    return pipeline.read(lvl, Box(start, stop))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("image", nargs="?", default=DEFAULT)
    ap.add_argument(
        "--channel", type=int, default=0, help="channel to show, for images with a c axis"
    )
    ap.add_argument("--angle", type=float, default=90.0, help="twist on the axis, degrees")
    ap.add_argument("--field", choices=["swirls", "swirl"], default="swirls")
    ap.add_argument("--count", type=int, default=8, help="swirls: how many")
    ap.add_argument("--seed", type=int, default=0, help="swirls: where they go")
    ap.add_argument("--radius", type=float, help="fall-off distance, image units")
    ap.add_argument(
        "--centre", help="swirl: a point on its axis; swirls: middle of where they go (z,y,x)"
    )
    ap.add_argument(
        "--spread",
        help="swirls: half-size of the box around --centre they fall in "
        "(default 1.5 radii with --centre, else the whole volume)",
    )
    ap.add_argument("--plane", default="y,x", help="swirl: the two axes that turn")
    ap.add_argument("--frames", type=int, default=24, help="frames from no twist to --angle")
    ap.add_argument("--animate", action="store_true", help="play the frames back and forth")
    ap.add_argument("--fps", type=float, default=4.0, help="playback speed, frames per second")
    ap.add_argument(
        "--chunk",
        default="8,128,128",
        help="output chunks, z,y,x: thin in z means less to compute for the x-y view",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--viewer-port", type=int, default=8001)
    ap.add_argument(
        "--public-url", help="address browsers use for chunks (default http://localhost:PORT)"
    )
    ap.add_argument("--threads", type=int, default=16, help="chunk-computing threads")
    args = ap.parse_args()

    import neuroglancer
    import uvicorn

    from chunkmirage.server import DatasetRegistry, create_app
    from chunkmirage.viewer import Viewer

    extra = f"field={args.field}&angle={args.angle:g}&frames={args.frames}&chunk={args.chunk}"
    extra += f"&radius={args.radius:g}" if args.radius else ""
    extra += f"&centre={args.centre}" if args.centre else ""
    if args.field == "swirl":
        extra += f"&plane={args.plane}"
    else:
        extra += f"&count={args.count}&seed={args.seed}"
        spread = args.spread or (f"{1.5 * args.radius:g}" if args.centre and args.radius else None)
        extra += f"&spread={spread}" if spread else ""
    registry = DatasetRegistry()
    registry.add("original", {"source": args.image})
    registry.add("swirled", {"source": warp_url(args.image, extra)})
    registry.add("swirl_field", {"source": warp_url(args.image, extra, field=True)})

    public = args.public_url or f"http://localhost:{args.port}"
    viewer = Viewer(registry, public, bind_address=args.host, port=args.viewer_port)
    viewer.set_dimensions("swirled")  # space, then t: the frames

    original, field = registry.get("original"), registry.get("swirl_field")
    lo, hi = np.percentile(coarse_read(original, args.channel), [1, 99.5])
    dmax = float(np.abs(coarse_read(field)).max()) or 1.0
    info0 = original.info(0)
    # Shader channels (getDataValue(0..2)) need a c^ dimension; zarr's channel axis is c'.
    viewer.rename_dimensions("swirl_field", {"c'": "c^"})
    if "t" in info0.axes:  # the image's single time point: shown at every frame
        viewer.rename_dimensions("original", {"t": "t'"})
    with viewer.viewer.txn() as s:
        s.layers["swirl_field"].shader = FIELD_SHADER.format(gain=0.5 / dmax, max_gain=2.0 / dmax)
        s.layers["swirl_field"].cross_section_render_scale = 2  # smooth: a coarser level will do
        for name, colour in (("original", "#ff0000"), ("swirled", "#00ff00")):
            layer = s.layers[name]
            layer.shader = IMAGE_SHADER.format(colour=colour)
            layer.shader_controls = {"normalized": {"range": [float(lo), float(max(hi, lo + 1))]}}
            layer.blend = "additive"
            if "c" in info0.axes:  # local dimensions: c', and t' for the original
                t_local = name == "original" and "t" in info0.axes
                layer.local_position = [0, args.channel] if t_local else [args.channel]
        spatial = [a for a in info0.axes if a in SPATIAL]
        vox = dict(zip(info0.axes, info0.voxel_size))
        extent = {a: n * vox[a] for a, n in zip(info0.axes, info0.shape) if a in spatial}
        if args.field == "swirl":
            radius = args.radius or 0.25 * min(extent[a] for a in args.plane.split(","))
            width = 3 * radius  # about three radii across a panel
        elif args.centre and args.radius:
            spread = [float(v) for v in (args.spread or f"{1.5 * args.radius}").split(",")]
            width = 2 * (spread[-1] + args.radius)
        else:
            width = extent[spatial[-1]]
        s.cross_section_scale = width / vox[spatial[-1]] / 800  # x voxels per screen pixel
        if args.centre:  # look where the swirl is (viewer position is in s0 voxels)
            at = dict(zip(spatial, map(float, args.centre.split(","))))
            origin = dict(zip(info0.axes, info0.translation))
            s.position = [
                (at[a] - origin[a]) / vox[a] if a in at else 0.5 for a in s.dimensions.names
            ]
        s.velocity["t"] = neuroglancer.DimensionPlaybackVelocity(
            velocity=args.fps, at_boundary="reverse", paused=not args.animate
        )
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=["original", "swirled"], layout="xy"),
                neuroglancer.LayerGroupViewer(layers=["swirl_field"], layout="xy"),
            ]
        )

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}  (forward both ports if this runs on a remote machine)")
    print(
        "left: original (red) + swirled (green); right: the swirl field as RGB (z, y, x)\n"
        f"t: {args.frames} frames from no twist to {args.angle:g} degrees; click t to play\n",
        flush=True,
    )
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
