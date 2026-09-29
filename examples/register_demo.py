"""Register one image onto another while you look: a deformable field solved on the GPU in
seconds, and the moving image served through it at every resolution, chunk by chunk, with
nothing written.

    uv run python examples/register_demo.py [FIXED MOVING] [--affine M.npy]
        [--fixed-channel N] [--moving-channel N] [--frames N]

FIXED and MOVING are anything chunkmirage reads (zarr, n5, precomputed, ...), with the same
spatial units; ``--affine`` is a fixed-to-moving matrix in those units (``register://``'s
``affine``). Without them, the moving image is a synthetic volume twisted by random swirls
(``warp://``) and the fixed one is the volume itself, so the answer is known. A python
Neuroglancer viewer opens with two panels that move together:

* left, ``before``: the affine alone;
* right, ``after``: the affine and the solved field.

Each panel is one layer holding the fixed image and the moving image as two channels, and
its shader compares them on the viewer's own GPU: ``mode`` 0 overlays them (fixed magenta,
moving green, white where they agree), 1 is a checkerboard (``squares`` pixels across),
2 fades from one to the other (``fade``), 3 is their difference. Hidden layers hold the
registered moving image with all its channels and the field itself (RGB = z, y, x).

``--frames N`` puts N snapshots of the solve on a ``t`` axis: click ``t`` (or play) to watch
it converge. While the viewer is open, type settings at the prompt (``smooth=3 grid=2``) to
solve again with them; the viewer refetches, keeping its camera.
"""

from __future__ import annotations

import argparse
import threading
from urllib.parse import quote

import numpy as np

SYNTHETIC = "synthetic://blobs+noise?shape=128,256,256&chunk=64,64,64&levels=4&voxel_size=64"
SWIRLS = "field=swirls&count=6&seed=2&angle=40&radius=3000"
SPATIAL = ("z", "y", "x")
EDITABLE = ("smooth", "grid", "window", "iterations", "levels")

COMPARE_SHADER = """#uicontrol int mode slider(min=0, max=3, default=0)
#uicontrol float fade slider(min=0, max=1, default=0.5)
#uicontrol float squares slider(min=4, max=256, default=48)
#uicontrol float gain slider(min=1, max=10, default=3)
#uicontrol invlerp fixed(range=[{f0:.6g}, {f1:.6g}], channel=0)
#uicontrol invlerp moving(range=[{m0:.6g}, {m1:.6g}], channel=1)
void main() {{
  float f = fixed();
  float m = moving();
  if (mode == 0) {{
    emitRGB(vec3(f, m, f));
  }} else if (mode == 1) {{
    float odd = mod(floor(gl_FragCoord.x / squares) + floor(gl_FragCoord.y / squares), 2.0);
    emitGrayscale(odd > 0.5 ? m : f);
  }} else if (mode == 2) {{
    emitGrayscale(mix(f, m, fade));
  }} else {{
    float d = gain * (m - f);
    emitRGB(vec3(max(d, 0.0), 0.0, max(-d, 0.0)));
  }}
}}
"""
FIELD_SHADER = """#uicontrol float gain slider(min=0.0, max={max_gain:.6g}, default={gain:.6g})
void main() {{
  vec3 d = vec3(getDataValue(0), getDataValue(1), getDataValue(2));
  emitRGB(clamp(vec3(0.5) + gain * d, 0.0, 1.0));
}}
"""


def coarse_read(pipeline, channel: int | None = None) -> np.ndarray:
    """A coarse level whole (the coarsest with at least 16 voxels on every spatial axis),
    for display ranges: one channel, or all of them; the last time point."""
    from chunkmirage.core import Box

    lvl = pipeline.num_levels - 1
    while lvl > 0 and min(pipeline.info(lvl).shape[-3:]) < 16:
        lvl -= 1
    info = pipeline.info(lvl)
    n_lead = info.ndim - 3
    lead = []
    for a, n in zip(info.axes[:n_lead], info.shape[:n_lead]):
        if a == "c" and channel is not None:
            lead.append((channel, channel + 1))
        else:
            lead.append((0, n) if a == "c" else (n - 1, n))
    start = tuple(lo for lo, _ in lead) + (0,) * 3
    stop = tuple(hi for _, hi in lead) + info.shape[n_lead:]
    return pipeline.read(lvl, Box(start, stop))


def display_range(a: np.ndarray) -> tuple[float, float]:
    """Contrast limits from the voxels with data: where the moving image does not reach,
    the registered one is 0, which would otherwise pull the lower limit down to it."""
    a = a[a > 0]
    lo, hi = np.percentile(a, [1, 99.8]) if a.size else (0.0, 1.0)
    return float(lo), float(max(hi, lo + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fixed", nargs="?", help="fixed image (default: a synthetic volume)")
    ap.add_argument("moving", nargs="?", help="moving image (default: the volume, swirled)")
    ap.add_argument("--affine", help="fixed-to-moving affine: .npy or text file, 4x4 or 3x4")
    ap.add_argument("--fixed-channel", type=int, default=0, help="channel of FIXED to match on")
    ap.add_argument("--moving-channel", type=int, default=0, help="channel of MOVING to match on")
    ap.add_argument("--smooth", type=float, help="weight of the field's smoothness (default 1)")
    ap.add_argument("--grid", type=float, help="control-point spacing, voxels (default 4)")
    ap.add_argument("--levels", help="fixed levels to solve on, coarse to fine, e.g. 6,5,4")
    ap.add_argument("--iterations", help="steps per level, one value or one per level")
    ap.add_argument("--frames", type=int, help="snapshots of the solve on a t axis")
    ap.add_argument("--device", default="auto", help="auto (the freest GPU), cpu, cuda:1, ...")
    ap.add_argument(
        "--chunk",
        default="8,128,128",
        help="output chunks, z,y,x: thin in z means less to compute for the x-y view",
    )
    ap.add_argument(
        "--source-cache-gb",
        type=float,
        default=2.0,
        help="decoded chunks kept per source array (large source chunks need it)",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="bind address (default: every interface, so links use this machine's IP and "
        "work for others; 127.0.0.1 keeps it local, with localhost links over plain http)",
    )
    ap.add_argument(
        "--no-https",
        action="store_true",
        help="serve chunks over plain http on the network: the viewer link still works "
        "from other machines, the appspot link does not (https pages cannot fetch http)",
    )
    ap.add_argument("--port", type=int, help="chunk server port (default: 8000 or the next free)")
    ap.add_argument(
        "--viewer-port", type=int, help="python viewer port (default: 8001 or the next free)"
    )
    ap.add_argument(
        "--public-url", help="address browsers use for chunks (default: this machine's IP)"
    )
    ap.add_argument("--threads", type=int, default=16, help="chunk-computing threads")
    args = ap.parse_args()
    if (args.fixed is None) != (args.moving is None):
        ap.error("give both FIXED and MOVING, or neither")

    import logging

    import neuroglancer
    import uvicorn

    from chunkmirage.netutil import free_port, is_loopback, public_host_for, serving_address
    from chunkmirage.server import DatasetRegistry, create_app
    from chunkmirage.viewer import Viewer

    log = logging.getLogger("chunkmirage")  # the solve's progress
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    log.propagate = False  # once, whatever a library did to the root logger
    fixed = args.fixed or SYNTHETIC
    moving = args.moving or f"warp://{SYNTHETIC}?{SWIRLS}"
    port = args.port or free_port(args.host, 8000)
    viewer_port = args.viewer_port or free_port(args.host, 8001, avoid={port})

    settings = {
        k: getattr(args, k) for k in ("smooth", "grid", "levels", "iterations") if getattr(args, k)
    }

    def url(dataset: str) -> str:
        q = {
            "fixed": quote(fixed, safe=""),
            "fixed_channel": args.fixed_channel,
            "moving_channel": args.moving_channel,
            "device": args.device,
            "chunk": args.chunk,
        }
        if args.affine:
            q["affine"] = quote(args.affine, safe="")
        if dataset == "before":  # the affine alone, whatever the settings
            q["iterations"] = 0
        else:
            q.update(settings)
            if args.frames:
                q["frames"] = args.frames
        q["show"] = {"before": "pair", "after": "pair", "field": "field"}.get(dataset, "image")
        return f"register://{moving}?" + "&".join(f"{k}={v}" for k, v in q.items())

    registry = DatasetRegistry(source_cache_bytes=int(args.source_cache_gb * 2**30))

    def load() -> None:  # the first add solves; the rest reuse its field
        for name in ("after", "before", "registered", "field"):
            registry.add(name, {"source": url(name)})

    load()
    https = not (args.no_https or is_loopback(args.host))
    address, ssl = serving_address(args.host, port, https=https)
    public = (args.public_url or address).rstrip("/")
    viewer = Viewer(
        registry,
        public,
        bind_address=args.host,
        port=viewer_port,
        public_host=public_host_for(args.host),
    )
    viewer.set_dimensions("after")  # space, then t: the frames
    for name in ("before", "after", "field"):
        viewer.rename_dimensions(name, {"c'": "c^"})  # the two images: shader channels
    before = registry.get("before")
    f0, f1 = display_range(coarse_read(before, 0))
    m0, m1 = display_range(coarse_read(before, 1))
    dmax = float(np.abs(coarse_read(registry.get("field"))).max()) or 1.0
    info = before.info(0)
    with viewer.viewer.txn() as s:
        for name in ("before", "after"):
            s.layers[name].shader = COMPARE_SHADER.format(f0=f0, f1=f1, m0=m0, m1=m1)
        s.layers["field"].shader = FIELD_SHADER.format(gain=0.5 / dmax, max_gain=2.0 / dmax)
        s.layers["field"].visible = False
        s.layers["registered"].visible = False
        s.cross_section_scale = info.shape[-1] / 700  # the whole x extent across a panel
        if args.frames:
            s.velocity["t"] = neuroglancer.DimensionPlaybackVelocity(
                velocity=4, at_boundary="loop", paused=True
            )
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=["before"], layout="xy"),
                neuroglancer.LayerGroupViewer(layers=["after", "registered", "field"], layout="xy"),
            ]
        )

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}")
    if https:
        print(
            "https:   self-signed certificate: each browser (yours and anyone you send a link "
            f"to) must trust it once: open {public}/ and accept the warning"
        )
    print(
        "left: the affine alone; right: affine + solved field. Layer shader `mode`: "
        "0 overlay, 1 checkerboard, 2 fade, 3 difference.\n"
        f"Type settings to solve again ({', '.join(EDITABLE)}), e.g. `smooth=3 grid=2`.\n",
        flush=True,
    )
    threading.Thread(target=prompt, args=(settings, load), daemon=True).start()
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **ssl)


def prompt(settings: dict, load) -> None:
    """Read ``key=value`` settings from the terminal and re-solve with them."""
    while True:
        try:
            line = input("register> ").strip()
        except EOFError:  # no terminal: nothing to read
            return
        if not line:
            continue
        try:
            new = dict(tok.split("=", 1) for tok in line.split())
        except ValueError:
            print("expected key=value pairs, e.g. smooth=3 grid=2")
            continue
        if bad := set(new) - set(EDITABLE):
            print(f"unknown: {sorted(bad)}; editable: {', '.join(EDITABLE)}")
            continue
        old = dict(settings)
        settings.update(new)
        try:
            load()
        except Exception as e:  # a bad value: say so, keep the last good settings
            print(f"error: {e}")
            settings.clear()
            settings.update(old)
            continue
        print("solved; the viewer is refetching", flush=True)


if __name__ == "__main__":
    main()
