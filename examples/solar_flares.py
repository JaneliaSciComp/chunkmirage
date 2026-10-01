"""A solar flare in the running difference, computed where you look: the SDO machine-learning
dataset (NASA's Solar Dynamics Observatory, AIA images of the sun every 6 minutes, 512 x 512,
73 thousand frames for 2014 in one 171 Å array of NASA's public bucket) and each frame minus
the one before it, computed for each chunk the viewer asks for. Nothing is precomputed.

    uv run python examples/solar_flares.py [--year 2014] [--channel 171A] [--frame 2636] [--lag 1]

A python Neuroglancer viewer opens with two panels that move together: the sun (log
brightness) and the running difference (white brighter, black dimmer than ``lag`` frames
before), the view solar physicists use to spot what changes. It opens on the X1.6 flare of
10 September 2014 at 17:36, erupting from the active region at the centre of the disk; scroll
to step through the frames (time is the third axis). Type ``lag=10`` at the prompt to
compare with an hour before.

The frames are in time order within each day, but the days are stored out of order (the
first frame of 2014 is from September), so the difference at a day's first frame compares
it with another day. The store's attributes hold every frame's FITS header, 221 MB of JSON,
read once when it opens. NASA's bucket allows no browser access (no CORS), so this one runs
through the Python server only.
"""

from __future__ import annotations

import argparse
import threading

SDOML = "https://gov-nasa-hdrl-data1.s3.amazonaws.com/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2.zarr"
LOG = """#uicontrol float lo slider(min=0, max=3, default=1)
#uicontrol float hi slider(min=2, max=5, default=4)
void main() { emitGrayscale(clamp((log(max(getDataValue(), 1.0)) / log(10.0) - lo) / (hi - lo), 0.0, 1.0)); }
"""
DIFFERENCE = """#uicontrol float range slider(min=10, max=2000, default=300)
void main() { emitGrayscale(clamp(0.5 + 0.5 * getDataValue() / range, 0.0, 1.0)); }
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--year", default="2014", help="2010 to 2020")
    ap.add_argument("--channel", default="171A", help="AIA channel: 94A, 131A, 171A, 193A, ...")
    ap.add_argument("--frame", type=int, default=2636, help="the frame the view opens on")
    ap.add_argument("--lag", type=int, default=1, help="compare with this many frames before")
    ap.add_argument("--zoom", type=float, default=1.0, help="image pixels per screen pixel")
    ap.add_argument("--chunk", default="1,256,256", help="output chunks: one frame, a quarter")
    ap.add_argument("--source-cache-gb", type=float, default=1.0, help="decoded source chunks kept")
    ap.add_argument("--cache-gb", type=float, default=1.0, help="computed chunks kept")
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

    import neuroglancer
    import uvicorn

    from chunkmirage.cache import LRUCache
    from chunkmirage.netutil import free_port, is_loopback, public_host_for, serving_address
    from chunkmirage.neuroglancer import cross_section_scale
    from chunkmirage.server import DatasetRegistry, create_app
    from chunkmirage.viewer import Viewer

    source = f"{SDOML}/{args.year}/{args.channel}"
    settings = {"lag": args.lag}
    chunk = [int(c) for c in args.chunk.split(",")]
    port = args.port or free_port(args.host, 8000)
    viewer_port = args.viewer_port or free_port(args.host, 8001, avoid={port})
    registry = DatasetRegistry(
        LRUCache(int(args.cache_gb * 2**30)), source_cache_bytes=int(args.source_cache_gb * 2**30)
    )
    # The store says nothing of its axes: a frame every 6 minutes (within a day), and AIA's
    # 0.6 arcsec pixels binned 8 times, about 3480 km on the sun's disk
    grid = {"voxel_size": [360, 3480, 3480], "units": ["s", "km", "km"], "chunk_shape": chunk}
    print("opening the store (its attributes are 221 MB of FITS headers)...", flush=True)
    registry.add("sun", {"source": source, **grid})

    def load() -> None:
        ops = [{"op": "diff", "axis": 0, "lag": settings["lag"]}]
        registry.add("difference", {"source": source, "ops": ops, **grid})

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
    shape = registry.get("sun").info(0).shape
    where = {"z": args.frame + 0.5, "y": shape[1] / 2, "x": shape[2] / 2}
    with viewer.viewer.txn() as s:
        s.layers["sun"].shader = LOG
        s.layers["difference"].shader = DIFFERENCE
        s.position = [where[n] for n in s.dimensions.names]
        dims = {
            n: [sc, u]
            for n, sc, u in zip(s.dimensions.names, s.dimensions.scales, s.dimensions.units)
        }
        s.cross_section_scale = cross_section_scale(dims, "x", args.zoom)
        s.cross_section_background_color = "#000000"
        s.show_axis_lines = False
        s.show_default_annotations = False  # the bounding box
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=["sun"], layout="xy"),
                neuroglancer.LayerGroupViewer(layers=["difference"], layout="xy"),
            ]
        )

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}")
    print(
        f"same as: chunkmirage serve '{source}' --op diff:axis=0,lag={settings['lag']} "
        f"--chunk {args.chunk} --python-viewer"
    )
    if https:
        print(
            "https:   self-signed certificate: each browser (yours and anyone you send a link "
            f"to) must trust it once: open {public}/ and accept the warning"
        )
    print("Scroll to step through the frames. Type `lag=10` to compare with an hour before.\n")
    threading.Thread(target=prompt, args=(settings, load), daemon=True).start()
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **ssl)


def prompt(settings: dict, load) -> None:
    """Read ``lag=N`` from the terminal and compute the difference against that frame."""
    while True:
        try:
            line = input("flares> ").strip()
        except EOFError:  # no terminal: nothing to read
            return
        key, _, value = line.partition("=")
        if key.strip() != "lag" or not value.strip().isdigit():
            if line:
                print("expected lag=N, e.g. lag=10")
            continue
        old = settings["lag"]
        settings["lag"] = int(value)
        try:
            load()
        except Exception as e:  # a bad value: say so, keep the last good one
            print(f"error: {e}")
            settings["lag"] = old
            continue
        print("recomputing; the viewer is refetching", flush=True)


if __name__ == "__main__":
    main()
