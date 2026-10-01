"""Hurricanes' cold wakes, day by day: NASA's MUR sea-surface temperature (daily since
June 2002, every ocean, 0.01° apart: 6443 x 17999 x 36000 values in AWS Open Data) read
straight from its public zarr store, and the change since the day before computed for each
chunk the viewer asks for. Nothing is precomputed or written.

    uv run python examples/hurricane_wakes.py [--date 2005-08-29] [--where 25,-89] [--lag 1]

A python Neuroglancer viewer opens with two panels that move together, north up: the
temperature on the left, its change since ``lag`` days before on the right (blue cooler,
red warmer). A hurricane stirs up cold water from below and leaves a cold swath along its
track: on 29 August 2005 Katrina's crosses the Gulf of Mexico, and Rita's a month later.
Scroll to step through the days (time is the third axis, so the change is a difference
along it). Type ``lag=3`` at the prompt to compare with three days before instead.

The store is xarray-written: chunkmirage takes its axes (time, lat, lon) from the dimension
names, its spacing and origin from the coordinate arrays (time in seconds since the first
day, degrees unitless), and unpacks its stored integers to kelvin, land as NaN. It has one
resolution and 65 MB tiles of 5 days x 18 x 36 degrees, so it opens zoomed in on a region:
zoomed out to the globe, the viewer would ask for every tile at full resolution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import threading

MUR = "https://mur-sst.s3.us-west-2.amazonaws.com/zarr-v1/analysed_sst"
FIRST_DAY = dt.date(2002, 6, 1)
LAND = "if (isnan(v)) { emitRGB(vec3(0.18)); return; }"
KELVIN = f"""#uicontrol invlerp temperature(range=[298, 305])
void main() {{ float v = getDataValue(); {LAND} emitRGB(colormapJet(clamp(temperature(), 0.0, 1.0))); }}
"""
CHANGE = f"""#uicontrol invlerp change(range=[-2, 2])
void main() {{
  float v = getDataValue(); {LAND}
  float t = clamp(change(), 0.0, 1.0) * 2.0 - 1.0;
  emitRGB(t < 0.0 ? mix(vec3(1.0), vec3(0.15, 0.35, 0.85), -t) : mix(vec3(1.0), vec3(0.85, 0.2, 0.15), t));
}}
"""


def spec(source: str, lag: int, chunk: list[int]) -> dict:
    return {"source": source, "chunk_shape": chunk, "ops": [{"op": "diff", "axis": 0, "lag": lag}]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default=MUR, help="a time, lat, lon zarr array")
    ap.add_argument("--date", default="2005-08-29", help="the day the view opens on")
    ap.add_argument("--where", default="25,-89", help="lat,lon the view opens on (degrees)")
    ap.add_argument("--lag", type=int, default=1, help="compare with this many days before")
    ap.add_argument("--zoom", type=float, default=2.0, help="0.01° voxels per screen pixel")
    ap.add_argument("--chunk", default="1,256,256", help="output chunks: one day, 2.56°")
    ap.add_argument("--source-cache-gb", type=float, default=2.0, help="decoded source tiles kept")
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

    settings = {"lag": args.lag}
    chunk = [int(c) for c in args.chunk.split(",")]
    port = args.port or free_port(args.host, 8000)
    viewer_port = args.viewer_port or free_port(args.host, 8001, avoid={port})
    registry = DatasetRegistry(
        LRUCache(int(args.cache_gb * 2**30)), source_cache_bytes=int(args.source_cache_gb * 2**30)
    )
    registry.add("temperature", {"source": args.source, "chunk_shape": chunk})

    def load() -> None:
        registry.add("change", spec(args.source, settings["lag"], chunk))

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
    info = registry.get("temperature").info(0)
    lat, lon = (float(v) for v in args.where.split(","))
    day = (dt.date.fromisoformat(args.date) - FIRST_DAY).days
    # the viewer's position is physical, in voxels: degrees / 0.01, and time in days
    where = {"time": day + 0.5, "lat": lat / info.voxel_size[1], "lon": lon / info.voxel_size[2]}
    with viewer.viewer.txn() as s:
        s.layers["temperature"].shader = KELVIN
        s.layers["change"].shader = CHANGE
        s.position = [where[n] for n in s.dimensions.names]
        dims = {
            n: [sc, u]
            for n, sc, u in zip(s.dimensions.names, s.dimensions.scales, s.dimensions.units)
        }
        s.cross_section_scale = cross_section_scale(dims, "lon", args.zoom)
        s.cross_section_orientation = [1, 0, 0, 0]  # latitude increases northward: north up
        s.show_axis_lines = False
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=["temperature"], layout="xy"),
                neuroglancer.LayerGroupViewer(layers=["change"], layout="xy"),
            ]
        )

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}")
    print(
        f"same as: chunkmirage serve '{args.source}' --op diff:axis=0,lag={settings['lag']} "
        f"--chunk {args.chunk} --python-viewer"
    )
    if https:
        print(
            "https:   self-signed certificate: each browser (yours and anyone you send a link "
            f"to) must trust it once: open {public}/ and accept the warning"
        )
    print("Scroll to step through the days. Type `lag=3` to compare with 3 days before.\n")
    threading.Thread(target=prompt, args=(settings, load), daemon=True).start()
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **ssl)


def prompt(settings: dict, load) -> None:
    """Read ``lag=N`` from the terminal and compute the change against that day."""
    while True:
        try:
            line = input("wakes> ").strip()
        except EOFError:  # no terminal: nothing to read
            return
        key, _, value = line.partition("=")
        if key.strip() != "lag" or not value.strip().isdigit():
            if line:
                print("expected lag=N, e.g. lag=3")
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
