"""Single mRNA molecules found where you look: an EASI-FISH round of a whole fly central
brain (Janelia's public janelia-data-examples bucket, 3.4 gigavoxels per channel), its two
FISH channels' spots detected chunk by chunk as the viewer asks for them. Nothing is
precomputed or written.

    uv run python examples/fish_spots.py [--threshold1 10] [--threshold2 8] [--sigma 1]

A python Neuroglancer viewer opens with two panels that move together: on the left the
round's first channel in grey and its two FISH channels in green and magenta, on the right
the FISH channels with the spots found in them drawn over them as segments (one colour per
spot, kept across chunk borders). A spot is found in 3-D, so a plane shows the spots whose
centre is within a slice or two of it. Spot detection is a difference of Gaussians and its local maxima, the step a
batch pipeline usually tunes on a crop before running on everything: here the threshold is
tuned on the whole brain. Type settings at the prompt (``threshold1=20 sigma=1.5``) and the
spots on screen are found again from images already in the cache; the camera stays. The
equivalent command lines are printed.

The file name lists the round's probes (Spab at 546 nm, Nplp1 at 647 nm); which channel holds
which is not in its metadata, so the layers are named by channel.
"""

from __future__ import annotations

import argparse
import threading

ROUND = (
    "https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/"
    "NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0"
)
EDITABLE = ("threshold1", "threshold2", "sigma", "sigma_z", "separation", "radius")
TINT = """#uicontrol invlerp v(range=[{lo:.0f}, {hi:.0f}])
void main() {{
  emitRGB(vec3({r}, {g}, {b}) * v());
}}
"""


def spec(image: str, channel: int, k: int, settings: dict, chunk: list[int]) -> dict:
    """Channel ``channel``'s spots, with the ``k``-th threshold: that channel of the image,
    then `spots`."""
    op = {key: settings[key] for key in ("sigma", "sigma_z", "separation", "radius")}
    op["threshold"] = settings[f"threshold{k}"]
    return {
        "source": image,
        "select": {"c": channel, "t": 0},
        "chunk_shape": chunk,
        "ops": [{"op": "spots", **op}],
    }


def command(image: str, channel: int, k: int, settings: dict, chunk: str) -> str:
    op = spec(image, channel, k, settings, [])["ops"][0]
    params = ",".join(f"{k}={v}" for k, v in op.items() if k != "op")
    return (
        f"chunkmirage serve '{image}' --select c={channel},t=0 --op spots:{params} "
        f"--chunk {chunk} --python-viewer"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", default=ROUND, help="multichannel image (t, c, z, y, x)")
    ap.add_argument("--channels", default="1,2", help="the FISH channels, two of them")
    ap.add_argument("--structure", type=int, default=0, help="channel shown in grey underneath")
    ap.add_argument("--threshold1", type=float, default=10, help="first channel's threshold")
    ap.add_argument("--threshold2", type=float, default=8, help="second channel's threshold")
    ap.add_argument("--sigma", type=float, default=1.0, help="spot size, y-x voxels")
    ap.add_argument("--sigma-z", type=float, default=0.6, help="spot size, z voxels")
    ap.add_argument("--separation", type=int, default=2, help="closest two spots, y-x voxels")
    ap.add_argument("--radius", type=int, default=2, help="drawn ball radius, y-x voxels")
    ap.add_argument(
        "--chunk",
        default="16,128,128",
        help="output chunks, z,y,x: thin in z means less to compute for the x-y view",
    )
    ap.add_argument("--zoom", type=float, default=0.4, help="full-resolution voxels per pixel")
    ap.add_argument("--source-cache-gb", type=float, default=2.0, help="decoded source chunks kept")
    ap.add_argument("--cache-gb", type=float, default=2.0, help="computed chunks kept")
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
    import numpy as np
    import uvicorn

    from chunkmirage.cache import LRUCache
    from chunkmirage.core import Box
    from chunkmirage.netutil import free_port, is_loopback, public_host_for, serving_address
    from chunkmirage.server import DatasetRegistry, create_app
    from chunkmirage.viewer import Viewer

    channels = [int(c) for c in args.channels.split(",")]
    if len(channels) != 2:
        ap.error("--channels takes two channels, e.g. 1,2")
    settings = {
        "threshold1": args.threshold1,
        "threshold2": args.threshold2,
        "sigma": args.sigma,
        "sigma_z": args.sigma_z,
        "separation": args.separation,
        "radius": args.radius,
    }
    chunk = [int(c) for c in args.chunk.split(",")]
    port = args.port or free_port(args.host, 8000)
    viewer_port = args.viewer_port or free_port(args.host, 8001, avoid={port})
    registry = DatasetRegistry(
        LRUCache(int(args.cache_gb * 2**30)), source_cache_bytes=int(args.source_cache_gb * 2**30)
    )
    names = {c: f"channel {c}" for c in (args.structure, *channels)}
    for c, name in names.items():  # the images, through the same cache the spots read from
        registry.add(name, {"source": args.image, "select": {"c": c, "t": 0}, "chunk_shape": chunk})

    def load() -> None:
        for k, c in enumerate(channels, start=1):
            registry.add(f"spots {c}", spec(args.image, c, k, settings, chunk))

    load()

    def contrast(name: str, spots: bool) -> tuple[float, float]:
        """Display range from one chunk at the centre of the volume: the background black,
        and for a FISH channel the noise too (its 99th percentile), so spots stand out."""
        p = registry.get(name)
        info = p.info(0)
        mid = [s // 2 for s in info.shape]
        box = Box(
            tuple(m - c // 2 for m, c in zip(mid, chunk)),
            tuple(m + c // 2 for m, c in zip(mid, chunk)),
        )
        a = p.read(0, box)
        lo, hi = np.percentile(a, [99, 99.99] if spots else [50, 99.95])
        return float(lo), float(max(hi, lo + 1))

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
    colours = {
        args.structure: (0.8, 0.8, 0.8),
        channels[0]: (0.2, 1.0, 0.3),
        channels[1]: (1.0, 0.3, 1.0),
    }
    with viewer.viewer.txn() as s:
        served = {layer.name: layer.layer for layer in s.layers}
        s.layers.clear()  # layers draw in order: the images underneath, the spots on top
        for c, name in names.items():
            s.layers[name] = served[name]
            lo, hi = contrast(name, c in channels)
            r, g, b = colours[c]
            s.layers[name].shader = TINT.format(lo=lo, hi=hi, r=r, g=g, b=b)
            s.layers[name].blend = "additive"
        for c in channels:
            s.layers[f"spots {c}"] = served[f"spots {c}"]
            s.layers[f"spots {c}"].selected_alpha = 0.9
        s.cross_section_scale = args.zoom
        # two panels that move together, one plane each (3-D panels would ask for every chunk
        # in the volume): the images, and the FISH channels with the spots found in them
        s.layout = neuroglancer.row_layout(
            [
                neuroglancer.LayerGroupViewer(layers=list(names.values()), layout="xy"),
                neuroglancer.LayerGroupViewer(
                    layers=[names[c] for c in channels] + [f"spots {c}" for c in channels],
                    layout="xy",
                ),
            ]
        )

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}")
    for k, c in enumerate(channels, start=1):
        print(f"same as: {command(args.image, c, k, settings, args.chunk)}")
    if https:
        print(
            "https:   self-signed certificate: each browser (yours and anyone you send a link "
            f"to) must trust it once: open {public}/ and accept the warning"
        )
    print(
        "Spots appear as the chunks on screen are computed; pan or step through z and they "
        f"follow. Type settings ({', '.join(EDITABLE)}), e.g. `threshold1=20 sigma=1.5`.\n",
        flush=True,
    )
    threading.Thread(target=prompt, args=(settings, load), daemon=True).start()
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **ssl)


def prompt(settings: dict, load) -> None:
    """Read ``key=value`` settings from the terminal and find the spots again with them."""
    while True:
        try:
            line = input("spots> ").strip()
        except EOFError:  # no terminal: nothing to read
            return
        if not line:
            continue
        try:
            new = dict(tok.split("=", 1) for tok in line.split())
        except ValueError:
            print("expected key=value pairs, e.g. threshold1=20 sigma=1.5")
            continue
        if bad := set(new) - set(EDITABLE):
            print(f"unknown: {sorted(bad)}; editable: {', '.join(EDITABLE)}")
            continue
        old = dict(settings)
        settings.update(
            {k: int(v) if k in ("separation", "radius") else float(v) for k, v in new.items()}
        )
        try:
            load()
        except Exception as e:  # a bad value: say so, keep the last good settings
            print(f"error: {e}")
            settings.clear()
            settings.update(old)
            continue
        print("finding spots again; the viewer is refetching", flush=True)


if __name__ == "__main__":
    main()
