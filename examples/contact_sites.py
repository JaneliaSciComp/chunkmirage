"""Contact sites between two organelles, computed where you look: the published
mitochondria and ER predictions of OpenOrganelle's jrc_hela-2 cell, stacked as two channels,
and every chunk on screen answered with the voxels within a few nanometres of both, labelled
and size-filtered. Nothing is precomputed or written: 122 gigavoxels of cell, and only the
chunks you view cost anything.

    uv run python examples/contact_sites.py [--radius 3] [--min-size 50] [--chunk 16,128,128]

A python Neuroglancer viewer opens with the EM (read by the browser straight from the public
bucket), the two predictions and the contact sites. While it is open, type settings at the
prompt (``radius=4 min_size=100``) to recompute: the viewer refetches what is on screen and
keeps its camera; the two predictions were read into chunkmirage's cache by the first pass,
so the second costs only the contact computation. The equivalent command line is printed.

The predictions are a network's, with its mistakes: where the EM has a darker band (one
crosses the cell near y 300-425 at z 2372) the ER network fires on half the voxels, and the
contacts it finds there are not real. The default view is clear of it.
"""

from __future__ import annotations

import argparse
import threading

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/"
EM = BUCKET + "jrc_hela-2.zarr/recon-1/em/fibsem-uint8"
# The N5 predictions are stored upside down in y relative to the EM (their metadata does not
# say so): flip:// mirrors them back, every level in place, nothing copied.
MITO = "flip://" + BUCKET + "jrc_hela-2.n5/labels/mito_pred?axes=y"
ER = "flip://" + BUCKET + "jrc_hela-2.n5/labels/er_pred?axes=y"
EDITABLE = ("radius", "min_size", "a_low", "b_low")

TINT_SHADER = """#uicontrol invlerp p(range=[64, 255])
#uicontrol float alpha slider(min=0, max=1, default={alpha})
void main() {{
  float v = p();
  emitRGBA(vec4({r}, {g}, {b}, v * alpha));
}}
"""


def spec(source: str, settings: dict) -> dict:
    """The contact-sites pipeline: the stack of two predictions, `contacts`, then `label`."""
    contacts = {k: v for k, v in settings.items() if k in ("radius", "a_low", "b_low")}
    label = {k: v for k, v in settings.items() if k in ("min_size",)}
    return {"source": source, "ops": [{"op": "contacts", **contacts}, {"op": "label", **label}]}


def command(source: str, settings: dict, chunk: str) -> str:
    ops = spec(source, settings)["ops"]
    parts = [
        f"--op {o['op']}:" + ",".join(f"{k}={v}" for k, v in o.items() if k != "op") for o in ops
    ]
    return f"chunkmirage serve '{source}' {' '.join(parts)} --chunk {chunk} --python-viewer"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", default=MITO, help="first structure: a probability map or segmentation")
    ap.add_argument("--b", default=ER, help="second structure, on the same grid")
    ap.add_argument("--em", default=EM, help="image the browser shows underneath (zarr URL)")
    ap.add_argument("--radius", type=float, default=3, help="reach in voxels (12 nm at 4 nm)")
    ap.add_argument("--min-size", type=int, default=50, help="drop sites smaller than this")
    ap.add_argument("--a-low", type=float, default=128, help="threshold on the first channel")
    ap.add_argument("--b-low", type=float, default=128, help="threshold on the second channel")
    ap.add_argument(
        "--chunk",
        default="16,128,128",
        help="output chunks, z,y,x: thin in z means less to compute for the x-y view",
    )
    ap.add_argument(
        "--position",
        default="2156,596,3028",
        help="where the view opens, z,y,x in full-resolution voxels (default: mitochondria "
        "wrapped by ER, among the places they touch most)",
    )
    ap.add_argument(
        "--zoom", type=float, default=1.0, help="full-resolution voxels per screen pixel"
    )
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
    import uvicorn

    from chunkmirage.cache import LRUCache
    from chunkmirage.netutil import free_port, is_loopback, public_host_for, serving_address
    from chunkmirage.server import DatasetRegistry, create_app
    from chunkmirage.viewer import Viewer

    source = f"stack://{args.a}|{args.b}"
    settings = {
        "radius": args.radius,
        "min_size": args.min_size,
        "a_low": args.a_low,
        "b_low": args.b_low,
    }
    chunk = [int(c) for c in args.chunk.split(",")]
    port = args.port or free_port(args.host, 8000)
    viewer_port = args.viewer_port or free_port(args.host, 8001, avoid={port})

    registry = DatasetRegistry(
        LRUCache(int(args.cache_gb * 2**30)), source_cache_bytes=int(args.source_cache_gb * 2**30)
    )

    def load() -> None:
        registry.add("contacts", {**spec(source, settings), "chunk_shape": chunk})

    registry.add("mito", {"source": args.a, "ops": [], "chunk_shape": chunk})
    registry.add("er", {"source": args.b, "ops": [], "chunk_shape": chunk})
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
    with viewer.viewer.txn() as s:
        served = {layer.name: layer.layer for layer in s.layers}
        s.layers.clear()  # layers draw in order: the EM underneath, the contact sites on top
        s.layers["em"] = neuroglancer.ImageLayer(source=f"zarr://{args.em}")
        for name in ("mito", "er", "contacts"):
            s.layers[name] = served[name]
        s.layers["mito"].shader = TINT_SHADER.format(r=0.2, g=0.9, b=0.3, alpha=0.35)
        s.layers["er"].shader = TINT_SHADER.format(r=0.9, g=0.3, b=0.9, alpha=0.35)
        s.layers["contacts"].selected_alpha = 0.9
        s.position = [float(v) + 0.5 for v in args.position.split(",")]
        s.cross_section_scale = args.zoom
        s.layout = "xy"  # one plane: the 3-D panels would ask for every chunk in the volume

    print(f"\nviewer:  {viewer.url}", flush=True)
    print(f"public:  {viewer.hosted_link()}")
    print(f"chunks:  {public}")
    print(f"same as: {command(source, settings, args.chunk)}")
    if https:
        print(
            "https:   self-signed certificate: each browser (yours and anyone you send a link "
            f"to) must trust it once: open {public}/ and accept the warning"
        )
    print(
        "Contact sites appear as the chunks on screen are computed; pan and they follow. Type "
        f"settings to recompute ({', '.join(EDITABLE)}), e.g. `radius=4 min_size=100`.\n",
        flush=True,
    )
    threading.Thread(target=prompt, args=(settings, load), daemon=True).start()
    app = create_app(registry, threads=args.threads)
    uvicorn.run(app, host=args.host, port=port, log_level="warning", **ssl)


def prompt(settings: dict, load) -> None:
    """Read ``key=value`` settings from the terminal and recompute with them."""
    while True:
        try:
            line = input("contacts> ").strip()
        except EOFError:  # no terminal: nothing to read
            return
        if not line:
            continue
        try:
            new = dict(tok.split("=", 1) for tok in line.split())
        except ValueError:
            print("expected key=value pairs, e.g. radius=4 min_size=100")
            continue
        if bad := set(new) - set(EDITABLE):
            print(f"unknown: {sorted(bad)}; editable: {', '.join(EDITABLE)}")
            continue
        old = dict(settings)
        settings.update({k: float(v) if k != "min_size" else int(v) for k, v in new.items()})
        try:
            load()
        except Exception as e:  # a bad value: say so, keep the last good settings
            print(f"error: {e}")
            settings.clear()
            settings.update(old)
            continue
        print("recomputing; the viewer is refetching", flush=True)


if __name__ == "__main__":
    main()
