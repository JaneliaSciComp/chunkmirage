"""Localhost demo: synthetic multiscale volume served raw and thresholded, viewable in the
hosted Neuroglancer (neuroglancer-demo.appspot.com) from this machine.

    uv run python examples/demo.py [--port 8000]
"""

import argparse
import json
import os
from urllib.parse import quote

import numpy as np
import tensorstore as ts
import uvicorn

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.ops import Gaussian, Threshold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "demo-data.zarr", "blobs")
VOXEL_NM = 8.0


def make_data(shape=(256, 256, 256), n_blobs=60, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape, dtype=np.float32)
    vol = np.zeros(shape, np.float32)
    for _ in range(n_blobs):
        c = rng.uniform(0, shape, 3)
        r = rng.uniform(8, 30)
        amp = rng.uniform(80, 200)
        d2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        vol += amp * np.exp(-d2 / (2 * r**2))
    vol += rng.normal(0, 12, shape)
    return np.clip(vol, 0, 255).astype(np.uint8)


def write_multiscale(vol):
    if os.path.exists(os.path.join(DATA, "s0", ".zarray")):
        return
    os.makedirs(DATA, exist_ok=True)
    levels = [vol, vol[::2, ::2, ::2], vol[::4, ::4, ::4]]
    for i, lvl in enumerate(levels):
        arr = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": os.path.join(DATA, f"s{i}")},
                "metadata": {
                    "chunks": [64, 64, 64],
                    "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3},
                },
            },
            create=True,
            delete_existing=True,
            dtype=ts.uint8,
            shape=lvl.shape,
        ).result()
        arr[...] = lvl
    with open(os.path.join(DATA, ".zgroup"), "w") as f:
        json.dump({"zarr_format": 2}, f)
    with open(os.path.join(DATA, ".zattrs"), "w") as f:
        json.dump(
            {
                "multiscales": [
                    {
                        "version": "0.4",
                        "axes": [{"name": a, "type": "space", "unit": "nanometer"} for a in "zyx"],
                        "datasets": [
                            {
                                "path": f"s{i}",
                                "coordinateTransformations": [
                                    {"type": "scale", "scale": [VOXEL_NM * 2**i] * 3}
                                ],
                            }
                            for i in range(len(levels))
                        ],
                    }
                ]
            },
            f,
        )


def neuroglancer_link(base, raw, thr):
    s = VOXEL_NM * 1e-9
    state = {
        "dimensions": {"x": [s, "m"], "y": [s, "m"], "z": [s, "m"]},
        "position": [128, 128, 128],
        "crossSectionScale": 1.5,
        "projectionScale": 512,
        "layers": [
            {"type": "image", "source": f"zarr3://{base}/raw/@{raw.digest()}/zarr3", "name": "raw"},
            {
                "type": "segmentation",
                "source": f"precomputed://{base}/thresh/@{thr.digest()}/precomputed",
                "name": "threshold",
                "segments": ["1"],
                "selectedAlpha": 0.35,
                "notSelectedAlpha": 0,
            },
        ],
        "layout": "4panel",
    }
    return "https://neuroglancer-demo.appspot.com/#!" + quote(
        json.dumps(state, separators=(",", ":")), safe=""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--host", default="0.0.0.0", help="bind address; 0.0.0.0 makes it reachable on the network"
    )
    ap.add_argument("--low", type=float, default=120)
    ap.add_argument("--sigma", type=float, default=1.0)
    args = ap.parse_args()

    write_multiscale(make_data())
    src = open_source(DATA)
    raw = Pipeline(src, [])
    thr = Pipeline(src, [Gaussian(sigma=args.sigma), Threshold(low=args.low)], cache=raw.cache)
    app = create_app({"raw": raw, "thresh": thr})
    from chunkmirage.netutil import public_host_for

    base = f"http://{public_host_for(args.host)}:{args.port}"
    print("\nOpen in Neuroglancer:\n" + neuroglancer_link(base, raw, thr))
    print("\nChange the threshold live (then reload the layer with the new URL from the response):")
    print(f"  curl -X PUT {base}/api/datasets/thresh -H 'content-type: application/json' \\")
    print(
        f'       -d \'{{"source": "{DATA}", "ops": [{{"op": "gaussian", "sigma": 1}}, {{"op": "threshold", "low": 160}}]}}\''
    )
    print(f"\nIndex / cache stats: {base}/  and  {base}/api/cache\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
