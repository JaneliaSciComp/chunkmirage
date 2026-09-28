"""Show one fly-brain template registered onto another, with no warped copy on disk.

The OME-Zarr RFC-5 example ``image_registration_3d`` holds two Drosophila brain templates,
JRC2018F and FCWB, and the transformation between them: a displacement field followed by
an affine, stored as a bijection with its inverse. This script downloads it (~49 MB),
rewrites it as a final OME-Zarr 0.6 scene (the published copy is a 0.6.dev1 draft whose
codec name tensorstore rejects), and serves three layers: JRC2018F, FCWB as stored (in
its own coordinates), and FCWB resampled onto the JRC2018F grid:

    uv run python examples/fly_brain_registration.py [OUT_DIR]

Neuroglancer cannot open the scene (it has no displacement-field support), but it shows
the printed link registered anyway: every chunk is resampled on the server. The
registered layer alone can be served by the CLI:

    chunkmirage serve "scene://OUT_DIR/fly_brains.zarr?image=FCWB&target=JRC2018F"
"""

from __future__ import annotations

import http.client
import json
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tensorstore as ts

BUCKET = "https://ngff-rfc5-coordinate-transformation-examples.s3.amazonaws.com"
PREFIX = "user_stories/image_registration_3d.zarr/"
S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def get(url: str, attempts: int = 4) -> bytes:
    for i in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except (OSError, http.client.HTTPException):
            if i == attempts - 1:
                raise
    raise AssertionError("unreachable")


def mirror(dest: Path) -> None:
    """Download every object under PREFIX (skipping ones already present)."""
    token = ""
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}" + (
            f"&continuation-token={token}" if token else ""
        )
        root = ET.fromstring(get(url))
        for item in root.iter(f"{S3}Contents"):
            key = item.find(f"{S3}Key").text
            out = dest / key[len(PREFIX) :]
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                part = out.with_name(out.name + ".part")
                part.write_bytes(get(f"{BUCKET}/{key}"))
                part.rename(out)
        token = root.findtext(f"{S3}NextContinuationToken") or ""
        if not token:
            return


def patch_codecs(tree: Path) -> None:
    """The draft files name the zstd codec "zstandard"; zarr v3 calls it "zstd"."""
    for meta in tree.rglob("zarr.json"):
        body = json.loads(meta.read_text())
        for codec in body.get("codecs", []):
            if codec.get("name") == "zstandard":
                codec["name"] = "zstd"
        meta.write_text(json.dumps(body, indent=2))


def group(path: Path, ome: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    body = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": {"version": "0.6", **ome}},
    }
    (path / "zarr.json").write_text(json.dumps(body, indent=2))


def image(src: Path, out: Path) -> None:
    """Copy an image's arrays; rewrite its draft multiscales metadata in the 0.6 form."""
    ms = json.loads((src / "zarr.json").read_text())["attributes"]["ome"]["multiscales"]
    cs = ms["coordinateSystems"][0]
    datasets = []
    for ds in ms["datasets"]:
        shutil.copytree(src / ds["path"], out / ds["path"], dirs_exist_ok=True)
        cts = []
        for ct in ds["coordinateTransformations"]:
            ct = {k: v for k, v in ct.items() if k not in ("input", "output")}
            cts.append({**ct, "input": {"path": ds["path"]}, "output": {"name": cs["name"]}})
        datasets.append({"path": ds["path"], "coordinateTransformations": cts})
    group(
        out, {"multiscales": [{"name": out.name, "coordinateSystems": [cs], "datasets": datasets}]}
    )


def field(src: Path, out: Path) -> None:
    """Draft field (bare array, vectors last) -> 0.6 multiscale field (vectors first)."""
    meta = json.loads((src / "zarr.json").read_text())["attributes"]["ome"]
    data = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(src)}}).result()
    vectors = np.moveaxis(data.read().result(), -1, 0)
    arr = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(out / "s0")},
            "metadata": {
                "shape": list(vectors.shape),
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [3, 64, 64, 64]},
                },
                "codecs": [{"name": "bytes"}, {"name": "zstd"}],
            },
        },
        create=True,
        delete_existing=True,
    ).result()
    arr.write(vectors).result()
    axes = meta["coordinateSystems"][0]["axes"]
    axes = [axes[-1]] + axes[:-1]  # vector axis first, then z, y, x
    scale = meta["coordinateTransformations"][0]["scale"]
    scale = [scale[-1]] + scale[:-1]
    ct = {"type": "scale", "scale": scale, "input": {"path": "s0"}, "output": {"name": "physical"}}
    group(
        out,
        {
            "multiscales": [
                {
                    "coordinateSystems": [{"name": "physical", "axes": axes}],
                    "datasets": [{"path": "s0", "coordinateTransformations": [ct]}],
                }
            ]
        },
    )


def scene(src: Path, out: Path) -> None:
    """The draft kept the bijection at the top level with bare names; 0.6 puts it under
    ``scene`` with {path, name} references to each image's coordinate system."""
    bij = json.loads((src / "zarr.json").read_text())["attributes"]["ome"][
        "coordinateTransformations"
    ][0]
    for part in (bij["forward"], bij["inverse"]):
        for t in part["transformations"]:
            t.pop("input", None)
            t.pop("name", None)
    bij["input"] = {"path": "JRC2018F", "name": "JRC2018F"}
    bij["output"] = {"path": "FCWB", "name": "FCWB"}
    group(out, {"scene": {"coordinateTransformations": [bij]}})
    group(out / "coordinateTransformations", {})


def build(workdir: Path) -> Path:
    raw, out = workdir / "draft.zarr", workdir / "fly_brains.zarr"
    if (out / "zarr.json").exists():
        return out
    print(f"downloading the RFC-5 example to {raw} ...")
    mirror(raw)
    patch_codecs(raw)
    for name in ("JRC2018F", "FCWB"):
        image(raw / name, out / name)
    for name in ("dfield", "invdfield"):
        field(raw / "coordinateTransformations" / name, out / "coordinateTransformations" / name)
    scene(raw, out)
    print(f"wrote the OME-Zarr 0.6 scene {out}")
    return out


# Fixed template green, moving template magenta, blended additively: aligned structures
# turn white, misaligned ones stay green or magenta.
COLOURS = {"JRC2018F": "#00ff00", "FCWB_original": "#ff00ff", "FCWB_registered": "#ff00ff"}
SHADER = """#uicontrol vec3 colour color(default="{colour}")
#uicontrol invlerp normalized
void main() {{ emitRGB(colour * normalized()); }}
"""


def demo(out: Path, public_url: str) -> tuple[dict, str]:
    """Three datasets and a Neuroglancer link showing them:

    * ``JRC2018F``: the fixed template, straight from disk;
    * ``FCWB_original``: the moving template, straight from disk, in its own coordinates;
    * ``FCWB_registered``: the moving template warped onto JRC2018F's grid, on the fly.
    """
    from urllib.parse import quote

    from chunkmirage import Pipeline
    from chunkmirage.neuroglancer import DEFAULT_VIEWER, source_url, viewer_state

    pipes = {
        "JRC2018F": Pipeline.from_spec({"source": str(out / "JRC2018F")}),
        "FCWB_original": Pipeline.from_spec({"source": str(out / "FCWB")}),
        "FCWB_registered": Pipeline.from_spec(
            {"source": f"scene://{out}?image=FCWB&target=JRC2018F"}
        ),
    }
    urls = {n: source_url(public_url, n, "zarr3", "zarr3", p.digest()) for n, p in pipes.items()}
    state = viewer_state(pipes, urls)
    for layer in state["layers"]:
        layer["shader"] = SHADER.format(colour=COLOURS[layer["name"]])
        layer["blend"] = "additive"
        layer["visible"] = layer["name"] != "FCWB_original"  # toggle it on to compare
    state["layout"] = "xy"
    return pipes, f"{DEFAULT_VIEWER}/#!{quote(json.dumps(state, separators=(',', ':')), safe='')}"


def main() -> None:
    import uvicorn

    from chunkmirage import create_app

    out = build(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
    from chunkmirage.netutil import free_port

    port = free_port("127.0.0.1", 8000)  # 8000, or the next free port if it is taken
    pipes, link = demo(out, f"http://localhost:{port}")
    print(
        "\nLayers: JRC2018F (green, fixed), FCWB_registered (magenta, warped on the fly) and"
        "\nFCWB_original (magenta, hidden: toggle it and FCWB_registered to compare)."
    )
    print("neuroglancer:", link)
    uvicorn.run(create_app(pipes), port=port)


if __name__ == "__main__":
    main()
