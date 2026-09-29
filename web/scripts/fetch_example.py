"""Fetch the browser page's example: two fly brain templates, as OME-Zarr 0.5.

    uv run python web/scripts/fetch_example.py OUT

JRC2018F (Bogovic et al. 2020) and FCWB (Costa et al. 2016) come from the OME-NGFF
transformation examples, with the transform published between them. That bucket allows
no CORS and uses a draft OME-Zarr 0.6 layout, so this copies the arrays byte for byte
and writes 0.5 metadata over them: OUT/fly/JRC2018F, OUT/fly/FCWB, and OUT/example.json,
which register.html fills its form from when a link names no images. The page starts from
the images as stored, finding an affine itself, or from the published one, and says how
close a found affine came to it.
The docs workflow runs it into the published site; locally, run it into web/public/data
(ignored by git), which the build copies next to the page.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE = "https://ngff-rfc5-coordinate-transformation-examples.s3.amazonaws.com/user_stories/image_registration_3d.zarr"
FIXED, MOVING = "JRC2018F", "FCWB"
ABOUT = (
    "Example: the fly brain templates JRC2018F (Bogovic et al. 2020) as fixed and FCWB "
    "(Costa et al. 2016) as moving, from the OME-NGFF transformation examples. As stored, "
    "FCWB is about half as deep and sits elsewhere."
)
PUBLISHED_ABOUT = (  # the page's second way to start
    "the published affine: the affine part of the JRC2018F-to-FCWB transform in the same "
    "examples, where a displacement field of its own comes first"
)


def get(url: str) -> bytes | None:
    """The body at `url`, or None if there is nothing there (a chunk of fill values)."""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):  # S3 answers 403 for a missing key when listing is off
            return None
        raise


def get_json(url: str) -> dict:
    body = get(url)
    if body is None:
        raise SystemExit(f"{url}: not found")
    return json.loads(body)


def affine_between(root: dict, fixed: str, moving: str) -> list[list[float]]:
    """The affine of the published fixed-to-moving transform, which must be a field then
    an affine: moving = A(p + u(p)), the form register.html fits u in."""
    for t in root["attributes"]["ome"]["coordinateTransformations"]:
        if (t.get("input"), t.get("output")) != (fixed, moving):
            continue
        steps = t["forward"]["transformations"]
        if [s["type"] for s in steps] != ["displacements", "affine"]:
            raise SystemExit(
                f"{fixed} to {moving} is {[s['type'] for s in steps]}, not a field then an affine"
            )
        return steps[1]["affine"]
    raise SystemExit(f"no transform from {fixed} to {moving} in {SOURCE}")


def multiscales_05(name: str, group: dict) -> dict:
    """0.6.dev1 multiscales (one object with coordinate systems) as 0.5 (a list, with axes)."""
    ms = group["attributes"]["ome"]["multiscales"]
    axes = [
        {k: a[k] for k in ("name", "type", "unit") if k in a}
        for a in ms["coordinateSystems"][0]["axes"]
    ]
    datasets = []
    for d in ms["datasets"]:
        (t,) = d["coordinateTransformations"]
        steps = t["transformations"] if t["type"] == "sequence" else [t]
        datasets.append(
            {
                "path": d["path"],
                "coordinateTransformations": [
                    {k: v for k, v in s.items() if k in ("type", "scale", "translation")}
                    for s in steps
                ],
            }
        )
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{"name": name, "axes": axes, "datasets": datasets}],
            }
        },
    }


def chunk_keys(meta: dict):
    enc = meta["chunk_key_encoding"]
    sep = enc.get("configuration", {}).get("separator", "/" if enc["name"] == "default" else ".")
    grid = [
        math.ceil(s / c)
        for s, c in zip(meta["shape"], meta["chunk_grid"]["configuration"]["chunk_shape"])
    ]
    for idx in itertools.product(*map(range, grid)):
        key = sep.join(map(str, idx))
        yield "c" + sep + key if enc["name"] == "default" else key


def copy_array(url: str, out: Path) -> int:
    meta = get_json(f"{url}/zarr.json")
    for c in meta["codecs"]:  # the spec's name for it; the bytes are the same
        if c["name"] == "zstandard":
            c["name"] = "zstd"
            c["configuration"] = {
                "level": c.get("configuration", {}).get("level", 0),
                "checksum": False,
            }
    out.mkdir(parents=True, exist_ok=True)
    (out / "zarr.json").write_text(json.dumps(meta, indent=2))

    def one(key: str) -> int:
        body = get(f"{url}/{key}")
        if body is None:
            return 0
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        (out / key).write_bytes(body)
        return len(body)

    with ThreadPoolExecutor(16) as pool:
        return sum(pool.map(one, chunk_keys(meta)))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="directory to write fly/ and example.json into")
    args = ap.parse_args(argv)

    affine = affine_between(get_json(f"{SOURCE}/zarr.json"), FIXED, MOVING)
    total = 0
    for name in (FIXED, MOVING):
        group = get_json(f"{SOURCE}/{name}/zarr.json")
        meta = multiscales_05(name, group)
        dest = args.out / "fly" / name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "zarr.json").write_text(json.dumps(meta, indent=2))
        for d in meta["attributes"]["ome"]["multiscales"][0]["datasets"]:
            total += copy_array(f"{SOURCE}/{name}/{d['path']}", dest / d["path"])
    example = {
        "fixed": f"fly/{FIXED}",
        "moving": f"fly/{MOVING}",  # relative to example.json
        "published_affine": affine,  # rows [A | t], to start from or compare with
        "about": ABOUT,
        "published_about": PUBLISHED_ABOUT,
        "source": SOURCE,
    }
    (args.out / "example.json").write_text(json.dumps(example, indent=2))
    print(f"wrote {args.out}: {FIXED} and {MOVING}, {total / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
