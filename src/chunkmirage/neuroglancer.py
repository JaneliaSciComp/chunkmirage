"""Build Neuroglancer viewer links for a served dataset."""

from __future__ import annotations

import json
from urllib.parse import quote

from chunkmirage.pipeline import Pipeline

_UNIT_TO_M = {"nm": 1e-9, "um": 1e-6, "µm": 1e-6, "mm": 1e-3, "m": 1.0, "": 1e-9}


def source_url(public_url: str, name: str, fmt: str, scheme: str, digest: str | None = None) -> str:
    base = public_url.rstrip("/")
    path = f"{base}/{name}/@{digest}/{fmt}" if digest else f"{base}/{name}/{fmt}"
    return f"{scheme}://{path}"


def viewer_state(pipeline: Pipeline, name: str, src_url: str) -> dict:
    info = pipeline.info(0)
    dims = {}
    for ax, vs, unit in zip(info.axes, info.voxel_size, info.units):
        if ax == "c":
            continue
        dims[ax] = [vs * _UNIT_TO_M.get(unit, 1e-9), "m"]
    is_seg = info.dtype.kind == "u" and info.dtype.itemsize >= 4
    layer = {"type": "segmentation" if is_seg else "image", "source": src_url, "name": name}
    if info.dtype.name == "uint8" and pipeline.ops and pipeline.ops[-1].name == "threshold":
        layer["shader"] = "void main() { emitGrayscale(float(getDataValue(0).value)); }"
    state = {
        "dimensions": dims,
        "position": [s / 2 for ax, s in zip(info.axes, info.shape) if ax != "c"],
        "layers": [layer],
        "layout": "4panel",
    }
    return state


def viewer_link(
    pipeline: Pipeline,
    name: str,
    src_url: str,
    viewer: str = "https://neuroglancer-demo.appspot.com",
) -> str:
    state = viewer_state(pipeline, name, src_url)
    return f"{viewer.rstrip('/')}/#!{quote(json.dumps(state, separators=(',', ':')), safe='')}"
