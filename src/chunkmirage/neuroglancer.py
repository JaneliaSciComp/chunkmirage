"""Build Neuroglancer viewer states and links for served datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.parse import quote

from chunkmirage.pipeline import Pipeline

_UNIT_TO_M = {
    "nm": 1e-9,
    "nanometer": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "micrometer": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
    "": 1e-9,
}
DEFAULT_VIEWER = "https://neuroglancer-demo.appspot.com"


def source_url(public_url: str, name: str, fmt: str, scheme: str, digest: str | None = None) -> str:
    base = public_url.rstrip("/")
    path = f"{base}/{name}/@{digest}/{fmt}" if digest else f"{base}/{name}/{fmt}"
    return f"{scheme}://{path}"


def is_segmentation(pipeline: Pipeline) -> bool:
    info = pipeline.info(0)
    if info.dtype.kind == "u" and info.dtype.itemsize >= 4:
        return True
    return bool(pipeline.ops) and pipeline.ops[-1].name == "threshold"


def layer_for(name: str, pipeline: Pipeline, src_url: str) -> dict:
    if is_segmentation(pipeline):
        layer = {"type": "segmentation", "source": src_url, "name": name}
        if pipeline.ops and pipeline.ops[-1].name == "threshold":
            value = getattr(pipeline.ops[-1], "value", 1)
            layer.update({"segments": [str(value)], "selectedAlpha": 0.4, "notSelectedAlpha": 0})
        return layer
    return {"type": "image", "source": src_url, "name": name}


def viewer_state(pipelines: Mapping[str, Pipeline], sources: Mapping[str, str]) -> dict:
    """One viewer state containing a layer per dataset. Dimensions come from the first."""
    names = list(pipelines)
    if not names:
        return {"layers": [], "layout": "4panel"}
    info = pipelines[names[0]].info(0)
    dims = {}
    for ax, vs, unit in zip(info.axes, info.voxel_size, info.units):
        if ax == "c":
            continue
        dims[ax] = [vs * _UNIT_TO_M.get(unit, 1e-9), "m"]
    return {
        "dimensions": dims,
        "position": [s / 2 for ax, s in zip(info.axes, info.shape) if ax != "c"],
        "layers": [layer_for(n, pipelines[n], sources[n]) for n in names],
        "layout": "4panel",
    }


def viewer_link(
    pipelines: Mapping[str, Pipeline],
    sources: Mapping[str, str],
    viewer: str = DEFAULT_VIEWER,
) -> str:
    state = viewer_state(pipelines, sources)
    return f"{viewer.rstrip('/')}/#!{quote(json.dumps(state, separators=(',', ':')), safe='')}"
