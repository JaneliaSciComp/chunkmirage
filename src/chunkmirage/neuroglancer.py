"""Build Neuroglancer viewer states and links for served datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.parse import quote

from chunkmirage.core import ArrayInfo
from chunkmirage.pipeline import Pipeline

_UNIT_TO_M = {
    "nm": 1e-9,
    "nanometer": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "micrometer": 1e-6,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "m": 1.0,
    "meter": 1.0,
    "angstrom": 1e-10,
    "": 1e-9,
}
_TIME_TO_S = {
    "s": 1.0,
    "second": 1.0,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "us": 1e-6,
    "microsecond": 1e-6,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
}
_SPATIAL = ("z", "y", "x")
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


def dimensions(info: ArrayInfo) -> dict[str, list]:
    """Each axis as Neuroglancer's zarr/OME reader names and scales it, ``{name: [scale,
    unit]}`` in axis order: spatial axes in metres (unitless ones taken as nm), time in
    seconds, the channel axis as the layer-local ``c'``, anything else unitless."""
    dims: dict[str, list] = {}
    for ax, vs, unit in zip(info.axes, info.voxel_size, info.units):
        if ax == "c":
            dims["c'"] = [1.0, ""]
        elif ax in _SPATIAL:
            dims[ax] = [vs * _UNIT_TO_M.get(unit, 1e-9), "m"]
        elif unit in _TIME_TO_S:
            dims[ax] = [vs * _TIME_TO_S[unit], "s"]
        else:
            dims[ax] = [vs, ""]
    return dims


def global_dimensions(info: ArrayInfo) -> tuple[dict[str, list], list[float], list[str]]:
    """Viewer dimensions, position and display dimensions for a dataset: the spatial
    axes, then the others (time last) except the layer-local channel; centred in space and
    at the first index of other axes, with the spatial axes displayed (x, y, z)."""
    dims = {n: v for n, v in dimensions(info).items() if n in _SPATIAL}
    dims |= {n: v for n, v in dimensions(info).items() if n not in dims and not n.endswith("'")}
    shape = dict(zip(info.axes, info.shape))
    position = [shape[n] / 2 if n in _SPATIAL else 0.5 for n in dims]
    return dims, position, [a for a in _SPATIAL[::-1] if a in dims]


def viewer_state(pipelines: Mapping[str, Pipeline], sources: Mapping[str, str]) -> dict:
    """One viewer state containing a layer per dataset. Dimensions come from the first."""
    names = list(pipelines)
    if not names:
        return {"layers": [], "layout": "4panel"}
    dims, position, display = global_dimensions(pipelines[names[0]].info(0))
    return {
        "dimensions": dims,
        "displayDimensions": display,
        "position": position,
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
