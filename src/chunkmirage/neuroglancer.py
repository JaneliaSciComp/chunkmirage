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
    "km": 1e3,
    "kilometer": 1e3,
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
    "d": 86400.0,
    "day": 86400.0,
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


def compare_shader(fixed: tuple[float, float], moving: tuple[float, float]) -> str:
    """Two channels compared on the viewer's GPU: the first magenta, the second green, white
    where they agree (``mode`` 0); a checkerboard, a fade or their difference (1 to 3)."""
    return f"""#uicontrol int mode slider(min=0, max=3, default=0)
#uicontrol float fade slider(min=0, max=1, default=0.5)
#uicontrol float squares slider(min=4, max=256, default=48)
#uicontrol float gain slider(min=1, max=10, default=3)
#uicontrol invlerp fixed(range=[{fixed[0]:.6g}, {fixed[1]:.6g}], channel=0)
#uicontrol invlerp moving(range=[{moving[0]:.6g}, {moving[1]:.6g}], channel=1)
void main() {{
  float f = fixed(); float m = moving();
  if (mode == 0) {{ emitRGB(vec3(f, m, f)); }}
  else if (mode == 1) {{
    float odd = mod(floor(gl_FragCoord.x / squares) + floor(gl_FragCoord.y / squares), 2.0);
    emitGrayscale(odd > 0.5 ? m : f);
  }}
  else if (mode == 2) {{ emitGrayscale(mix(f, m, fade)); }}
  else {{ float d = gain * (m - f); emitRGB(vec3(max(d, 0.0), 0.0, max(-d, 0.0))); }}
}}
"""


def field_shader(scale: float) -> str:
    """A displacement field's three components: how far each point moved as a heat map up
    to ``scale`` (physical units), or with ``direction`` its direction as colour."""
    return f"""#uicontrol float scale slider(min={scale / 20:.3g}, max={scale * 4:.3g}, default={scale:.3g})
#uicontrol bool direction checkbox(default=false)
void main() {{
  vec3 d = vec3(getDataValue(0), getDataValue(1), getDataValue(2));
  if (direction) {{ emitRGB(clamp(vec3(0.5) + 0.5 * d / scale, 0.0, 1.0)); }}
  else {{ emitRGB(colormapJet(clamp(length(d) / scale, 0.0, 1.0))); }}
}}
"""


def layer_for(name: str, pipeline: Pipeline, src_url: str) -> dict:
    if is_segmentation(pipeline):
        layer = {"type": "segmentation", "source": src_url, "name": name}
        if pipeline.ops and pipeline.ops[-1].name == "threshold":
            value = getattr(pipeline.ops[-1], "value", 1)
            layer.update({"segments": [str(value)], "selectedAlpha": 0.4, "notSelectedAlpha": 0})
        return layer
    shader = getattr(pipeline.source, "shader", None)
    if shader:  # the channel axis is the shader's: c' becomes c^
        dims = {("c^" if k == "c'" else k): v for k, v in dimensions(pipeline.info(0)).items()}
        source = {"url": src_url, "transform": {"outputDimensions": dims}}
        return {"type": "image", "source": source, "name": name, "shader": shader}
    return {"type": "image", "source": src_url, "name": name}


def dimensions(info: ArrayInfo) -> dict[str, list]:
    """Each axis as Neuroglancer's zarr/OME reader names and scales it, ``{name: [scale,
    unit]}`` in axis order: time (any axis in a time unit) in seconds, spatial axes in
    metres (unitless ones taken as nm, as the frontends serve them), the channel axis as the
    layer-local ``c'``, anything else unitless."""
    dims: dict[str, list] = {}
    for ax, vs, unit in zip(info.axes, info.voxel_size, info.units):
        if ax == "c":
            dims["c'"] = [1.0, ""]
        elif unit in _TIME_TO_S:
            dims[ax] = [vs * _TIME_TO_S[unit], "s"]
        elif ax in _SPATIAL:
            dims[ax] = [vs * _UNIT_TO_M.get(unit, 1e-9), "m"]
        else:
            dims[ax] = [vs, ""]
    return dims


def cross_section_scale(dims: Mapping[str, list], axis: str, voxels_per_pixel: float) -> float:
    """Neuroglancer's ``crossSectionScale`` for ``voxels_per_pixel`` voxels of ``axis`` per
    screen pixel. Neuroglancer counts it in the smallest scale among the dimensions,
    whatever their units, so with a time axis in seconds beside space in metres a plain
    ``1`` is not one voxel per pixel."""
    smallest = min(float(scale) for scale, _ in dims.values())
    return voxels_per_pixel * float(dims[axis][0]) / smallest


def global_dimensions(info: ArrayInfo) -> tuple[dict[str, list], list[float], list[str]]:
    """Viewer dimensions, position and display dimensions for a dataset: the spatial
    axes, then the others (time last) except the layer-local channel; centred in space and
    at the first index of other axes, with the spatial axes displayed (x, y, z). Data
    without z, y, x axes (time, lat, lon) shows its last three in their place; images in y, x
    over another axis (time) scroll through that one as z."""
    spatial = [a for a in info.axes if a in _SPATIAL]
    others = [a for a in info.axes if a not in _SPATIAL and a != "c"]
    if not spatial:
        spatial = [a for a in info.axes if a != "c"][-3:]
    elif len(spatial) == 2 and others:
        spatial = [a for a in info.axes if a in spatial or a == others[-1]]
    dims = {n: v for n, v in dimensions(info).items() if n in spatial}
    dims |= {n: v for n, v in dimensions(info).items() if n not in dims and not n.endswith("'")}
    shape = dict(zip(info.axes, info.shape))
    position = [shape[n] / 2 if n in spatial else 0.5 for n in dims]
    return dims, position, spatial[::-1]


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
