"""OME-NGFF multiscales metadata shared by zarr v2 (0.4) and zarr v3 (0.5) frontends."""

from __future__ import annotations

from chunkmirage.pipeline import Pipeline

_AXIS_TYPES = {
    "x": "space",
    "y": "space",
    "z": "space",
    "t": "time",
    "time": "time",
    "c": "channel",
}
_TIME_UNITS = {"s", "second", "ms", "millisecond", "min", "minute", "h", "hour", "d", "day"}


def multiscales(pipeline: Pipeline, version: str) -> dict:
    info0 = pipeline.info(0)
    axes = []
    for name, unit in zip(info0.axes, info0.units):
        kind = "time" if unit in _TIME_UNITS else _AXIS_TYPES.get(name, "space")
        ax = {"name": name, "type": kind}
        if not unit and name in ("z", "y", "x"):
            # unitless z, y, x are nanometres, as the N5 and precomputed frontends and the
            # viewer helpers take them; left out, Neuroglancer would read them as metres
            unit = "nm"
        if unit:
            ax["unit"] = {
                "nm": "nanometer",
                "um": "micrometer",
                "µm": "micrometer",
                "m": "meter",
                "km": "kilometer",
                "s": "second",
            }.get(unit, unit)
        axes.append(ax)
    datasets = []
    for lvl in range(pipeline.num_levels):
        info = pipeline.info(lvl)
        ct = [{"type": "scale", "scale": [float(v) for v in info.voxel_size]}]
        if any(t != 0 for t in info.translation):
            ct.append({"type": "translation", "translation": [float(t) for t in info.translation]})
        datasets.append({"path": f"s{lvl}", "coordinateTransformations": ct})
    return {
        "version": version,
        "name": pipeline.source.name or "chunkmirage",
        "axes": axes,
        "datasets": datasets,
        "type": "chunkmirage",
        "metadata": {"description": "virtual dataset served by chunkmirage"},
    }
