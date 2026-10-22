"""OME-NGFF multiscales metadata for the zarr v2 (0.4) and zarr v3 (0.5) frontends, written
by worldarray (``wa.ome_multiscales``), so what is served and what its reader reads back
follow one implementation."""

from __future__ import annotations

import worldarray as wa

from chunkmirage.core import spatial_axes
from chunkmirage.pipeline import Pipeline


def group_attrs(pipeline: Pipeline, version: str) -> dict:
    """The group's attributes: ``{"multiscales": [...]}`` for 0.4, ``{"ome": ...}`` for 0.5.
    An axis is time when named so or in a time unit (an xarray ``time`` in seconds),
    channel when named ``c``, space when spatial (named z, y or x, else the last three);
    unitless z, y and x are nanometres, as the N5 and precomputed frontends and the viewer
    helpers take them (left out, Neuroglancer would read them as metres)."""
    info0 = pipeline.info(0)
    spatial = set(spatial_axes(info0.axes))
    types = [
        "time" if a in ("t", "time") or wa.units.kind(u) == "time"
        else "channel" if a == "c"
        else "space" if i in spatial
        else None
        for i, (a, u) in enumerate(zip(info0.axes, info0.units))
    ]
    units = [u or ("nanometer" if a in ("z", "y", "x") else "") for a, u in zip(info0.axes, info0.units)]
    levels = [
        (f"s{lvl}", pipeline.info(lvl).voxel_size, pipeline.info(lvl).translation)
        for lvl in range(pipeline.num_levels)
    ]
    name = pipeline.source.name or "chunkmirage"
    return wa.ome_multiscales(info0.axes, units, levels, version, axis_types=types, name=name)
