"""Meshes computed when a viewer asks for them, in Neuroglancer's precomputed (legacy) mesh
format: one segment, ``1``, whose fragments are the chunks of one level of a pipeline, each
meshed when it is fetched (the format names fragments in a manifest and leaves them to be
fetched one by one, so nothing is computed up front; its multi-resolution sibling needs
every fragment's byte offsets first).

    <dataset>/mesh/info        {"@type": "neuroglancer_legacy_mesh"}
    <dataset>/mesh/1:0         {"fragments": ["1:0:0_0_0", ...]}, every chunk of the level
    <dataset>/mesh/1:0:i_j_k   chunk (i, j, k) meshed: uint32 vertex count, float32 x, y, z
                               per vertex (nanometres), uint32 triangle corners

Two kinds. ``surface``: the boundary of the voxels at or above ``threshold``, by marching
cubes (scikit-image), from the chunk and one voxel more on its high sides so neighbouring
fragments meet, closed at the array's edges. ``terrain``: an elevation model (``y, x``, or
``z, y, x`` with one z) as a surface of two triangles per cell, the elevation times
``exaggeration`` as height; cells with a NaN corner are left out. Imports nothing but
numpy, pydantic and scikit-image, so the browser engine's workers run it too.
"""

from __future__ import annotations

import struct
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from chunkmirage.core import ArrayInfo, Box

_TO_NM = {"": 1.0, "nm": 1.0, "nanometer": 1.0, "um": 1e3, "micrometer": 1e3, "µm": 1e3,
          "mm": 1e6, "millimeter": 1e6, "m": 1e9, "meter": 1e9, "km": 1e12, "kilometer": 1e12}  # fmt: skip
MAX_SIDE = 512  # the default level: the finest whose longest side is at most this


class MeshSpec(BaseModel):
    """What a dataset's ``mesh`` frontend meshes."""

    kind: Literal["surface", "terrain"] = Field(
        "surface", description="surface: an isosurface of the volume; terrain: an elevation model"
    )
    level: int | None = Field(
        None,
        ge=0,
        description=f"The level meshed; default the finest whose longest side is at most {MAX_SIDE}",
    )
    threshold: float = Field(128.0, description="surface: values at or above this are inside")
    exaggeration: float = Field(1.0, gt=0, description="terrain: the elevation's scale")


def mesh_level(infos: list[ArrayInfo], spec: MeshSpec) -> int:
    if spec.level is not None:
        if spec.level >= len(infos):
            raise ValueError(f"mesh level {spec.level}: the dataset has {len(infos)} levels")
        return spec.level
    small = [i for i, info in enumerate(infos) if max(info.shape[-3:]) <= MAX_SIDE]
    return small[0] if small else len(infos) - 1


def fragment_names(info: ArrayInfo) -> list[str]:
    return ["1:0:" + "_".join(map(str, idx)) for idx in np.ndindex(*info.chunk_grid)]


def fragment_box(info: ArrayInfo, index) -> Box:
    """A fragment's voxels: its chunk and one more on each high side (clipped), so its
    surface meets its neighbours'."""
    box = info.chunk_box(index)
    return Box(box.start, tuple(min(b + 1, s) for b, s in zip(box.stop, info.shape))).clip(
        info.shape
    )


def _nm(info: ArrayInfo) -> np.ndarray:
    return np.array([v * _TO_NM.get(u, 1.0) for v, u in zip(info.voxel_size, info.units)])


def fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo) -> bytes:
    """Mesh ``block`` (the level's voxels over ``box``), vertices in nanometres."""
    scale = _nm(info)
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(info.translation, info.units)])
    if spec.kind == "terrain":
        return _terrain(spec, block, box, info, scale, origin)
    mask = np.asarray(block) >= spec.threshold
    if mask.ndim != 3:
        raise ValueError(f"a surface mesh needs a z, y, x volume, not {mask.ndim} axes")
    # closed at the array's edges: one voxel of outside beyond them
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))
    if not padded.any() or padded.all():
        return encode(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    from skimage.measure import marching_cubes

    verts, faces, _, _ = marching_cubes(padded.astype(np.float32), 0.5)
    zyx = (verts - lo + np.array(box.start)) * scale[-3:] + origin[-3:]
    return encode(zyx[:, ::-1].astype(np.float32), faces[:, ::-1].astype(np.uint32))


def _terrain(spec, block, box: Box, info: ArrayInfo, scale, origin) -> bytes:
    z = np.asarray(block, dtype=np.float64)
    z = z.reshape(z.shape[-2:])  # y, x (a z of one dropped)
    h, w = z.shape
    y0, x0 = box.start[-2:]
    ys = (np.arange(y0, y0 + h) * scale[-2] + origin[-2])[:, None]
    xs = (np.arange(x0, x0 + w) * scale[-1] + origin[-1])[None, :]
    up = _TO_NM.get(info.units[-1], 1.0) * spec.exaggeration  # elevation in the grid's unit
    verts = np.stack(np.broadcast_arrays(xs, ys, np.nan_to_num(z) * up), -1).reshape(-1, 3)
    i = np.arange(h * w).reshape(h, w)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    ok = ~(
        np.isnan(z[:-1, :-1]) | np.isnan(z[:-1, 1:]) | np.isnan(z[1:, :-1]) | np.isnan(z[1:, 1:])
    )
    faces = np.concatenate([np.stack([a, c, b], -1)[ok], np.stack([b, c, d], -1)[ok]])
    return encode(verts.astype(np.float32), faces.astype(np.uint32))


def encode(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A legacy mesh fragment: vertex count, the vertices, the triangles' corners (LE)."""
    return (
        struct.pack("<I", len(verts))
        + np.ascontiguousarray(verts, "<f4").tobytes()
        + np.ascontiguousarray(faces, "<u4").tobytes()
    )
