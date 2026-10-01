"""Meshes computed when a viewer asks for them, in Neuroglancer's precomputed (legacy) mesh
format: one segment, ``1``, whose fragments are the chunks of one level of a pipeline, each
meshed when it is fetched (the format names fragments in a manifest and leaves them to be
fetched one by one, so nothing is computed up front; its multi-resolution sibling needs
every fragment's byte offsets first).

    <dataset>/mesh/info        {"@type": "neuroglancer_legacy_mesh"}
    <dataset>/mesh/1:0         {"fragments": ["1:0:0_0_0", ...]}, every chunk of the level
    <dataset>/mesh/1:0:i_j_k   chunk (i, j, k) meshed: uint32 vertex count, float32 x, y, z
                               per vertex (nanometres), uint32 triangle corners

With ``lods`` above 1, a surface is served in Neuroglancer's multi-resolution format instead,
whose meshes get finer where the viewer zooms in:

    <dataset>/mesh/info        {"@type": "neuroglancer_multilod_draco", ...}
    <dataset>/mesh/1.index     the octree: per level of detail, the nodes and their sizes
    <dataset>/mesh/1           every node's fragment, one after another (HTTP Range requests)

Level of detail ``i`` is pyramid level ``level - lods + 1 + i``, its nodes the chunks of that
level, so the coarsest is ``level``'s chunks and each finer one splits them in eight. The
format lists every fragment's byte size before any is fetched, so each is padded to
``FRAGMENT_BYTES`` (Draco decoders read what they need and ignore the rest) and its mesh
kept under ``MAX_TRIANGLES``; nodes are listed only near the surface of the coarsest level
(``multires_nodes``), so the index stays small while the finer levels exist only where they
are fetched.

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
FRAGMENT_BYTES = 1 << 17  # every multi-resolution fragment, padded: sizes are listed up front
MAX_TRIANGLES = 100_000  # a fragment's mesh, coarsened above it: Draco takes under a byte each
BITS = 16  # multi-resolution vertex positions: integers in [0, 2**BITS) across a node
BAND = 2  # nodes are listed within this many coarsest-level voxels of its surface
MAX_NODES = 2_000_000  # an index's nodes (16 bytes each): fewer levels of detail past it


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
    lods: int = Field(
        1,
        ge=1,
        le=8,
        description="surface: levels of detail. 1 is one mesh of `level`; more serve "
        "Neuroglancer's multi-resolution format, `level` the coarsest and each further one a "
        "pyramid level finer, meshed where the viewer zooms in",
    )


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


# ------------------------------------------------------------------ multi-resolution
def lod_levels(infos: list[ArrayInfo], spec: MeshSpec) -> list[int]:
    """The pyramid level of each level of detail, finest first."""
    if spec.kind != "surface" and spec.lods > 1:
        raise ValueError("levels of detail are for surface meshes")
    top = mesh_level(infos, spec)
    if top - spec.lods + 1 < 0:
        raise ValueError(f"lods={spec.lods} from level {top}: the dataset has levels 0..{top} below it")
    return list(range(top - spec.lods + 1, top + 1))


def _morton(xyz: np.ndarray) -> np.ndarray:
    """Z-curve keys of (n, 3) x, y, z positions (x the lowest bit)."""
    key = np.zeros(len(xyz), np.uint64)
    v = xyz.astype(np.uint64)
    for b in range(21):
        for a in range(3):
            key |= ((v[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return key


def surface_band(mask: np.ndarray, core: Box | None = None) -> np.ndarray:
    """The voxels within ``BAND`` of the surface of ``mask`` (inside or not; beyond its edges
    is outside), cropped to ``core`` (where ``mask`` is a block with a ``BAND`` border, so the
    parts of a level can be done apart and joined)."""
    from scipy.ndimage import maximum_filter, minimum_filter

    m = np.asarray(mask).astype(np.uint8)
    size = 2 * BAND + 1  # separable passes: a cube's dilation and erosion
    band = maximum_filter(m, size, mode="constant", cval=0) > minimum_filter(m, size, mode="constant", cval=0)
    return band if core is None else band[core.slices()]


def multires_nodes(spec: MeshSpec, band: np.ndarray, infos: list[ArrayInfo], chunk) -> list[np.ndarray]:
    """Per level of detail (finest first), its nodes, (n, 3) chunk positions ``z, y, x`` of its
    level in Z-curve order: those that meet ``band``, the voxels of the coarsest level near its
    surface (``surface_band``). A finer level's surface lies near the coarser one's, and a node
    listed but empty only costs its fetch."""
    levels = lod_levels(infos, spec)
    band = np.asarray(band, bool)
    chunk = np.asarray(chunk)
    out = []
    for i, level in enumerate(levels):
        scale = 2 ** (len(levels) - 1 - i)  # this level's voxels per coarsest voxel
        grid = -(-np.asarray(infos[level].shape[-3:]) // chunk)
        if (chunk % scale == 0).all():  # a node is a block of whole coarsest voxels
            b = chunk // scale
            pad = np.zeros(tuple(grid * b), bool)
            pad[tuple(slice(0, min(n, g)) for n, g in zip(band.shape, grid * b))] = band[tuple(slice(0, g) for g in grid * b)]
            nodes = np.argwhere(pad.reshape(grid[0], b[0], grid[1], b[1], grid[2], b[2]).any((1, 3, 5)))
        else:  # nodes smaller than a coarsest voxel: every node each band voxel covers
            f = scale // chunk
            at = np.argwhere(band)
            offsets = np.array(np.meshgrid(*[np.arange(n) for n in f], indexing="ij")).reshape(3, -1).T
            nodes = (at[:, None, :] * f + offsets[None]).reshape(-1, 3)
            nodes = nodes[(nodes < grid).all(1)]
        out.append(nodes[np.argsort(_morton(nodes[:, ::-1]), kind="stable")])
    if (n := sum(len(o) for o in out)) > MAX_NODES:
        raise ValueError(f"lods={spec.lods} lists {n} mesh nodes, over {MAX_NODES}: ask for fewer levels of detail")
    return out


def multires_index(nodes: list[np.ndarray], infos: list[ArrayInfo], levels: list[int], chunk) -> bytes:
    """The ``1.index`` manifest: nodes sized ``chunk`` voxels of each level (nanometres, x y z),
    each level's origin offset from the finest's, every fragment ``FRAGMENT_BYTES``."""
    first = infos[levels[0]]
    voxel = _nm(first)[-3:]
    origin = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(first.translation, first.units)])[-3:]
    shape = np.asarray(chunk, float) * voxel
    lod_scales = [float(_nm(infos[lv])[-3:].min()) for lv in levels]
    offsets = []
    for lv in levels:
        i = infos[lv]
        o = np.array([t * _TO_NM.get(u, 1.0) for t, u in zip(i.translation, i.units)])[-3:]
        offsets.extend((o - origin)[::-1])
    out = struct.pack("<3f", *shape[::-1]) + struct.pack("<3f", *origin[::-1]) + struct.pack("<I", len(levels))
    out += struct.pack(f"<{len(levels)}f", *lod_scales) + struct.pack(f"<{3 * len(levels)}f", *offsets)
    out += struct.pack(f"<{len(levels)}I", *[len(n) for n in nodes])
    for n in nodes:
        out += np.ascontiguousarray(n[:, ::-1].T, "<u4").tobytes() + np.full(len(n), FRAGMENT_BYTES, "<u4").tobytes()
    return out


def multires_info() -> dict:
    return {"@type": "neuroglancer_multilod_draco", "vertex_quantization_bits": BITS,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "lod_scale_multiplier": 1}


def node_box(info: ArrayInfo, node, chunk) -> Box:
    """A node's voxels: its chunk of the level and one more on its high sides (clipped)."""
    start = tuple(int(n) * c for n, c in zip(node, chunk))
    stop = tuple(min(s + c + 1, n) for s, c, n in zip(start, chunk, info.shape))
    return Box(start, stop)


def multires_fragment(spec: MeshSpec, block: np.ndarray, box: Box, info: ArrayInfo, chunk) -> tuple[np.ndarray, np.ndarray]:
    """A node's mesh as the format stores it: vertices as integers across the node (x, y, z,
    ``[0, 2**BITS)``) and triangles, marching cubes run on each octant apart so no triangle
    crosses one (as coarser levels must be), coarsened past ``MAX_TRIANGLES``."""
    from skimage.measure import marching_cubes

    mask = np.asarray(block) >= spec.threshold
    lo = [int(a == 0) for a in box.start]
    hi = [int(b == s) for b, s in zip(box.stop, info.shape)]
    padded = np.pad(mask, list(zip(lo, hi)))  # closed at the array's edges
    chunk = np.asarray(chunk)
    half = chunk // 2
    for step in (1, 2, 4, 8):
        verts, faces, n = [], [], 0
        for octant in np.ndindex(2, 2, 2):
            o = np.array(octant) * half
            stop = np.where(octant, chunk, half) + 1
            # in the padded block, node voxel j is at j + lo
            sub = padded[tuple(slice(a + e, b + e) for a, b, e in zip(o, stop, lo))]
            if min(sub.shape) < 2 or sub.all() or not sub.any():
                continue
            v, f, _, _ = marching_cubes(sub.astype(np.float32), 0.5, step_size=step)
            verts.append(v + o)
            faces.append(f + n)
            n += len(v)
        if not verts:
            return np.zeros((0, 3), np.uint32), np.zeros((0, 3), np.uint32)
        f = np.concatenate(faces)
        if len(f) <= MAX_TRIANGLES or step == 8:
            break
    v = np.concatenate(verts)  # node coordinates: the node's voxel 0 at 0
    q = np.clip(np.rint(v / chunk * (2**BITS - 1)), 0, 2**BITS - 1).astype(np.uint32)
    return np.ascontiguousarray(q[:, ::-1]), np.ascontiguousarray(f[:, ::-1].astype(np.uint32))


def encode_draco(verts: np.ndarray, faces: np.ndarray) -> bytes:
    """A fragment as Neuroglancer reads it: Draco, its integer positions kept as they are
    (quantized to ``BITS`` over exactly ``[0, 2**BITS - 1]``), padded to ``FRAGMENT_BYTES``.
    Needs DracoPy (the ``mesh`` extra); the browser engine encodes with Draco's own wasm."""
    import DracoPy

    if not len(faces):
        return b"\0" * FRAGMENT_BYTES
    data = DracoPy.encode(verts, faces, quantization_bits=BITS, quantization_range=2**BITS - 1,
                          quantization_origin=[0, 0, 0], compression_level=7)
    if len(data) > FRAGMENT_BYTES:
        raise ValueError(f"a mesh fragment took {len(data)} bytes, over {FRAGMENT_BYTES}")
    return data + b"\0" * (FRAGMENT_BYTES - len(data))
