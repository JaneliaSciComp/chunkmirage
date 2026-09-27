"""``scene://`` sources: an image resampled through OME-Zarr 0.6 coordinate transformations.

    scene://<group>?image=<image path>&target=<image path or coordinate system>

serves the image at ``image`` resampled onto the grid of ``target``, so viewers that cannot
apply the transformations (displacement fields in particular; none can as of 0.6) still
show it registered. Each output chunk is computed on its own: its voxel centres are pushed
through the transformation chain into the source image, the bounding box of the results
is read, and the values are interpolated. Nothing is written.

Query parameters:

* ``image``: path of the image to resample, relative to the group (default: the group
  itself, when it is a multiscale image).
* ``target``: an image path (use its grid and levels) or a coordinate system name (use
  the source image's grid, or ``shape``/``voxel_size``/``translation``). Default: the
  scene's first coordinate system, else the image's own.
* ``interpolation``: ``linear`` (default) or ``nearest`` (default for label images).
* ``inverse``: ``exact`` (default) or ``approx``, which lets the path walk displacement
  fields backwards by estimating their inverse numerically.
* ``shape``, ``voxel_size``, ``translation``, ``levels``, ``chunk``: explicit output grid
  (spatial axes, C order, in the target coordinate system's units).
* ``field_cache_gb``: decoded-chunk cache per displacement field (default 1/16 GB).
* ``large_field_chunks``: fields stored in chunks over 256 MiB are refused, because every
  output chunk would decode a whole one and a viewer's parallel requests could exhaust
  memory; ``1`` accepts them and decodes one at a time (set ``field_cache_gb`` above one
  chunk so each is decoded once).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ngff import DEFAULT_FIELDS, CoordinateSystem, FieldOptions, Image, Node, Scene
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.tensorstore_source import TensorStoreSource
from chunkmirage.transforms import Affine, Sequence, Transform, simplify

MAX_POINTS = 1 << 20  # voxels resampled at once; bounds the coordinate arrays' memory
MAX_WINDOW_BYTES = 2 << 30  # most one output slab may read from the source image
_PARAMS = {
    "image",
    "target",
    "interpolation",
    "inverse",
    "shape",
    "voxel_size",
    "translation",
    "levels",
    "chunk",
    "field_cache_gb",
    "large_field_chunks",
}


@dataclass
class _Grid:
    """One output level: a spatial voxel grid and where it sits in the target system."""

    shape: tuple[int, ...]
    voxel_size: tuple[float, ...]
    translation: tuple[float, ...]
    chunk: tuple[int, ...]
    to_target: Affine  # grid index (spatial axes only) -> target coordinate system


def _embed(cs: CoordinateSystem, voxel, translation) -> Affine:
    """Spatial grid index -> points in ``cs``; non-spatial axes of ``cs`` are held at 0."""
    s = cs.spatial
    m = np.zeros((cs.ndim, len(s) + 1))
    for k, axis in enumerate(s):
        m[axis, k] = voxel[k]
        m[axis, -1] = translation[k]
    return Affine(m)


def _image_grids(img: Image, levels: list[TensorStoreSource]) -> list[_Grid]:
    s = img.cs.spatial
    grids = []
    for aff, lvl in zip(img.level_affines, levels):
        voxel = tuple(float(aff.linear[a, a]) for a in s)
        trans = tuple(float(aff.offset[a]) for a in s)
        grids.append(
            _Grid(
                shape=tuple(lvl.info.shape[a] for a in s),
                voxel_size=voxel,
                translation=trans,
                chunk=tuple(lvl.info.chunk_shape[a] for a in s),
                to_target=_embed(img.cs, voxel, trans),
            )
        )
    return grids


def _per_axis(q: dict, key: str, n: int, default, kind) -> list:
    """``key`` from the query as ``n`` values: one value applies to every axis."""
    if key not in q:
        return [default] * n
    vals = [kind(v) for v in q[key].split(",")]
    if len(vals) not in (1, n):
        raise ValueError(f"{key} needs 1 or {n} values (spatial axes, C order), got {len(vals)}")
    return vals * n if len(vals) == 1 else vals


def _explicit_grids(cs: CoordinateSystem, q: dict, n_levels: int) -> list[_Grid]:
    shape = [int(v) for v in q["shape"].split(",")]
    n = len(shape)
    if n != len(cs.spatial):
        raise ValueError(f"shape has {n} axes; target {cs.name!r} has {len(cs.spatial)} spatial")
    voxel = _per_axis(q, "voxel_size", n, 1.0, float)
    trans = _per_axis(q, "translation", n, 0.0, float)
    chunk = _per_axis(q, "chunk", n, 64, int)
    count = int(q.get("levels", n_levels))
    grids = []
    for lvl in range(count):
        f = 2**lvl
        shp = tuple(max(1, math.ceil(s / f)) for s in shape)
        vox = tuple(v * f for v in voxel)
        # Level centres shift by half the growth in voxel size (OME-NGFF convention).
        tr = tuple(t + (f - 1) * v / 2 for t, v in zip(trans, voxel))
        grids.append(_Grid(shp, vox, tr, tuple(chunk), _embed(cs, vox, tr)))
        if all(s == 1 for s in shp):
            break
    return grids


def _select(cs: CoordinateSystem) -> Affine:
    """Points in ``cs`` -> their spatial coordinates."""
    s = cs.spatial
    m = np.zeros((len(s), cs.ndim + 1))
    for k, axis in enumerate(s):
        m[k, axis] = 1.0
    return Affine(m)


def _pick_level(grid: _Grid, to_level0: Transform, src: Image) -> int:
    """The coarsest source level still at least as fine as one output voxel, measured
    through the transform at the grid centre (so a zoomed-out view reads a small level)."""
    n = len(grid.shape)
    centre = (np.asarray(grid.shape, dtype=float) - 1) / 2
    steps = np.vstack([centre + 0.5 * e for e in np.eye(n)] + [centre - 0.5 * e for e in np.eye(n)])
    mapped = to_level0.apply(steps)
    jac = (mapped[:n] - mapped[n:]).T  # (source spatial, output spatial), level-0 voxels
    extent = np.maximum(np.abs(jac).sum(axis=1), 1.0)
    s = src.cs.spatial
    base = np.array([src.level_affines[0].linear[a, a] for a in s])
    best = 0
    for lvl, aff in enumerate(src.level_affines):
        factor = np.array([aff.linear[a, a] for a in s]) / base
        if np.all(factor <= extent * 1.01):
            best = lvl
    return best


class SceneLevelSource(Source):
    """One output level: resamples ``src`` at ``mapping(grid index)`` (source voxel units)."""

    def __init__(
        self, src: Source, mapping: Transform, info: ArrayInfo, n_lead: int, order: int, key: str
    ):
        self.src = src
        self.mapping = mapping
        self._info = info
        self.n_lead = n_lead  # leading non-spatial axes (time, channel), passed through
        self.order = order
        self._key = key
        self._src_spatial = np.asarray(src.info.shape[n_lead:], dtype=float)

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        k = self.n_lead
        lead = Box(box.start[:k], box.stop[:k])
        space = Box(box.start[k:], box.stop[k:])
        out = np.zeros(box.shape, dtype=self._info.dtype)
        rows = max(1, MAX_POINTS // max(1, math.prod(space.shape[1:])))
        for z0 in range(space.start[0], space.stop[0], rows):
            z1 = min(z0 + rows, space.stop[0])
            slab = Box((z0, *space.start[1:]), (z1, *space.stop[1:]))
            idx = (slice(None),) * k + (slice(z0 - space.start[0], z1 - space.start[0]),)
            out[idx] = self._resample(lead, slab)
        return out

    def _resample(self, lead: Box, slab: Box) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        n = math.prod(slab.shape)
        pts = np.indices(slab.shape, dtype=float).reshape(len(slab.shape), -1).T
        coords = self.mapping.apply(pts + np.asarray(slab.start, dtype=float))
        # A voxel owns [-0.5, 0.5) around its centre; points outside the image get 0.
        inside = np.all((coords >= -0.5) & (coords < self._src_spatial - 0.5), axis=1)
        res = np.zeros((*lead.shape, n), dtype=self._info.dtype)
        if inside.any():
            c = coords[inside]
            if self.order == 0:
                c = np.floor(c + 0.5)
                lo, hi = c.min(axis=0).astype(int), c.max(axis=0).astype(int) + 1
            else:
                lo, hi = (
                    np.floor(c.min(axis=0)).astype(int),
                    np.floor(c.max(axis=0)).astype(int) + 2,
                )
            lo = np.clip(lo, 0, self._src_spatial.astype(int))
            hi = np.clip(hi, 0, self._src_spatial.astype(int))
            window = Box(tuple(lead.start) + tuple(lo), tuple(lead.stop) + tuple(hi))
            nbytes = math.prod(window.shape) * self.src.info.dtype.itemsize
            if nbytes > MAX_WINDOW_BYTES:  # refuse rather than risk the machine's memory
                raise MemoryError(
                    f"{self._key}: output region {slab} would read {nbytes / 2**30:.1f} GiB of "
                    f"the source (window {window.shape}); the transformation spreads it over a "
                    "large area. Use smaller output chunks (chunk=...)."
                )
            data = self.src.read(window)
            local = c - lo
            if self.order == 0:
                vals = data[(Ellipsis, *local.astype(np.intp).T)]
            else:
                vals = np.empty((*lead.shape, len(c)))
                for li in np.ndindex(*lead.shape):
                    vals[li] = map_coordinates(
                        data[li], local.T, order=1, mode="nearest", prefilter=False, output=float
                    )
                if np.issubdtype(self._info.dtype, np.integer):
                    lim = np.iinfo(self._info.dtype)
                    vals = np.clip(np.rint(vals), lim.min, lim.max)
            res[..., inside] = vals
        return res.reshape(*lead.shape, *slab.shape)


def _resolve_target(scene: Scene, src: Image, target: str | None) -> tuple[Node, Image | None]:
    if target is None:
        if scene.systems:
            return (None, next(iter(scene.systems))), None
        return (src.path, src.intrinsic), None
    t = target.strip("/")
    if t in scene.images:  # every image the graph references is already loaded
        img = scene.images[t]
        return (img.path, img.intrinsic), img
    if target in scene.systems:
        return (None, target), None
    if target in src.systems:
        return (src.path, target), None
    owners = [p for p, im in scene.images.items() if target in im.systems]
    if len(owners) == 1:
        return (owners[0], target), None
    try:  # an image the graph does not mention yet
        img = scene.image(t)
        return (img.path, img.intrinsic), img
    except (ValueError, KeyError):
        pass
    raise ValueError(
        f"target {target!r} is neither an image nor a unique coordinate system; images: "
        f"{sorted(scene.images)}, scene systems: {sorted(scene.systems)}"
    )


def open_scene(url: str, *, cache_bytes: int = 0) -> MultiscaleSource:
    location, _, query = url[len("scene://") :].partition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if unknown := set(q) - _PARAMS:
        raise ValueError(
            f"unknown scene:// parameters {sorted(unknown)}; allowed: {sorted(_PARAMS)}"
        )
    fields = FieldOptions(
        cache_bytes=int(float(q["field_cache_gb"]) * 2**30)
        if "field_cache_gb" in q
        else DEFAULT_FIELDS.cache_bytes,
        allow_large_chunks=q.get("large_field_chunks", "0").lower() in ("1", "true", "yes"),
    )
    scene = Scene(location, fields=fields)
    image = q.get("image", "")
    if image == "" and "" not in scene.images:
        raise ValueError(
            f"{location} is a scene; choose an image with ?image=, e.g. one of "
            f"{sorted({p for a, b, _ in scene.edges for p in (a[0], b[0]) if p})}"
        )
    src = scene.image(image)
    s_cs = src.cs
    n_lead = min(s_cs.spatial) if s_cs.spatial else 0
    if s_cs.spatial != list(range(n_lead, s_cs.ndim)):
        raise ValueError(f"{src.url}: spatial axes must come last, got types {s_cs.types}")
    src_levels = [
        TensorStoreSource.from_path(f"{src.url}/{p}", cache_bytes=cache_bytes) for p in src.datasets
    ]

    target_node, target_img = _resolve_target(scene, src, q.get("target"))
    t_cs = scene.system(target_node)
    approx = q.get("inverse", "exact") == "approx"
    chain = scene.transform(target_node, (src.path, src.intrinsic), approximate=approx)

    if target_img is not None:
        tgt_levels = [
            TensorStoreSource.from_path(f"{target_img.url}/{p}", cache_bytes=cache_bytes)
            for p in target_img.datasets
        ]
        grids = _image_grids(target_img, tgt_levels)
    elif "shape" in q:
        grids = _explicit_grids(t_cs, q, len(src_levels))
    else:
        if len(t_cs.spatial) != len(s_cs.spatial):
            raise ValueError(f"target {t_cs.name!r} has no grid of its own; pass shape=...")
        grids = [
            _Grid(
                g.shape,
                g.voxel_size,
                g.translation,
                g.chunk,
                _embed(t_cs, g.voxel_size, g.translation),
            )
            for g in _image_grids(src, src_levels)
        ]
    if "chunk" in q:
        chunk = tuple(_per_axis(q, "chunk", len(grids[0].shape), 64, int))
        grids = [_Grid(g.shape, g.voxel_size, g.translation, chunk, g.to_target) for g in grids]

    default = "nearest" if src.is_label else "linear"
    interpolation = q.get("interpolation", default)
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"interpolation must be linear or nearest, got {interpolation!r}")
    order = 0 if interpolation == "nearest" else 1

    select = _select(s_cs)
    lead_shape = src_levels[0].info.shape[:n_lead]
    lead_chunk = src_levels[0].info.chunk_shape[:n_lead]
    target_label = q.get("target") or t_cs.name
    levels: list[Source] = []
    for grid in grids:

        def mapping_for(level: int, grid=grid) -> Transform:
            to_index = src.level_affines[level].inverse()
            return simplify(Sequence([grid.to_target, chain, to_index, select]))

        lvl = _pick_level(grid, mapping_for(0), src)
        mapping = mapping_for(lvl)
        lead = src.level_affines[lvl]  # time and channel pass through with their own scale
        info = ArrayInfo(
            shape=tuple(lead_shape) + grid.shape,
            dtype=src_levels[lvl].info.dtype,
            chunk_shape=tuple(lead_chunk) + grid.chunk,
            voxel_size=tuple(float(lead.linear[a, a]) for a in range(n_lead)) + grid.voxel_size,
            units=tuple(s_cs.units[:n_lead]) + tuple(t_cs.units[a] for a in t_cs.spatial),
            axes=tuple(s_cs.names[:n_lead]) + tuple(t_cs.names[a] for a in t_cs.spatial),
            translation=tuple(float(lead.offset[a]) for a in range(n_lead)) + grid.translation,
        )
        ident = (
            f"{location}|{image}|{target_label}|{order}|{mapping.digest()}|"
            f"{src_levels[lvl].cache_key()}|{info.shape}"
        )
        key = "scene:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        levels.append(SceneLevelSource(src_levels[lvl], mapping, info, n_lead, order, key))
    return MultiscaleSource(levels, name=f"{image or 'image'}@{target_label}")
