"""``register://`` sources: a moving image registered onto a fixed image's grid, the
deformable part solved (on a GPU if there is one) when the source opens.

    register://<moving>?fixed=<fixed>&affine=<matrix.npy>
    register://<moving>?fixed=...&show=pair
    register://<moving>?fixed=...&show=field
    register://<moving>?fixed=...&frames=12
    register://<moving>?fixed=...&refine=3&halo=8

(the parameters follow the last ``?``, so ``<moving>`` may have a query of its own;
percent-encode ``fixed`` if it has one). Opening the source reads a few coarse levels of
both images and fits a smooth displacement field ``u`` in the fixed image's space
(``chunkmirage.registration``), which stays in memory: a whole organ's is megabytes.
Every level of the output is the moving image sampled at ``affine(p + u(p))`` for each
fixed voxel ``p``, computed chunk by chunk by the ``scene://`` resampler, so even full
resolution is served without anything being written. Solved fields are kept per process
(the last few), so reopening the same URL, as editing a downstream op does, reuses them.

With ``refine``, the field of the levels below the solved ones is fitted where it is
looked at: each has its own control lattice, in blocks of ``block`` voxels, and a block is
fitted when a chunk it covers is first requested, starting from the solved field over the
block plus ``halo`` voxels of context, coarse to fine within the block, then kept
(``_blocks``). So the fit's detail grows with the zoom, its cost with what is viewed rather
than with the volume, and any client asking for chunks drives it.

Query parameters:

* ``fixed``: the fixed image (required): anything ``open_source`` reads. The output has
  its grid and levels.
* ``affine``: the fixed-to-moving affine in physical units, C order: a ``.npy`` or text
  file holding a 4x4 or 3x4 matrix, or its 12 or 16 values inline, row by row (default:
  identity), or ``auto`` to find one from the images' intensity moments and a correlation
  fit (``mirrored``: the moving image is a mirror image, so the search tries only mirrored
  orientations). The moving image is sampled at ``affine(p + u(p))``.
* ``fixed_channel``, ``moving_channel``: the channel each image is matched on, for images
  with a ``c`` axis (default 0). The output has every channel of the moving image.
* ``levels``: fixed-image levels to solve on, coarse to fine, e.g. ``6,5,4`` (default:
  from the coarsest with at least 16 voxels on every axis to the finest with at most
  2^25). Each is matched against the moving level nearest its voxel size.
* ``iterations``: Adam steps per level, one value, one per level or one per level with
  the refined ones (default 100). ``0`` skips the solve: the affine alone, which needs no
  GPU.
* ``refine``: levels below the solved ones fitted block by block on demand (default 0),
  ``halo``: voxels of context around a block when it is fitted (default 8), ``block``:
  voxels per block, C order (default 64,128,128).
* ``smooth``: weight of the penalty on the field's gradient (default 1).
* ``grid``: control-point spacing of the field, in voxels of each level (default 4).
* ``window``: correlation window, voxels, odd: one value, one per level or one per level
  with the refined ones (default 7). Fine levels gain from a wider one.
* ``show``: ``image`` (default), ``pair`` (a ``c`` axis of two: the fixed image's channel,
  then the registered moving image's, to compare in one shader) or ``field`` (``u``
  itself, components first, in physical units).
* ``frames``: a leading ``t`` axis of that many fields from the solve, evenly spaced from
  none (the affine alone) to the final one, to watch it converge. The moving image's own
  ``t`` axis must have one time point, which the frames replace.
* ``interpolation``: ``linear`` (default) or ``nearest`` (default for uint32/uint64).
* ``chunk``: output chunk shape of the spatial axes, C order (default the fixed image's).
* ``device``: ``auto`` (default: the GPU with the most free memory, else the CPU),
  ``cpu``, ``cuda:1``, ...
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, field_validator

from chunkmirage import demand
from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.neuroglancer import compare_shader, field_shader
from chunkmirage.registration import (
    Grid,
    Level,
    Settings,
    _control_shape,
    _halve,
    _range,
    find_affine,
    solve,
)
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource, _pick_level
from chunkmirage.sources.warp import _SPATIAL, FieldLevelSource, FramesSource
from chunkmirage.transforms import Affine, Displacements, Sequence, VectorField, simplify

log = logging.getLogger("chunkmirage")


class RegisterParams(BaseModel):
    """The query of a ``register://`` URL. The one definition of it: the browser engine
    (``web/``) generates its TypeScript types and form defaults from this model's JSON
    Schema (``chunkmirage schema``)."""

    model_config = ConfigDict(extra="forbid")

    fixed: str = Field(
        description="The fixed image: anything open_source reads. The output has its grid and levels."
    )
    affine: str | None = Field(
        None,
        description=(
            "The fixed-to-moving affine in physical units, C order: a .npy or text file holding a 4x4"
            " or 3x4 matrix, or its 12 or 16 values inline, row by row; or auto, to find one from"
            " the images' intensity moments and a correlation fit. Default: identity."
        ),
    )
    mirrored: bool = Field(
        False,
        description=(
            "The moving image is a mirror image of the fixed one (one axis reversed, as when a stack is"
            " acquired the other way round), for affine=auto: the search then tries only mirrored"
            " orientations. Correlation cannot tell handedness on a nearly symmetric specimen."
        ),
    )
    fixed_channel: int = Field(
        0, ge=0, description="The fixed image's channel to match on, for images with a c axis."
    )
    moving_channel: int = Field(
        0,
        ge=0,
        description=(
            "The moving image's channel to match on; the output has every channel of the moving image."
        ),
    )
    levels: list[int] | None = Field(
        None,
        description=(
            "Fixed-image levels to solve on, coarse to fine. Default: from the coarsest with at least"
            " 16 voxels on every axis to the finest with at most 2^25."
        ),
    )
    iterations: list[NonNegativeInt] = Field(
        [100],
        description=(
            "Adam steps per level: one value, one per level, or one per level including the"
            " refined ones. 0 skips the solve (the affine alone)."
        ),
    )
    refine: int = Field(
        0,
        ge=0,
        description=(
            "Levels below the solved ones whose field is fitted on demand, block by block, where"
            " it is viewed: refine=2 with levels 6,5,4 fits levels 3 and 2, a block (one output"
            " chunk) when a chunk it covers is first requested, from the solved field, coarse to"
            " fine within the block. The finest fitted level's field serves every finer one."
        ),
    )
    halo: int = Field(
        8,
        ge=0,
        description=(
            "Voxels of context on every side of a block whose field is fitted on demand: how far"
            " beyond the block the fit looks, so neighbouring blocks agree."
        ),
    )
    block: list[PositiveInt] = Field(
        [64, 128, 128],
        min_length=3,
        max_length=3,
        description=(
            "Voxels per block of a field fitted on demand, C order. Independent of the output's"
            " chunks: a chunk's field reaches into the neighbouring blocks, so blocks deeper than a"
            " thin chunk fit fewer voxels in all, even for one slice."
        ),
    )
    smooth: float = Field(1.0, ge=0, description="Weight of the penalty on the field's gradient.")
    grid: float = Field(
        4.0, gt=0, description="Control-point spacing of the field, in voxels of each level."
    )
    window: list[int] = Field(
        [7],
        min_length=1,
        description=(
            "Correlation window, voxels, odd: one value, one per level, or one per level"
            " including the refined ones. Fine levels gain from a wider one."
        ),
    )
    show: Literal["image", "pair", "field"] = Field(
        "image",
        description=(
            "image: the registered moving image; pair: a c axis of two, the fixed image's channel"
            " then the registered one; field: u itself, components first, in physical units."
        ),
    )
    frames: int | None = Field(
        None,
        ge=1,
        description=(
            "A leading t axis of that many fields from the solve, from none (the affine alone) to the"
            " final one, to watch it converge."
        ),
    )
    interpolation: Literal["linear", "nearest"] | None = Field(
        None, description="Default: nearest for uint32/uint64 (labels), else linear."
    )
    chunk: list[PositiveInt] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Output chunk shape of the spatial axes, C order. Default: the fixed image's.",
    )
    device: str = Field(
        "auto",
        description="auto (the GPU with the most free memory, else the CPU), cpu, cuda:1, ...",
    )

    @classmethod
    def from_query(cls, q: dict[str, str]) -> RegisterParams:
        if unknown := set(q) - set(cls.model_fields):
            raise ValueError(
                f"unknown register:// parameters {sorted(unknown)}; allowed: {sorted(cls.model_fields)}"
            )
        return cls(**q)

    @field_validator("levels", "iterations", "window", "chunk", "block", mode="before")
    @classmethod
    def _comma_separated(cls, v):
        return [int(x) for x in v.split(",")] if isinstance(v, str) else v

    @field_validator("window")
    @classmethod
    def _odd(cls, v):
        if any(w < 1 or w % 2 == 0 for w in v):
            raise ValueError(f"window must be positive odd numbers, got {v}")
        return v


MAX_SOLVE_VOXELS = 1 << 25  # finest default level: its whole volume sits on the GPU
MIN_SOLVE_SIZE = 16  # coarsest default level: at least this many voxels on every axis
KEEP = 8  # solved fields remembered per process
BLOCK_CACHE_BYTES = 1 << 30  # blocks of refined fields remembered per process
_solved: OrderedDict[str, list[Grid]] = OrderedDict()
_affines: dict[str, np.ndarray] = {}  # found affines, per image pair
_ranges: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}  # per solve key
_blocks: LRUCache[np.ndarray] = LRUCache(BLOCK_CACHE_BYTES)
# fits of refined blocks: three at once (they read while one holds the GPU), the finest level
# and latest requests first, dropped if every request for one gives up before it starts
_block_queue = demand.queues["refined blocks"] = demand.Queue(slots=3)
_solve_lock = threading.Lock()  # one solve at a time: they share the GPU


class PairSource(Source):
    """The fixed image's channel and the registered moving image's, as the two channels of
    one ``(c, z, y, x)`` array (to compare them in a single shader)."""

    def __init__(self, fixed: Source, fixed_lead, registered: Source, moving_lead, info, key):
        self.parts = ((fixed, tuple(fixed_lead)), (registered, tuple(moving_lead)))
        self._info = info
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        space = Box(box.start[1:], box.stop[1:])
        out = np.empty(box.shape, dtype=self._info.dtype)
        for i, c in enumerate(range(box.start[0], box.stop[0])):
            src, lead = self.parts[c]
            b = Box(lead + space.start, tuple(v + 1 for v in lead) + space.stop)
            out[i] = src.read(b).reshape(space.shape)
        return out


def _n_lead(ms: MultiscaleSource, url: str) -> int:
    axes = ms.levels[0].info.axes
    n = 0
    while n < len(axes) and axes[n] not in _SPATIAL:
        n += 1
    if len(axes) - n != 3 or any(a not in _SPATIAL for a in axes[n:]):
        raise ValueError(f"{url}: register:// needs axes ending in z, y, x; got {axes}")
    return n


def _lead(ms: MultiscaleSource, n_lead: int, channel: int, what: str) -> tuple[int, ...]:
    """Index of the matched channel on the leading axes (the first of any other)."""
    info = ms.levels[0].info
    axes, shape = info.axes[:n_lead], info.shape[:n_lead]
    if "c" not in axes and channel:
        raise ValueError(f"{what}_channel={channel}, but the {what} image has no c axis")
    if "c" in axes and not 0 <= channel < shape[axes.index("c")]:
        raise ValueError(f"{what}_channel={channel} is outside its {shape[axes.index('c')]}")
    return tuple(channel if a == "c" else 0 for a in axes)


def _level(src: Source, lead: tuple[int, ...], start=None, stop=None) -> Level:
    """The matched channel of ``src`` (its voxels ``start:stop``, default all) as a ``Level``."""
    info = src.info
    n = len(lead)
    start = (0,) * 3 if start is None else tuple(int(v) for v in start)
    stop = info.shape[n:] if stop is None else tuple(int(v) for v in stop)
    box = Box(lead + start, tuple(v + 1 for v in lead) + stop)
    vox = np.asarray(info.voxel_size[n:], dtype=float)
    return Level(
        src.read(box).reshape(box.shape[n:]),
        vox,
        np.asarray(info.translation[n:], dtype=float) + np.asarray(start) * vox,
    )


def _spatial_vox(ms: MultiscaleSource, n_lead: int) -> list[np.ndarray]:
    return [np.asarray(lvl.info.voxel_size[n_lead:], dtype=float) for lvl in ms.levels]


def _default_levels(ms: MultiscaleSource, n_lead: int) -> list[int]:
    shapes = [lvl.info.shape[n_lead:] for lvl in ms.levels]
    fits = [i for i, s in enumerate(shapes) if math.prod(s) <= MAX_SOLVE_VOXELS]
    finest = fits[0] if fits else len(shapes) - 1
    coarse = [i for i, s in enumerate(shapes) if min(s) >= MIN_SOLVE_SIZE]
    coarsest = max(max(coarse, default=finest), finest)
    return list(range(coarsest, finest - 1, -1))


def _display_range(data: np.ndarray) -> tuple[float, float]:
    """Contrast limits from the voxels with data (which skips padding)."""
    v = data[data > 0]
    lo, hi = np.percentile(v, [1, 99.8]) if v.size else (0.0, 1.0)
    return float(lo), float(max(hi, lo + 1))


def _digest(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n]


def _affine(value: str | None) -> np.ndarray:
    if value is None:
        return np.eye(4)
    try:
        m = np.array([float(v) for v in value.split(",")])
    except ValueError:  # a file
        path = Path(value)
        m = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    m = np.asarray(m, dtype=float)
    if m.size in (12, 16):
        m = m.reshape(-1, 4)
    if m.shape == (3, 4):
        m = np.vstack([m, [0, 0, 0, 1]])
    if m.shape != (4, 4):
        raise ValueError(f"affine must be 4x4 or 3x4 (or 12/16 values), got shape {m.shape}")
    return m


def _refined_field(
    fixed: Source,
    flead: tuple[int, ...],
    moving: Source,
    mlead: tuple[int, ...],
    affine: np.ndarray,
    parent: VectorField,
    lo: np.ndarray,
    hi: np.ndarray,
    block_voxels: tuple[int, ...],
    settings: Settings,
    halo: int,
    stages: int,
    ranges,
    device: str,
    level: int,
) -> VectorField:
    """The field of a fixed level below the solved ones: a control lattice ``settings.grid``
    voxels apart, in blocks of ``block_voxels``. A block is fitted when first read,
    from ``parent``'s values (the solved field) over the block plus ``halo`` voxels of
    context, against the fixed and moving voxels those reach, coarse to fine over
    ``stages`` halvings of them (so the block's coarsest copy is about as coarse as the
    solved level), and kept in ``_blocks``, context included. Neighbouring blocks overlap
    there, and a lattice point's value is the average of the blocks covering it, each
    weighted 1 over its own block and less the further into its context (``_tent``), so the
    field turns smoothly from one block's fit to the next's across the overlap instead of
    stepping at their boundary. A block's fit depends on nothing but the solved field and
    the images, so neither does the blend on the order blocks are fitted in."""
    finfo, minfo = fixed.info, moving.info
    nf, nm = len(flead), len(mlead)
    fvox = np.asarray(finfo.voxel_size[nf:], dtype=float)
    ft = np.asarray(finfo.translation[nf:], dtype=float)
    fshape = np.asarray(finfo.shape[nf:])
    mvox = np.asarray(minfo.voxel_size[nm:], dtype=float)
    mt = np.asarray(minfo.translation[nm:], dtype=float)
    mshape = np.asarray(minfo.shape[nm:])
    a = np.asarray(affine, dtype=float)
    spacing = settings.grid * fvox
    n = np.asarray(_control_shape(lo, hi, fvox, settings.grid))
    block = np.asarray([math.ceil(c / settings.grid) for c in block_voxels])
    context = math.ceil(halo / settings.grid)  # lattice points
    window = settings.window[0]
    w2 = window // 2
    margin = halo + w2 + 1  # moving voxels beyond the parent field's reach a block may move
    info = ArrayInfo(
        shape=(*n, 3),
        dtype=np.float32,
        chunk_shape=(*block, 3),
        voxel_size=(*spacing, 1.0),
        units=(*finfo.units[nf:], ""),
        axes=("z", "y", "x", "c"),
        translation=(*lo, 0.0),
    )
    key = "register:" + _digest(
        f"refine|{parent.key}|{fixed.cache_key()}|{moving.cache_key()}|{flead}|{mlead}|"
        f"{np.round(a, 12).tolist()}|{dataclasses.asdict(settings)}|{halo}|{block_voxels}"
    )

    def sampled(blo, bhi, shape) -> np.ndarray:
        """``parent`` on a grid of ``shape`` points spanning the box, (z, y, x, 3)."""
        g = np.mgrid[tuple(slice(0, s) for s in shape)].reshape(3, -1).T
        pts = blo + g * (bhi - blo) / (np.asarray(shape) - 1)
        return parent.sample(pts).reshape(*shape, 3).astype(np.float32)

    def extent(index):
        """Block ``index``'s own lattice points ``c0:c1`` and those it is fitted over, ``k0:k1``."""
        c0 = np.asarray(index) * block
        c1 = np.minimum(c0 + block, n)
        k0, k1 = np.maximum(c0 - context, 0), np.minimum(c1 + context, n)
        k1 = np.minimum(np.maximum(k1, k0 + 2), n)  # a fit needs two points per axis
        k0 = k1 - np.maximum(k1 - k0, 2)
        return c0, c1, k0, k1

    def compute(index) -> np.ndarray:
        """Block ``index`` fitted: its values over ``k0:k1``, (z, y, x, 3)."""
        c0, c1, k0, k1 = extent(index)
        blo, bhi = lo + k0 * spacing, lo + (k1 - 1) * spacing
        if settings.iterations[0] == 0:
            return sampled(blo, bhi, k1 - k0)
        # the fixed voxels whose correlation windows the block's points reach
        v0 = np.maximum(np.floor((blo - ft) / fvox).astype(int) - w2, 0)
        v1 = np.minimum(np.ceil((bhi - ft) / fvox).astype(int) + w2 + 1, fshape)
        if (v1 - v0 < window).any():  # beyond the image, or too thin to correlate
            return sampled(blo, bhi, k1 - k0)
        fl = _level(fixed, flead, v0, v1)
        # the moving voxels those reach under the parent field and the affine, with room to move
        q = fl.translation + np.mgrid[0:9, 0:9, 0:9].reshape(3, -1).T / 8 * (v1 - v0 - 1) * fvox
        q = (q + parent.sample(q)) @ a[:3, :3].T + a[:3, 3]
        q = (q - mt) / mvox
        m0 = np.maximum(np.floor(q.min(axis=0)).astype(int) - margin, 0)
        m1 = np.minimum(np.ceil(q.max(axis=0)).astype(int) + margin + 1, mshape)
        if (m1 <= m0).any():  # nothing of the moving image here
            return sampled(blo, bhi, k1 - k0)
        ml = _level(moving, mlead, m0, m1)
        fs, ms = [fl], [ml]
        for _ in range(stages):  # coarse to fine within the block, while a copy is worth it
            if min(fs[0].data.shape) < 2 * max(MIN_SOLVE_SIZE, window):
                break
            fs.insert(0, _halve(fs[0]))
            ms.insert(0, _halve(ms[0]))
        init = sampled(blo, bhi, _control_shape(blo, bhi, fs[0].voxel_size, settings.grid))
        with _solve_lock:  # one fit at a time: they share the GPU
            grid = solve(
                fs,
                ms,
                a,
                dataclasses.replace(
                    settings,
                    iterations=settings.iterations[:1] * len(fs),
                    window=settings.window[:1] * len(fs),
                ),
                box=(blo, bhi),
                device=device,
                init=init,
                ranges=ranges,
                label=f" level {level} block {tuple(int(i) for i in index[:3])}",
            )[0]
        return grid.values

    def fitted(index: tuple[int, ...]) -> np.ndarray:
        """Block ``index``'s fit: kept in ``_blocks``, fitted once through ``_block_queue``
        however many requests want it, and dropped unfitted if they all give up first."""
        hit = _blocks.get((key, index))
        if hit is not None:
            return hit

        def fit() -> np.ndarray:
            values = np.ascontiguousarray(compute(index), dtype=np.float32)
            _blocks.put((key, index), values)
            return values

        return _block_queue.run((key, index), level, fit)

    nblocks = -(-n // block)

    def read(sl: tuple[slice, ...]) -> np.ndarray:
        """The blended lattice over ``sl``: every block whose fit reaches it, weighted."""
        w0 = np.asarray([s.start or 0 for s in sl[:3]])
        w1 = np.asarray([size if s.stop is None else s.stop for s, size in zip(sl[:3], n)])
        acc = np.zeros((*(w1 - w0), 3))
        total = np.zeros(tuple(w1 - w0))
        b0 = np.maximum((w0 - context) // block, 0)
        b1 = np.minimum((w1 - 1 + context) // block, nblocks - 1)
        for index in np.ndindex(*(b1 - b0 + 1)):
            index = tuple(int(v) for v in b0 + index)
            c0, c1, k0, k1 = extent(index)
            i0, i1 = np.maximum(k0, w0), np.minimum(k1, w1)
            if (i1 <= i0).any():
                continue
            weight = _tent(i0, i1, c0, c1, context)
            if not weight.any():
                continue
            values = fitted(index)[tuple(slice(a - k, b - k) for a, b, k in zip(i0, i1, k0))]
            box = tuple(slice(a - w, b - w) for a, b, w in zip(i0, i1, w0))
            acc[box] += weight[..., None] * values
            total[box] += weight
        return (acc / total[..., None]).astype(np.float32)[..., sl[3]]

    return VectorField(read, info.shape, 3, Affine.scale_translation(spacing, lo), key)


def _tent(i0, i1, c0, c1, context: int) -> np.ndarray:
    """A block's weight at lattice points ``i0:i1``: 1 on its own points ``c0:c1``, falling
    linearly to 1 / (context + 1) at the edge of its context, 0 beyond; the product over axes."""
    weight = np.ones(())
    for a, b, lo, hi in zip(i0, i1, c0, c1):
        k = np.arange(a, b)
        d = np.maximum(np.maximum(lo - k, k - (hi - 1)), 0)
        weight = np.multiply.outer(weight, np.clip(1 - d / (context + 1), 0, 1))
    return weight


_FIELDS = ("shape", "chunk_shape", "voxel_size", "units", "axes", "translation")


def _axes_of(info: ArrayInfo, sl: slice) -> ArrayInfo:
    """``info`` restricted to the axes ``sl``."""
    return ArrayInfo(dtype=info.dtype, **{f: tuple(getattr(info, f)[sl]) for f in _FIELDS})


def _prepend(space: ArrayInfo, lead: ArrayInfo) -> ArrayInfo:
    """Axes of ``lead`` followed by those of ``space``, with ``lead``'s dtype."""
    return ArrayInfo(
        dtype=lead.dtype, **{f: tuple(getattr(lead, f)) + tuple(getattr(space, f)) for f in _FIELDS}
    )


def open_register(url: str, *, cache_bytes: int = 0) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    location, _, query = url[len("register://") :].rpartition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if "fixed" not in q or not location:
        raise ValueError(
            "register:// needs a moving image and fixed=...: register://<moving>?fixed=<fixed>"
            " (the parameters follow the last '?')"
        )
    p = RegisterParams.from_query(q)

    mov = open_source(location, cache_bytes=cache_bytes)
    fix = open_source(p.fixed, cache_bytes=cache_bytes)
    nm, nf = _n_lead(mov, location), _n_lead(fix, p.fixed)
    minfo, finfo = mov.levels[0].info, fix.levels[0].info
    if tuple(minfo.axes[nm:]) != tuple(finfo.axes[nf:]):
        raise ValueError(f"spatial axes differ: moving {minfo.axes}, fixed {finfo.axes}")
    if tuple(minfo.units[nm:]) != tuple(finfo.units[nf:]):
        raise ValueError(
            f"spatial units differ: moving {minfo.units[nm:]}, fixed {finfo.units[nf:]}"
        )
    mlead = _lead(mov, nm, p.moving_channel, "moving")
    flead = _lead(fix, nf, p.fixed_channel, "fixed")

    levels = p.levels or _default_levels(fix, nf)
    if any(not 0 <= i < len(fix.levels) for i in levels):
        raise ValueError(f"levels {levels}: the fixed image has levels 0..{len(fix.levels) - 1}")
    if p.refine > levels[-1]:
        raise ValueError(
            f"refine={p.refine}: {levels[-1]} level(s) lie below level {levels[-1]}, the finest solved"
        )
    if p.refine and p.frames is not None:
        raise ValueError(
            "frames and refine do not combine: refined blocks are fitted after the solve"
        )
    n_all = len(levels) + p.refine

    def per_level(name: str, values: list[int]) -> list[int]:
        """``values`` for every level, refined ones included: one value for all, one per
        solved level (the refined ones take the last) or one per level."""
        if len(values) not in {1, len(levels), n_all}:
            counts = " or ".join(str(c) for c in sorted({1, len(levels), n_all}))
            raise ValueError(
                f"{name} needs {counts} values: one, one per level, or one per level with the"
                " refined ones"
            )
        values = values * n_all if len(values) == 1 else values
        return values + values[-1:] * (n_all - len(values))

    iters, windows = per_level("iterations", p.iterations), per_level("window", p.window)
    settings = Settings(
        iterations=tuple(iters[: len(levels)]),
        smooth=p.smooth,
        grid=p.grid,
        window=tuple(windows[: len(levels)]),
    )
    moving_t = p.show == "image" and "t" in minfo.axes  # which the frames replace
    if p.frames is not None and moving_t and (minfo.axes[0] != "t" or minfo.shape[0] != 1):
        raise ValueError(
            f"{location}: frames need a moving image without a time axis, or with one time"
            f" point first; axes {minfo.axes}, shape {minfo.shape}"
        )

    labels = minfo.dtype.kind == "u" and minfo.dtype.itemsize >= 4
    interpolation = p.interpolation or ("nearest" if labels else "linear")
    order = 0 if interpolation == "nearest" else 1

    # Solve (or recall): moving levels nearest each fixed level's voxel size.
    mvox, fvox = _spatial_vox(mov, nm), _spatial_vox(fix, nf)
    mlevels = [int(np.argmin([np.abs(np.log(m / fvox[i])).sum() for m in mvox])) for i in levels]
    coarsest: list[Level] = []  # the coarsest solved level of each image, read once if needed

    def read_coarsest() -> list[Level]:
        if not coarsest:
            coarsest.append(_level(fix.levels[levels[0]], flead))
            coarsest.append(_level(mov.levels[mlevels[0]], mlead))
        return coarsest

    if p.affine == "auto":
        pair = [fix.levels[levels[0]].cache_key(), mov.levels[mlevels[0]].cache_key(), flead, mlead]
        pair.append(p.mirrored)
        akey = _digest(json.dumps(pair), 16)
        if akey not in _affines:
            with _solve_lock:
                _affines[akey] = find_affine(
                    *read_coarsest(), device=p.device, mirrored=p.mirrored
                )[0]
        affine = _affines[akey]
    else:
        affine = _affine(p.affine)
    ident = {
        "fixed": [fix.levels[i].cache_key() for i in levels],
        "moving": [mov.levels[j].cache_key() for j in mlevels],
        "lead": [flead, mlead],
        "affine": np.round(affine, 12).tolist(),
        "settings": dataclasses.asdict(settings),
        "frames": p.frames,
    }
    solve_key = _digest(json.dumps(ident, sort_keys=True), 16)
    lo = np.asarray(finfo.translation[nf:], dtype=float) - fvox[0] / 2
    hi = lo + np.asarray(finfo.shape[nf:]) * fvox[0]

    with _solve_lock:
        grids = _solved.get(solve_key)
        if grids is not None:
            _solved.move_to_end(solve_key)
    if grids is None:
        # read outside the lock, so other opens need not wait behind this one's I/O; the
        # affine alone reads nothing
        solving = sum(settings.iterations) > 0
        fixed_levels, moving_levels = [], []
        if solving:
            fixed_levels = [read_coarsest()[0]] + [_level(fix.levels[i], flead) for i in levels[1:]]
            moving_levels = [read_coarsest()[1]] + [
                _level(mov.levels[j], mlead) for j in mlevels[1:]
            ]
        with _solve_lock:  # one solve at a time: they share the GPU
            if solve_key not in _solved:
                log.info("register: %s onto %s at levels %s", location, p.fixed, levels)
                _solved[solve_key] = solve(
                    fixed_levels,
                    moving_levels,
                    affine,
                    settings,
                    box=(lo, hi),
                    device=p.device,
                    snapshots=p.frames or 1,
                )
                if solving:
                    _ranges[solve_key] = (
                        _range(fixed_levels[0].data),
                        _range(moving_levels[0].data),
                    )
                while len(_solved) > KEEP:
                    _ranges.pop(_solved.popitem(last=False)[0], None)
            grids = _solved[solve_key]

    to_moving = Affine(affine[:3])

    def field_of(k: int) -> VectorField:
        g = grids[k]
        values = g.values
        return VectorField(
            lambda sl: values[sl],
            values.shape,
            3,
            Affine.scale_translation(g.spacing, g.origin),
            f"register:{solve_key}:{k}",
        )

    fields = [field_of(k) for k in range(len(grids))]

    def space_chunk(i: int) -> tuple[int, ...]:
        return tuple(p.chunk or fix.levels[i].info.chunk_shape[nf:])

    # Below the solved levels, each refined level's field is fitted block by block on
    # demand from the solved field (nothing is fitted until a chunk is asked for).
    refined: dict[int, VectorField] = {}
    if p.refine:
        ranges = _ranges.get(solve_key)
        if ranges is None:
            ranges = _ranges[solve_key] = tuple(_range(lvl.data) for lvl in read_coarsest())
        for r, i in enumerate(range(levels[-1] - 1, levels[-1] - 1 - p.refine, -1)):
            j = int(np.argmin([np.abs(np.log(m / fvox[i])).sum() for m in mvox]))
            refined[i] = _refined_field(
                fix.levels[i],
                flead,
                mov.levels[j],
                mlead,
                affine,
                fields[0],
                lo,
                hi,
                tuple(p.block),
                dataclasses.replace(
                    settings,
                    iterations=(iters[len(levels) + r],),
                    window=(windows[len(levels) + r],),
                ),
                p.halo,
                levels[-1] - i,
                ranges,
                p.device,
                i,
            )

    def field_for(i: int, k: int) -> VectorField:
        """The field output level ``i`` is served through: the finest fitted at or above it."""
        if not refined or i >= levels[-1]:
            return fields[k]
        return refined[max(i, min(refined))]

    def level(i: int, k: int) -> Source:
        """Fixed level ``i`` through field ``k``."""
        fl = fix.levels[i].info
        grid = dataclasses.replace(_axes_of(fl, slice(nf, None)), chunk_shape=space_chunk(i))
        to_space = Affine.scale_translation(fvox[i], fl.translation[nf:])
        field = field_for(i, k)
        key = "register:" + _digest(f"register|{field.key}|{i}|{p.show}|{order}|{grid.chunk_shape}")
        if p.show == "field":
            info = _prepend(grid, ArrayInfo((3,), np.float32, (3,), (1.0,), ("",), ("c",)))
            return FieldLevelSource(field, to_space, info, key)

        def mapping_for(j: int, through: VectorField):
            ml = mov.levels[j].info
            to_index = Affine.scale_translation(mvox[j], ml.translation[nm:]).inverse()
            return simplify(Sequence([to_space, Displacements(through), to_moving, to_index]))

        # the moving level is chosen through the solved field: sampling a refined one
        # here would fit blocks before anything is viewed
        j = _pick_level(grid.shape, mapping_for(0, fields[k]), mvox)
        ml = mov.levels[j].info
        info = _prepend(grid, _axes_of(ml, slice(nm)))
        registered = SceneLevelSource(mov.levels[j], mapping_for(j, field), info, nm, order, key)
        if p.show == "image":
            return registered
        dtype = np.result_type(fl.dtype, ml.dtype)
        pair = _prepend(grid, ArrayInfo((2,), dtype, (2,), (1.0,), ("",), ("c",)))
        return PairSource(fix.levels[i], flead, registered, mlead, pair, key + ":pair")

    out: list[Source] = []
    for i in range(len(fix.levels)):
        frames = [level(i, k) for k in range(len(grids))]
        if p.frames is None:
            out.append(frames[0])
            continue
        key = "register:" + _digest("frames|" + "|".join(f.cache_key() for f in frames))
        out.append(FramesSource(frames, moving_t, key))
    # how a viewer should show it: the pair compared in colour, the field as a heat map
    shader = None
    if p.show == "pair":
        shader = compare_shader(*(_display_range(lvl.data) for lvl in read_coarsest()))
    elif p.show == "field":
        moved = np.linalg.norm(grids[-1].values, axis=-1)
        shader = field_shader(max(float(np.percentile(moved, 99)), float(np.mean(fvox[0]))))
    return MultiscaleSource(out, name=f"{mov.name or location}~registered", shader=shader)
