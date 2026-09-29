"""``register://`` sources: a moving image registered onto a fixed image's grid, the
deformable part solved (on a GPU if there is one) when the source opens.

    register://<moving>?fixed=<fixed>&affine=<matrix.npy>
    register://<moving>?fixed=...&show=pair
    register://<moving>?fixed=...&show=field
    register://<moving>?fixed=...&frames=12

(the parameters follow the last ``?``, so ``<moving>`` may have a query of its own;
percent-encode ``fixed`` if it has one). Opening the source reads a few coarse levels of
both images and fits a smooth displacement field ``u`` in the fixed image's space
(``chunkmirage.registration``), which stays in memory: a whole organ's is megabytes.
Every level of the output is the moving image sampled at ``affine(p + u(p))`` for each
fixed voxel ``p``, computed chunk by chunk by the ``scene://`` resampler, so even full
resolution is served without anything being written. Solved fields are kept per process
(the last few), so reopening the same URL, as editing a downstream op does, reuses them.

Query parameters:

* ``fixed``: the fixed image (required): anything ``open_source`` reads. The output has
  its grid and levels.
* ``affine``: the fixed-to-moving affine in physical units, C order: a ``.npy`` or text
  file holding a 4x4 or 3x4 matrix, or its 12 or 16 values inline, row by row (default:
  identity). The moving image is sampled at ``affine(p + u(p))``.
* ``fixed_channel``, ``moving_channel``: the channel each image is matched on, for images
  with a ``c`` axis (default 0). The output has every channel of the moving image.
* ``levels``: fixed-image levels to solve on, coarse to fine, e.g. ``6,5,4`` (default:
  from the coarsest with at least 16 voxels on every axis to the finest with at most
  2^25). Each is matched against the moving level nearest its voxel size.
* ``iterations``: Adam steps per level, one value or one per level (default 100). ``0``
  skips the solve: the affine alone, which needs no GPU.
* ``smooth``: weight of the penalty on the field's gradient (default 1).
* ``grid``: control-point spacing of the field, in voxels of each level (default 4).
* ``window``: correlation window, voxels, odd (default 7).
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
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.registration import Grid, Level, Settings, solve
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource, _pick_level
from chunkmirage.sources.warp import _SPATIAL, FieldLevelSource, FramesSource
from chunkmirage.transforms import Affine, Displacements, Sequence, VectorField, simplify

log = logging.getLogger("chunkmirage")

_PARAMS = {
    "fixed",
    "affine",
    "fixed_channel",
    "moving_channel",
    "levels",
    "iterations",
    "smooth",
    "grid",
    "window",
    "show",
    "frames",
    "interpolation",
    "chunk",
    "device",
}
MAX_SOLVE_VOXELS = 1 << 25  # finest default level: its whole volume sits on the GPU
MIN_SOLVE_SIZE = 16  # coarsest default level: at least this many voxels on every axis
KEEP = 8  # solved fields remembered per process
_solved: OrderedDict[str, list[Grid]] = OrderedDict()
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


def _level(src: Source, lead: tuple[int, ...]) -> Level:
    info = src.info
    n = len(lead)
    box = Box(lead + (0,) * 3, tuple(v + 1 for v in lead) + info.shape[n:])
    return Level(
        src.read(box).reshape(info.shape[n:]),
        np.asarray(info.voxel_size[n:], dtype=float),
        np.asarray(info.translation[n:], dtype=float),
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


def _ints(q: dict, key: str) -> list[int] | None:
    return [int(v) for v in q[key].split(",")] if key in q else None


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
    if unknown := set(q) - _PARAMS:
        raise ValueError(
            f"unknown register:// parameters {sorted(unknown)}; allowed: {sorted(_PARAMS)}"
        )
    show = q.get("show", "image")
    if show not in ("image", "pair", "field"):
        raise ValueError(f"show must be image, pair or field, got {show!r}")

    mov = open_source(location, cache_bytes=cache_bytes)
    fix = open_source(q["fixed"], cache_bytes=cache_bytes)
    nm, nf = _n_lead(mov, location), _n_lead(fix, q["fixed"])
    minfo, finfo = mov.levels[0].info, fix.levels[0].info
    if tuple(minfo.axes[nm:]) != tuple(finfo.axes[nf:]):
        raise ValueError(f"spatial axes differ: moving {minfo.axes}, fixed {finfo.axes}")
    if tuple(minfo.units[nm:]) != tuple(finfo.units[nf:]):
        raise ValueError(
            f"spatial units differ: moving {minfo.units[nm:]}, fixed {finfo.units[nf:]}"
        )
    mlead = _lead(mov, nm, int(q.get("moving_channel", 0)), "moving")
    flead = _lead(fix, nf, int(q.get("fixed_channel", 0)), "fixed")
    affine = _affine(q.get("affine"))

    levels = _ints(q, "levels") or _default_levels(fix, nf)
    if any(not 0 <= i < len(fix.levels) for i in levels):
        raise ValueError(f"levels {levels}: the fixed image has levels 0..{len(fix.levels) - 1}")
    iters = _ints(q, "iterations") or [100]
    if len(iters) not in (1, len(levels)) or min(iters) < 0:
        raise ValueError(f"iterations needs 1 or {len(levels)} values >= 0 (one per level)")
    window = int(q.get("window", 7))
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd number, got {window}")
    settings = Settings(
        iterations=tuple(iters * len(levels) if len(iters) == 1 else iters),
        smooth=float(q.get("smooth", 1.0)),
        grid=float(q.get("grid", 4.0)),
        window=window,
    )
    if settings.grid <= 0 or settings.smooth < 0:
        raise ValueError("grid must be > 0 and smooth >= 0")
    n_frames = int(q["frames"]) if "frames" in q else None
    if n_frames is not None:
        if n_frames < 1:
            raise ValueError(f"frames must be at least 1, got {n_frames}")
        if show == "image" and "t" in minfo.axes and (minfo.axes[0] != "t" or minfo.shape[0] != 1):
            raise ValueError(
                f"{location}: frames need a moving image without a time axis, or with one time"
                f" point first; axes {minfo.axes}, shape {minfo.shape}"
            )

    labels = minfo.dtype.kind == "u" and minfo.dtype.itemsize >= 4
    interpolation = q.get("interpolation", "nearest" if labels else "linear")
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"interpolation must be linear or nearest, got {interpolation!r}")
    order = 0 if interpolation == "nearest" else 1
    chunk = _ints(q, "chunk")
    if chunk is not None and (len(chunk) != 3 or min(chunk) < 1):
        raise ValueError("chunk needs 3 positive sizes (z,y,x)")

    # Solve (or recall): moving levels nearest each fixed level's voxel size.
    mvox, fvox = _spatial_vox(mov, nm), _spatial_vox(fix, nf)
    mlevels = [int(np.argmin([np.abs(np.log(m / fvox[i])).sum() for m in mvox])) for i in levels]
    ident = {
        "fixed": [fix.levels[i].cache_key() for i in levels],
        "moving": [mov.levels[j].cache_key() for j in mlevels],
        "lead": [flead, mlead],
        "affine": np.round(affine, 12).tolist(),
        "settings": dataclasses.asdict(settings),
        "frames": n_frames,
    }
    solve_key = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    lo = np.asarray(finfo.translation[nf:], dtype=float) - fvox[0] / 2
    hi = lo + np.asarray(finfo.shape[nf:]) * fvox[0]

    def compute() -> list[Grid]:
        log.info("register: %s onto %s at levels %s", location, q["fixed"], levels)
        return solve(
            [_level(fix.levels[i], flead) for i in levels],
            [_level(mov.levels[j], mlead) for j in mlevels],
            affine,
            settings,
            box=(lo, hi),
            device=q.get("device", "auto"),
            snapshots=n_frames or 1,
        )

    with _solve_lock:
        if solve_key in _solved:
            _solved.move_to_end(solve_key)
        else:
            _solved[solve_key] = compute()
            while len(_solved) > KEEP:
                _solved.popitem(last=False)
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

    def level(i: int, k: int) -> Source:
        """Fixed level ``i`` through field ``k``."""
        fl = fix.levels[i].info
        space_chunk = tuple(chunk or fl.chunk_shape[nf:])
        grid = dataclasses.replace(_axes_of(fl, slice(nf, None)), chunk_shape=space_chunk)
        to_space = Affine.scale_translation(fvox[i], fl.translation[nf:])
        ident = f"register|{solve_key}|{k}|{i}|{show}|{order}|{space_chunk}"
        key = "register:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        if show == "field":
            info = _prepend(grid, ArrayInfo((3,), np.float32, (3,), (1.0,), ("",), ("c",)))
            return FieldLevelSource(fields[k], to_space, info, key)

        def mapping_for(j: int):
            ml = mov.levels[j].info
            to_index = Affine.scale_translation(mvox[j], ml.translation[nm:]).inverse()
            return simplify(Sequence([to_space, Displacements(fields[k]), to_moving, to_index]))

        j = _pick_level(grid.shape, mapping_for(0), mvox)
        ml = mov.levels[j].info
        info = _prepend(grid, _axes_of(ml, slice(nm)))
        registered = SceneLevelSource(mov.levels[j], mapping_for(j), info, nm, order, key)
        if show == "image":
            return registered
        dtype = np.result_type(fl.dtype, ml.dtype)
        pair = _prepend(grid, ArrayInfo((2,), dtype, (2,), (1.0,), ("",), ("c",)))
        return PairSource(fix.levels[i], flead, registered, mlead, pair, key + ":pair")

    out: list[Source] = []
    for i in range(len(fix.levels)):
        frames = [level(i, k) for k in range(len(grids))]
        if n_frames is None:
            out.append(frames[0])
            continue
        ident = "frames|" + "|".join(f.cache_key() for f in frames)
        key = "register:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        out.append(FramesSource(frames, show == "image" and "t" in minfo.axes, key))
    return MultiscaleSource(out, name=f"{mov.name or location}~registered")
