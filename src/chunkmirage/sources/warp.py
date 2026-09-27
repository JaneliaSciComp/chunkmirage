"""``warp://`` sources: an image resampled through a procedural deformation, on its own grid.

    warp://<image>?field=swirl&angle=90&radius=200&centre=z,y,x&plane=y,x
    warp://<image>?field=swirls&count=8&seed=0&angle=90&radius=200&centre=z,y,x&spread=z,y,x
    warp://<image>?field=...&show=field
    warp://<image>?field=...&frames=24

(the parameters follow the last ``?``, so ``<image>`` may have a query of its own). The
first serves ``<image>`` (anything ``open_source`` reads) twisted by a swirl computed
from coordinates; the second by several 3-D swirls at random places with random axes
(3-D images only); the third serves the displacement itself, components first, in the
image's units (for example as an RGB layer); the fourth adds a leading ``t`` axis of frames
that twist from 0 to ``angle``, so a viewer animates the swirl by playing ``t`` rather
than by changing the URL. All are computed chunk by chunk with the same resampler as
``scene://`` and nothing is stored, so a new parameter value (a REST ``PUT`` of the new
URL) shows up as soon as the viewer refetches.

Query parameters:

* ``field``: the deformation (required). ``swirl``: one twist about an axis parallel to
  a spatial axis, fading with distance from the axis. ``swirls``: ``count`` twists, each
  about its own axis through its own centre and fading with distance from the centre, so
  each is a ball of twist; their displacements add.
* ``angle``: twist on the axis, in degrees (default 90). For ``swirls``, each turns by
  between half and all of it, clockwise or anticlockwise.
* ``radius``: distance over which the twist fades, image units (default a quarter of the
  smaller in-plane extent for ``swirl``, a fifth of the smallest extent for ``swirls``,
  each of which gets between half and all of it).
* ``centre``: a point on the axis, image units, C order (default the volume centre). For
  ``swirls``, the middle of the box their centres are drawn from.
* ``plane``: ``swirl`` only: the two axes that turn (default the last two, e.g. ``y,x``).
* ``count``, ``seed``, ``spread``: ``swirls`` only: how many (default 8), the random seed
  (default 0) and the half-size of the box around ``centre`` their centres fall in, one
  value or ``z,y,x`` (default half the volume, so the whole volume). Axes point in
  uniformly random directions.
* ``show``: ``image`` (default) or ``field``.
* ``frames``: number of frames on a leading ``t`` axis; frame ``i`` twists by
  ``angle * i / (frames - 1)``. An image's own ``t`` axis must have one time point, which
  the frames replace.
* ``interpolation``: ``linear`` (default) or ``nearest`` (default for uint32/uint64).
* ``chunk``: output chunk shape of the spatial axes, C order (default the image's). Thin
  chunks such as ``8,128,128`` compute less for a viewer showing one plane.
"""

from __future__ import annotations

import dataclasses
import hashlib
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource
from chunkmirage.transforms import Affine, Displacements, Sequence, SwirlField, Swirls, simplify

_PARAMS = {
    "count",
    "seed",
    "spread",
    "field",
    "angle",
    "radius",
    "centre",
    "plane",
    "show",
    "interpolation",
    "frames",
    "chunk",
}
_SPATIAL = ("z", "y", "x")


class FieldLevelSource(Source):
    """The displacement vectors of ``field`` at the voxel centres of one level."""

    def __init__(self, field, to_space: Affine, info: ArrayInfo, key: str):
        self.field = field
        self.to_space = to_space
        self._info = info
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        space = Box(box.start[1:], box.stop[1:])
        pts = np.indices(space.shape, dtype=float).reshape(len(space.shape), -1).T
        d = self.field.sample(self.to_space.apply(pts + np.asarray(space.start, dtype=float)))
        vectors = d.T.reshape(-1, *space.shape).astype(self._info.dtype)
        return vectors[box.start[0] : box.stop[0]]


class FramesSource(Source):
    """Frames stacked on a new leading ``t`` axis, one per source in ``frames``, which
    all have the same info. With ``replace_t`` their own leading ``t`` axis (one time
    point) is the one the frames replace."""

    def __init__(self, frames: list[Source], replace_t: bool, key: str):
        self.frames = frames
        self.replace_t = replace_t
        self._key = key
        f = frames[0].info
        k = 1 if replace_t else 0
        self._info = ArrayInfo(
            shape=(len(frames), *f.shape[k:]),
            dtype=f.dtype,
            chunk_shape=(1, *f.chunk_shape[k:]),
            voxel_size=(1.0, *f.voxel_size[k:]),
            units=("", *f.units[k:]),
            axes=("t", *f.axes[k:]),
            translation=(0.0, *f.translation[k:]),
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        start, stop = box.start[1:], box.stop[1:]
        if self.replace_t:
            start, stop = (0, *start), (1, *stop)
        blocks = [self.frames[i].read(Box(start, stop)) for i in range(box.start[0], box.stop[0])]
        return np.concatenate(blocks) if self.replace_t else np.stack(blocks)


def _floats(q: dict, key: str) -> list[float] | None:
    return [float(v) for v in q[key].split(",")] if key in q else None


def open_warp(url: str, *, cache_bytes: int = 0) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    # The warp's parameters follow the LAST '?': the image URL may have its own query
    # (synthetic://blobs?shape=...?field=swirl&angle=90).
    location, _, query = url[len("warp://") :].rpartition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if "field" not in q:
        raise ValueError(
            "warp:// needs field=swirl or field=swirls after the image URL: "
            "warp://<image>?field=... (the warp's parameters follow the last '?')"
        )
    if unknown := set(q) - _PARAMS:
        raise ValueError(
            f"unknown warp:// parameters {sorted(unknown)}; allowed: {sorted(_PARAMS)}"
        )
    kind = q["field"]
    if kind not in ("swirl", "swirls"):
        raise ValueError(f"warp:// needs field=swirl or field=swirls, got {kind!r}")
    only = {"swirl": {"plane"}, "swirls": {"count", "seed", "spread"}}
    if misplaced := set(q) & only["swirls" if kind == "swirl" else "swirl"]:
        raise ValueError(f"{sorted(misplaced)} do not apply to field={kind}")
    show = q.get("show", "image")
    if show not in ("image", "field"):
        raise ValueError(f"show must be image or field, got {show!r}")

    src = open_source(location, cache_bytes=cache_bytes)
    info0 = src.levels[0].info
    names = list(info0.axes)
    n_lead = 0
    while n_lead < len(names) and names[n_lead] not in _SPATIAL:
        n_lead += 1
    spatial = names[n_lead:]
    if not spatial or any(a not in _SPATIAL for a in spatial):
        raise ValueError(f"{location}: axes {names} must end in spatial axes (z, y, x)")
    shape = np.asarray(info0.shape[n_lead:], dtype=float)
    vox = np.asarray(info0.voxel_size[n_lead:], dtype=float)
    origin = np.asarray(info0.translation[n_lead:], dtype=float)

    extent = shape * vox
    centre = _floats(q, "centre") or list(origin + (shape - 1) / 2 * vox)
    if len(centre) != len(spatial):
        raise ValueError(f"centre needs {len(spatial)} values ({','.join(spatial)})")
    angle = float(q.get("angle", 90))
    if kind == "swirl":
        plane_names = q.get("plane", ",".join(spatial[-2:])).split(",")
        if len(plane_names) != 2 or any(p not in spatial for p in plane_names):
            raise ValueError(f"plane must name two of the spatial axes {spatial}")
        plane = tuple(spatial.index(p) for p in plane_names)
        radius = float(q.get("radius", 0.25 * min(extent[plane[0]], extent[plane[1]])))

        def make_field(fraction: float):
            return SwirlField(centre, radius, angle * fraction, plane)
    else:
        if len(spatial) != 3:
            raise ValueError(f"{location}: swirls need three spatial axes, got {spatial}")
        count = int(q.get("count", 8))
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")
        spread = _floats(q, "spread") or list(extent / 2)
        if len(spread) not in (1, 3):
            raise ValueError("spread needs one value or three (z,y,x)")
        radius = float(q.get("radius", 0.2 * min(extent)))
        rng = np.random.default_rng(int(q.get("seed", 0)))
        centres = np.asarray(centre) + rng.uniform(-1, 1, (count, 3)) * np.asarray(spread)
        axes = rng.normal(size=(count, 3))
        radii = radius * rng.uniform(0.5, 1, count)
        twists = angle * rng.uniform(0.5, 1, count) * rng.choice([-1, 1], count)

        def make_field(fraction: float):
            return Swirls(centres, axes, radii, twists * fraction)

    labels = info0.dtype.kind == "u" and info0.dtype.itemsize >= 4
    interpolation = q.get("interpolation", "nearest" if labels else "linear")
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"interpolation must be linear or nearest, got {interpolation!r}")
    order = 0 if interpolation == "nearest" else 1
    chunk = tuple(int(v) for v in q["chunk"].split(",")) if "chunk" in q else None
    if chunk is not None and (len(chunk) != len(spatial) or min(chunk) < 1):
        raise ValueError(f"chunk needs {len(spatial)} positive sizes ({','.join(spatial)})")

    n_frames = int(q["frames"]) if "frames" in q else None
    if n_frames is not None:
        if n_frames < 1:
            raise ValueError(f"frames must be at least 1, got {n_frames}")
        if "t" in names and (names[0] != "t" or info0.shape[0] != 1):
            raise ValueError(
                f"{location}: frames need an image without a time axis, or with one time "
                f"point first; axes {names}, shape {info0.shape}"
            )
        fractions = [i / (n_frames - 1) for i in range(n_frames)] if n_frames > 1 else [1.0]
    else:
        fractions = [1.0]
    fields = [make_field(f) for f in fractions]

    def level(lvl: Source, i: int, field) -> Source:
        info = lvl.info
        if chunk is not None:
            info = dataclasses.replace(info, chunk_shape=(*info.chunk_shape[:n_lead], *chunk))
        to_space = Affine.scale_translation(info.voxel_size[n_lead:], info.translation[n_lead:])
        ident = f"warp|{lvl.cache_key()}|{field.key}|{show}|{order}|{i}"
        key = "warp:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        if show == "field":
            n = len(spatial)
            finfo = ArrayInfo(
                shape=(n, *info.shape[n_lead:]),
                dtype=np.float32,
                chunk_shape=(n, *info.chunk_shape[n_lead:]),
                voxel_size=(1.0, *info.voxel_size[n_lead:]),
                units=("", *info.units[n_lead:]),
                axes=("c", *spatial),
                translation=(0.0, *info.translation[n_lead:]),
            )
            return FieldLevelSource(field, to_space, finfo, key)
        # grid index -> space -> swirl -> space -> index of the same level
        mapping = simplify(Sequence([to_space, Displacements(field), to_space.inverse()]))
        return SceneLevelSource(lvl, mapping, info, n_lead, order, key)

    levels: list[Source] = []
    for i, lvl in enumerate(src.levels):
        frames = [level(lvl, i, f) for f in fields]
        if n_frames is None:
            levels.append(frames[0])
            continue
        ident = "frames|" + "|".join(f.cache_key() for f in frames)
        key = "warp:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        levels.append(FramesSource(frames, "t" in names and show == "image", key))
    return MultiscaleSource(levels, name=f"{src.name or location}~swirl")
