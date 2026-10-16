"""``warp://`` sources: an image resampled through a displacement field, on its own grid or
another's.

    warp://<image>?field=<source>&affine=<matrix>&target=<source>
    warp://<image>?field=swirl&angle=90&radius=200&centre=z,y,x&plane=y,x
    warp://<image>?field=swirls&count=8&seed=0&angle=90&radius=200&centre=z,y,x&spread=z,y,x
    warp://<image>?field=...&show=field
    warp://<image>?field=...&frames=24

(the parameters follow the last ``?``, so ``<image>`` may have a query of its own). The
first serves ``<image>`` (anything ``open_source`` reads) through a field read from any
other source: a ``(c, z, y, x)`` array whose components are a displacement, in the image's
units, at each voxel of the field's own grid, in the output's space. Another dataset of
the server (``dataset://<name>``, a pipeline's output: a block matcher's lattice, a model's
prediction), a stored zarr, a warp's own ``show=field``. A point ``x`` of the output is read
from the image at ``A(x + d(x))``, ``A`` the ``affine`` (identity by default), as
``register://`` places its points. The second and third twist the image by a swirl computed
from coordinates, or by several 3-D swirls at random places with random axes (3-D images
only), so nothing is stored and any parameter can change at any time. The output grid is
``target``'s (any source: its levels are the output's, each reading the image level that
suits it) or the image's own. ``show=field`` serves the displacement itself, components
first, in the image's units (for example as an RGB layer). ``frames`` (the swirls only)
adds a leading ``t`` axis of frames that twist from 0 to ``angle``, so a viewer animates
the swirl by playing ``t`` rather than by changing the URL. All are computed chunk by
chunk with the same resampler as ``scene://``, so a new parameter value (a REST ``PUT`` of
the new URL) shows up as soon as the viewer refetches, and the field's keys are part of
the output's, so an edit upstream of the field reaches it. A field or target URL with a
query of its own is percent-encoded (``urllib.parse.quote(url, safe="")``).

Query parameters:

* ``field``: the deformation (required). A source URL: displacements sampled from it, at
  the field level nearest each output level's voxel size. ``swirl``: one twist about an
  axis parallel to a spatial axis, fading with distance from the axis. ``swirls``:
  ``count`` twists, each about its own axis through its own centre and fading with distance
  from the centre, so each is a ball of twist; their displacements add.
* ``affine``: applied after the displacement: 12 numbers, 3 rows of 4 in C order
  (``z, y, x``), or 16 with the last row ``0 0 0 1``; for a 2-D image 6 or 9. Identity by
  default.
* ``target``: the source whose grid (levels, shape, voxel size, position) the output
  takes; the image's own by default.
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
* ``frames``: the swirls only: number of frames on a leading ``t`` axis; frame ``i``
  twists by ``angle * i / (frames - 1)``. An image's own ``t`` axis must have one time
  point, which the frames replace.
* ``interpolation``: ``linear`` (default) or ``nearest`` (default for uint32/uint64).
* ``chunk``: output chunk shape of the spatial axes, C order (default the grid's). Thin
  chunks such as ``8,128,128`` compute less for a viewer showing one plane.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source
from chunkmirage.sources.scene import SceneLevelSource, _pick_level
from chunkmirage.transforms import (
    Affine,
    Displacements,
    Sequence,
    SwirlField,
    Swirls,
    VectorField,
    simplify,
)

_PARAMS = {
    "count",
    "seed",
    "spread",
    "field",
    "affine",
    "target",
    "angle",
    "radius",
    "centre",
    "plane",
    "show",
    "interpolation",
    "frames",
    "chunk",
}
_SWIRL_PARAMS = {"angle", "radius", "centre", "plane", "count", "seed", "spread", "frames"}
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


def sampled_field(level: Source, key: str) -> VectorField:
    """The displacements stored in ``level``, a ``(c, z, y, x)`` array (components first, in
    the output's units), as a field over its own grid: sampled between its voxels linearly,
    held at its edges beyond them."""
    info = level.info

    def read(sl: tuple[slice, ...]) -> np.ndarray:
        start = tuple(s.start or 0 for s in sl)
        stop = tuple(n if s.stop is None else s.stop for s, n in zip(sl, info.shape))
        return level.read(Box(start, stop))

    return VectorField(read, info.shape, 0, Affine.scale_translation(info.voxel_size[1:], info.translation[1:]), key)


def _affine(value: str | None, spatial: list[str]) -> Affine:
    n = len(spatial)
    if value is None:
        return Affine.identity(n)
    v = [float(x) for x in value.split(",")]
    if len(v) == n * (n + 1):
        m = np.asarray(v).reshape(n, n + 1)
    elif len(v) == (n + 1) ** 2:
        m = np.asarray(v).reshape(n + 1, n + 1)[:n]
    else:
        raise ValueError(
            f"affine needs {n * (n + 1)} numbers ({n} rows of {n + 1}, C order {','.join(spatial)}) "
            f"or {(n + 1) ** 2} with the last row 0 ... 1, got {len(v)}"
        )
    return Affine(m)


def _floats(q: dict, key: str) -> list[float] | None:
    return [float(v) for v in q[key].split(",")] if key in q else None


def _n_lead(info: ArrayInfo, url: str) -> int:
    names = list(info.axes)
    n = 0
    while n < len(names) and names[n] not in _SPATIAL:
        n += 1
    if n == len(names) or any(a not in _SPATIAL for a in names[n:]):
        raise ValueError(f"{url}: axes {names} must end in spatial axes (z, y, x)")
    return n


def _spatial(info: ArrayInfo, k: int, chunk: tuple[int, ...] | None) -> ArrayInfo:
    """The grid of ``info``'s spatial axes, those from ``k`` on, chunked as ``chunk`` if given."""
    return ArrayInfo(
        shape=info.shape[k:],
        dtype=info.dtype,
        chunk_shape=chunk or info.chunk_shape[k:],
        voxel_size=info.voxel_size[k:],
        units=info.units[k:],
        axes=info.axes[k:],
        translation=info.translation[k:],
    )


def open_warp(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    # The warp's parameters follow the LAST '?': the image URL may have its own query
    # (synthetic://blobs?shape=...?field=swirl&angle=90).
    location, _, query = url[len("warp://") :].rpartition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if "field" not in q:
        raise ValueError(
            "warp:// needs a field after the image URL: warp://<image>?field=<source URL>, "
            "field=swirl or field=swirls (the warp's parameters follow the last '?')"
        )
    if unknown := set(q) - _PARAMS:
        raise ValueError(
            f"unknown warp:// parameters {sorted(unknown)}; allowed: {sorted(_PARAMS)}"
        )
    kind = q["field"]
    procedural = kind in ("swirl", "swirls")
    if procedural:
        only = {"swirl": {"plane"}, "swirls": {"count", "seed", "spread"}}
        if misplaced := set(q) & only["swirls" if kind == "swirl" else "swirl"]:
            raise ValueError(f"{sorted(misplaced)} do not apply to field={kind}")
    elif misplaced := set(q) & _SWIRL_PARAMS:
        raise ValueError(f"{sorted(misplaced)} apply to field=swirl or swirls, not to a field read from {kind}")
    show = q.get("show", "image")
    if show not in ("image", "field"):
        raise ValueError(f"show must be image or field, got {show!r}")

    src = open_source(location, cache_bytes=cache_bytes, cache=cache)
    info0 = src.levels[0].info
    names = list(info0.axes)
    n_lead = _n_lead(info0, location)
    spatial = names[n_lead:]
    n = len(spatial)
    shape = np.asarray(info0.shape[n_lead:], dtype=float)
    vox = np.asarray(info0.voxel_size[n_lead:], dtype=float)
    origin = np.asarray(info0.translation[n_lead:], dtype=float)
    affine = _affine(q.get("affine"), spatial)
    affine_key = "identity" if q.get("affine") is None else str(np.round(affine.matrix, 12).tolist())

    target = open_source(q["target"], cache_bytes=cache_bytes, cache=cache) if "target" in q else None
    if target is not None:
        tinfo = target.levels[0].info
        t_lead = _n_lead(tinfo, q["target"])
        if list(tinfo.axes[t_lead:]) != spatial:
            raise ValueError(f"target {q['target']}: axes {tinfo.axes[t_lead:]} differ from the image's {spatial}")

    if procedural:
        extent = shape * vox
        centre = _floats(q, "centre") or list(origin + (shape - 1) / 2 * vox)
        if len(centre) != n:
            raise ValueError(f"centre needs {n} values ({','.join(spatial)})")
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
            if n != 3:
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
    else:
        field_ms = open_source(kind, cache_bytes=cache_bytes, cache=cache)
        finfo = field_ms.levels[0].info
        if finfo.ndim != n + 1 or finfo.shape[0] != n or list(finfo.axes[1:]) != spatial:
            raise ValueError(
                f"field {kind}: a displacement field is (c, {', '.join(spatial)}) with c = {n} "
                f"components (z, y, x displacements), got axes {finfo.axes} of shape {finfo.shape}"
            )

    labels = info0.dtype.kind == "u" and info0.dtype.itemsize >= 4
    interpolation = q.get("interpolation", "nearest" if labels else "linear")
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"interpolation must be linear or nearest, got {interpolation!r}")
    order = 0 if interpolation == "nearest" else 1
    chunk = tuple(int(v) for v in q["chunk"].split(",")) if "chunk" in q else None
    if chunk is not None and (len(chunk) != n or min(chunk) < 1):
        raise ValueError(f"chunk needs {n} positive sizes ({','.join(spatial)})")

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
    fields = [make_field(f) for f in fractions] if procedural else []
    img_vox = [np.asarray(lvl.info.voxel_size[n_lead:], dtype=float) for lvl in src.levels]

    def field_for(grid: ArrayInfo, k: int) -> VectorField | SwirlField | Swirls:
        if procedural:
            return fields[k]
        # the stored field's level nearest this level's voxels: coarse for a coarse view
        j = field_ms.nearest_level(grid.voxel_size)
        return sampled_field(field_ms.levels[j], f"field|{field_ms.levels[j].cache_key()}")

    def level(i: int, k: int) -> Source:
        """Output level ``i`` (the target's, or the image's own) through field ``k``."""
        base, k_lead = (src.levels[i].info, n_lead) if target is None else (target.levels[i].info, t_lead)
        grid = _spatial(base, k_lead, chunk)
        to_space = Affine.scale_translation(grid.voxel_size, grid.translation)
        field = field_for(grid, k)
        grid_key = f"{grid.shape}|{grid.voxel_size}|{grid.translation}|{grid.chunk_shape}"
        if show == "field":
            finfo = ArrayInfo(
                shape=(n, *grid.shape),
                dtype=np.float32,
                chunk_shape=(n, *grid.chunk_shape),
                voxel_size=(1.0, *grid.voxel_size),
                units=("", *grid.units),
                axes=("c", *grid.axes),
                translation=(0.0, *grid.translation),
            )
            ident = f"warp|field|{field.key}|{grid_key}|{i}"
            return FieldLevelSource(field, to_space, finfo, "warp:" + hashlib.sha1(ident.encode()).hexdigest()[:12])

        def mapping_for(j: int, displaced: bool = True):
            # grid index -> space -> displaced -> affine -> index of image level j
            to_index = Affine.scale_translation(img_vox[j], src.levels[j].info.translation[n_lead:]).inverse()
            steps = [to_space, Displacements(field), affine, to_index] if displaced else [to_space, affine, to_index]
            return simplify(Sequence(steps))

        # on its own grid the image's level i is read; on a target's, the image level
        # whose voxels suit the output's through the mapping (a coarse view reads a coarse
        # level). A sampled field is left out of that choice: sampling it would compute
        # the dataset it comes from before anything is viewed (so register:// chooses
        # through its solved field, never a refined one); a swirl costs nothing.
        j = i if target is None else _pick_level(grid.shape, mapping_for(0, displaced=procedural), img_vox)
        lvl = src.levels[j]
        li = lvl.info
        info = ArrayInfo(
            shape=li.shape[:n_lead] + grid.shape,
            dtype=li.dtype,
            chunk_shape=li.chunk_shape[:n_lead] + grid.chunk_shape,
            voxel_size=li.voxel_size[:n_lead] + grid.voxel_size,
            units=li.units[:n_lead] + grid.units,
            axes=li.axes[:n_lead] + grid.axes,
            translation=li.translation[:n_lead] + grid.translation,
            kind=li.kind,
        )
        ident = f"warp|{lvl.cache_key()}|{field.key}|{affine_key}|{grid_key}|{order}|{i}"
        return SceneLevelSource(lvl, mapping_for(j), info, n_lead, order, "warp:" + hashlib.sha1(ident.encode()).hexdigest()[:12])

    levels: list[Source] = []
    for i in range(len(src.levels) if target is None else len(target.levels)):
        frames = [level(i, k) for k in range(len(fractions))]
        if n_frames is None:
            levels.append(frames[0])
            continue
        ident = "frames|" + "|".join(f.cache_key() for f in frames)
        key = "warp:" + hashlib.sha1(ident.encode()).hexdigest()[:12]
        levels.append(FramesSource(frames, "t" in names and show == "image", key))
    return MultiscaleSource(levels, name=f"{src.name or location}~{'swirl' if procedural else 'warped'}")
