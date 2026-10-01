"""tensorstore-backed sources (zarr v2/v3, n5, neuroglancer precomputed; file/s3/gcs/http).

tensorstore does all I/O in C++ threads (GIL released), supports async reads, and has its
own byte-bounded ``cache_pool`` so the *raw* chunks of a remote source are cached without
any extra work on our side.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

import numpy as np
import tensorstore as ts

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source


def _open_kvstore(url_or_path: str) -> ts.KvStore:
    """Open a kvstore rooted at a *directory*; tensorstore concatenates keys literally, so
    the root must end in '/'."""
    root = url_or_path if url_or_path.endswith("/") else url_or_path + "/"
    if re.match(r"^(s3|gs|http|https|file|memory)://", root):
        return ts.KvStore.open(root).result()
    return ts.KvStore.open({"driver": "file", "path": root}).result()


def _read_json(kv: ts.KvStore, key: str) -> dict | None:
    r = kv.read(key).result()
    if r.state != "value":
        return None
    return json.loads(bytes(r.value).decode())


def _detect_driver(kv: ts.KvStore) -> str | None:
    zj = _read_json(kv, "zarr.json")
    if zj is not None:
        return "zarr3-group" if zj.get("node_type") == "group" else "zarr3"
    if _read_json(kv, ".zarray") is not None:
        return "zarr"
    if _read_json(kv, ".zgroup") is not None:
        return "zarr-group"
    if _read_json(kv, "info") is not None:
        return "neuroglancer_precomputed"
    attrs = _read_json(kv, "attributes.json")
    if attrs is not None:
        return "n5" if "dataType" in attrs or "dimensions" in attrs else "n5-group"
    return None


def _node_attrs(kv: ts.KvStore) -> dict:
    """User attributes of a zarr v2/v3 or N5 node (array or group); OME 0.5 is unwrapped."""
    for key in (".zattrs", "attributes.json"):
        attrs = _read_json(kv, key)
        if attrs is not None:
            return attrs
    attrs = (_read_json(kv, "zarr.json") or {}).get("attributes", {})
    return attrs.get("ome", attrs)


def _split_parent(path: str) -> tuple[str | None, str]:
    """``.../group/s0`` -> (``.../group``, ``s0``); (None, name) if there is no parent."""
    head, _, name = path.rstrip("/").rpartition("/")
    return (head if head and not head.endswith(":/") else None), name


_contexts: dict[int, ts.Context] = {}
_contexts_lock = threading.Lock()


def shared_context(cache_bytes: int) -> ts.Context:
    """One tensorstore context per cache budget, shared by every source this process opens.
    Its cache pool holds decoded chunks, and tensorstore shares them between stores opened in
    one context: the levels of an image, two opens of it (a before and an after dataset, or
    the images a registration reads while its output serves them too) and several images
    all draw on one budget, rather than each array on a budget and a copy of its own."""
    with _contexts_lock:
        ctx = _contexts.get(int(cache_bytes))
        if ctx is None:
            ctx = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_bytes)}})
            _contexts[int(cache_bytes)] = ctx
        return ctx


def open_tensorstore(
    path: str,
    *,
    cache_bytes: int = 0,
    context: ts.Context | None = None,
    driver: str | None = None,
    scale_index: int | None = None,
) -> ts.TensorStore:
    """Open a single array with tensorstore, auto-detecting the format."""
    kv = _open_kvstore(path)
    driver = driver or _detect_driver(kv)
    if driver is None or driver.endswith("-group"):
        raise ValueError(f"{path}: not an array (driver detected: {driver})")
    spec: dict[str, Any] = {"driver": driver, "kvstore": kv.spec().to_json()}
    if driver == "neuroglancer_precomputed" and scale_index is not None:
        spec["scale_index"] = scale_index
    if context is None:
        context = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_bytes)}})
    return ts.open(spec, read=True, write=False, context=context).result()


# Per-level metadata is a dict with any of ``voxel_size``, ``units``, ``translation``, ``axes``
# (C order, length ndim); missing keys fall back to defaults in ``from_path``.


def _per_axis(vals, ndim: int, fill, reverse: bool = False) -> tuple | None:
    """Normalise a per-axis list to ``ndim`` C-order entries. Lists that only cover the
    spatial axes are left-padded with ``fill`` (leading channel/time axes); ``fill=None``
    means the list must already have ``ndim`` entries."""
    if not isinstance(vals, (list, tuple)):
        return None
    vals = list(vals)[::-1] if reverse else list(vals)
    if len(vals) > ndim or (fill is None and len(vals) != ndim):
        return None
    return tuple([fill] * (ndim - len(vals)) + vals)


def _floats(vals) -> tuple[float, ...] | None:
    return None if vals is None else tuple(float(v) for v in vals)


def _multiscale_datasets(group_attrs: dict) -> list[dict]:
    ms = group_attrs.get("multiscales")
    if isinstance(ms, dict):  # OME-Zarr 0.6 drafts
        ms = [ms]
    if not isinstance(ms, list) or not ms or not isinstance(ms[0], dict):
        return []
    return [d for d in ms[0].get("datasets", []) if isinstance(d, dict) and "path" in d]


def _find_dataset(group_attrs: dict, name: str) -> dict | None:
    """The ``multiscales[0].datasets`` entry whose ``path`` is ``name`` (never by position)."""
    for ds in _multiscale_datasets(group_attrs):
        if str(ds["path"]).strip("/") == name:
            return ds
    return None


def _transform_metadata(t: dict, ndim: int) -> dict:
    """COSEM-style ``transform`` {axes, scale, translate, units}. Lists are C order (the
    OpenOrganelle N5s write ``axes: [z, y, x]``) unless ``ordering`` is ``"F"``."""
    rev = t.get("ordering") == "F"
    meta = {
        "voxel_size": _floats(_per_axis(t.get("scale"), ndim, 1.0, rev)),
        "translation": _floats(_per_axis(t.get("translate"), ndim, 0.0, rev)),
        "units": _per_axis(t.get("units"), ndim, "", rev),
        "axes": _per_axis(t.get("axes"), ndim, None, rev),
    }
    return {k: v for k, v in meta.items() if v is not None}


def _n5_scale_metadata(attrs: dict, ndim: int, group_attrs: dict, name: str) -> dict:
    """Voxel size / units / translation / axes for an N5 level, in priority order:
    the level's ``transform``; the parent group's ``multiscales[].datasets[].transform``
    matched by path; ``pixelResolution``/``resolution`` (level, else group) times the
    level's ``downsamplingFactors``, plus ``offset``. Plain N5 lists are x-first."""
    if isinstance(attrs.get("transform"), dict):
        return _transform_metadata(attrs["transform"], ndim)
    ds = _find_dataset(group_attrs, name)
    if ds is not None and isinstance(ds.get("transform"), dict):
        return _transform_metadata(ds["transform"], ndim)
    base = attrs if ("pixelResolution" in attrs or "resolution" in attrs) else group_attrs
    res, units = None, base.get("units")
    if isinstance(base.get("pixelResolution"), dict):
        res = base["pixelResolution"].get("dimensions")
        if base["pixelResolution"].get("unit") and isinstance(res, list):
            units = [base["pixelResolution"]["unit"]] * len(res)
    elif isinstance(base.get("resolution"), list):
        res = base["resolution"]
    meta: dict = {}
    if isinstance(res, list):
        factors = attrs.get("downsamplingFactors")
        if isinstance(factors, list) and len(factors) == len(res):
            res = [float(r) * float(f) for r, f in zip(res, factors)]
        meta["voxel_size"] = _floats(_per_axis(res, ndim, 1.0, reverse=True))
    meta["units"] = _per_axis(units, ndim, "", reverse=True)
    meta["translation"] = _floats(_per_axis(attrs.get("offset"), ndim, 0.0, reverse=True))
    meta["axes"] = _per_axis(attrs.get("axes") or group_attrs.get("axes"), ndim, None, True)
    return {k: v for k, v in meta.items() if v is not None}


def _ome_transforms(cts, ndim: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(scale, translation) of an OME ``coordinateTransformations`` list; identity if absent.
    Scales and translations compose in order, so ``index * scale + translation`` holds
    whatever their order; a 0.6 ``sequence`` is unpacked. Other types are ignored here
    (``scene://`` sources apply them)."""
    flat: list = []
    for ct in cts if isinstance(cts, list) else []:
        if isinstance(ct, dict) and ct.get("type") == "sequence":
            flat.extend(ct.get("transformations") or [])
        else:
            flat.append(ct)
    scale, trans = np.ones(ndim), np.zeros(ndim)
    for ct in flat:
        if not isinstance(ct, dict):
            continue
        if ct.get("type") == "scale" and len(ct.get("scale", [])) == ndim:
            s = np.asarray(_floats(ct["scale"]))
            scale, trans = scale * s, trans * s
        elif ct.get("type") == "translation" and len(ct.get("translation", [])) == ndim:
            trans = trans + np.asarray(_floats(ct["translation"]))
    return tuple(float(v) for v in scale), tuple(float(v) for v in trans)


def _ome_scale_metadata(group_attrs: dict, name: str, ndim: int) -> dict | None:
    """Voxel size / units / translation / axes from OME-NGFF ``multiscales`` (C order),
    for the dataset whose ``path`` is ``name``; None if the group doesn't list it.

    Up to 0.5, the optional multiscale-level ``coordinateTransformations`` apply to every
    dataset, after its own: world = s_top * (s_ds * index + t_ds) + t_top. In 0.6 they map
    to *other* named coordinate systems instead, so they are not applied here (a
    ``scene://`` source resamples into those); axes come from the intrinsic system.
    """
    ds = _find_dataset(group_attrs, name)
    if ds is None:
        return None
    ms = group_attrs["multiscales"]
    ms = ms if isinstance(ms, dict) else ms[0]
    systems = ms.get("coordinateSystems")
    axes = ms.get("axes", [])
    if isinstance(systems, list) and systems:
        out = next(iter(ds.get("coordinateTransformations") or [{}]), {}).get("output")
        out = out.get("name") if isinstance(out, dict) else out
        cs = next((c for c in systems if c.get("name") == out), systems[0])
        axes = cs.get("axes", [])
    meta: dict = {}
    if len(axes) == ndim:
        meta["units"] = tuple(a.get("unit", "") if isinstance(a, dict) else "" for a in axes)
        meta["axes"] = tuple(a.get("name") if isinstance(a, dict) else str(a) for a in axes)
    s_ds, t_ds = _ome_transforms(ds.get("coordinateTransformations"), ndim)
    top = None if systems else ms.get("coordinateTransformations")
    s_top, t_top = _ome_transforms(top, ndim)
    meta["voxel_size"] = tuple(a * b for a, b in zip(s_ds, s_top))
    meta["translation"] = tuple(t * s + o for t, s, o in zip(t_ds, s_top, t_top))
    return meta


def _zarr_array_metadata(attrs: dict, ndim: int) -> dict:
    """Legacy per-array attributes (C order): funlib-style ``resolution``/``voxel_size``,
    ``offset``, ``units``, ``axis_names``, or a COSEM-style ``transform``."""
    if isinstance(attrs.get("transform"), dict):
        return _transform_metadata(attrs["transform"], ndim)
    res = attrs.get("voxel_size", attrs.get("resolution"))
    units = attrs.get("units")
    if isinstance(units, str) and isinstance(res, list):
        units = [units] * len(res)
    meta = {
        "voxel_size": _floats(_per_axis(res, ndim, 1.0)),
        "translation": _floats(_per_axis(attrs.get("offset"), ndim, 0.0)),
        "units": _per_axis(units, ndim, ""),
        "axes": _per_axis(attrs.get("axis_names"), ndim, None),
    }
    return {k: v for k, v in meta.items() if v is not None}


_SECONDS = {"day": 86400.0, "hour": 3600.0, "minute": 60.0, "second": 1.0}


def _cf_unit(units: str) -> tuple[str, float]:
    """A CF coordinate's ``units`` as (unit, factor to it): ``days since ...`` and the like
    become seconds, degrees of latitude or longitude unitless (no viewer has degrees)."""
    u = units.strip().lower()
    if m := re.match(r"(day|hour|minute|second)s?\s+since\b", u):
        return "s", _SECONDS[m.group(1)]
    if u.startswith("degree"):
        return "", 1.0
    return units.strip(), 1.0


def _sig(v: float, digits: int) -> float:
    return float(f"{v:.{digits}g}")


def _cf_coordinates(parent: str, dims: list[str], shape: tuple[int, ...]) -> dict:
    """Voxel size, translation and units from an xarray-style group's coordinate arrays:
    the 1-D array named after each dimension, where it is evenly spaced and increasing
    (checked end to end from its first two and last values, so a long one costs two chunk
    reads). Other dimensions keep the defaults."""
    vs, tr, un = [1.0] * len(dims), [0.0] * len(dims), [""] * len(dims)
    for a, (dim, n) in enumerate(zip(dims, shape)):
        try:
            kv = _open_kvstore(f"{parent}/{dim}")
            coord = open_tensorstore(f"{parent}/{dim}", context=shared_context(0))
        except Exception:
            continue
        if coord.rank != 1 or coord.shape[0] != n or n < 2:
            continue
        first = np.asarray(coord[0:2].read().result(), dtype=np.float64)
        last = float(np.asarray(coord[n - 1].read().result()))
        step = (last - first[0]) / (n - 1)  # the whole span: float32 coordinates are coarse
        if step <= 0 or abs(first[1] - first[0] - step) > 0.01 * step:
            continue
        unit, f = _cf_unit(str(_node_attrs(kv).get("units", "")))
        # to the coordinates' own precision: float32 degrees 0.01 apart are not 0.0099999998
        digits = 6 if coord.dtype.numpy_dtype.itemsize <= 4 else 12
        vs[a], tr[a], un[a] = _sig(step * f, digits), _sig(first[0] * f, digits), unit
    return {"voxel_size": tuple(vs), "translation": tuple(tr), "units": tuple(un)}


def _cf_decoding(attrs: dict, store: ts.TensorStore) -> tuple[float, float, float | None] | None:
    """(scale_factor, add_offset, missing value) of a CF-packed array (stored integers that
    mean ``value * scale_factor + add_offset``), else None. The missing value is
    ``_FillValue``, else the array's fill value."""
    if "scale_factor" not in attrs and "add_offset" not in attrs:
        return None
    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    if fill is None and store.fill_value is not None:
        fill = np.asarray(store.fill_value).item()
    return float(attrs.get("scale_factor", 1.0)), float(attrs.get("add_offset", 0.0)), fill


def _precomputed_scale_metadata(info: dict, scale_index: int | None, ndim: int) -> dict:
    """``resolution`` (nm) and ``voxel_offset`` (voxels) of one scale, x-first -> C order."""
    try:
        sc = info["scales"][scale_index or 0]
        res = [float(v) for v in sc["resolution"]]
    except (KeyError, IndexError, TypeError):
        return {}
    off = [float(o) * r for o, r in zip(sc.get("voxel_offset", [0] * len(res)), res)]
    lead = ndim - len(res)  # the trailing channel axis is leading once transposed
    return {
        "voxel_size": (1.0,) * lead + tuple(res[::-1]),
        "units": ("",) * lead + ("nm",) * len(res),
        "translation": (0.0,) * lead + tuple(off[::-1]),
    }


class TensorStoreSource(Source):
    """One array. ``decode`` is a CF-packed array's (scale, offset, missing value): reads
    then return ``stored * scale + offset`` as float32, the missing value as NaN."""

    def __init__(
        self,
        store: ts.TensorStore,
        info: ArrayInfo,
        key: str,
        decode: tuple[float, float, float | None] | None = None,
    ):
        self.store = store
        self._info = info
        self._key = key
        self.decode = decode

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def _decoded(self, raw) -> np.ndarray:
        if self.decode is None:
            return np.asarray(raw, dtype=self._info.dtype)
        scale, offset, missing = self.decode
        raw = np.asarray(raw)
        out = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
        if missing is not None:
            out[raw == missing] = np.nan
        return out

    def read(self, box: Box) -> np.ndarray:
        return self._decoded(self.store[box.slices()].read().result())

    async def read_async(self, box: Box) -> np.ndarray:
        return self._decoded(await self.store[box.slices()].read())

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        cache_bytes: int = 0,
        scale_index: int | None = None,
        voxel_size=None,
        units=None,
        translation=None,
        axes=None,
    ) -> TensorStoreSource:
        kv = _open_kvstore(path)
        driver = _detect_driver(kv)
        store = open_tensorstore(
            path, context=shared_context(cache_bytes), driver=driver, scale_index=scale_index
        )
        if driver in ("n5", "neuroglancer_precomputed"):
            # Both drivers expose dimensions x-first, (x, y, z[, c]); we are C order.
            store = store[ts.d[:].transpose[::-1]]
        ndim = store.rank
        # Arrays are 0-based here; a non-zero origin (precomputed voxel_offset) is carried
        # as ``translation`` by the metadata below.
        if any(int(o) != 0 for o in store.domain.inclusive_min):
            store = store[ts.d[:].translate_to[0]]
        shape = tuple(int(s) for s in store.domain.exclusive_max)
        chunk_shape = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)

        parent, name = _split_parent(path)
        group_attrs: dict = {}
        if parent is not None and driver != "neuroglancer_precomputed":
            try:
                group_attrs = _node_attrs(_open_kvstore(parent))
            except Exception:
                group_attrs = {}
        if driver == "n5":
            attrs = _read_json(kv, "attributes.json") or {}
            meta = _n5_scale_metadata(attrs, ndim, group_attrs, name)
        elif driver == "neuroglancer_precomputed":
            meta = _precomputed_scale_metadata(_read_json(kv, "info") or {}, scale_index, ndim)
        decode = None
        if driver in ("zarr", "zarr3"):
            # OME-NGFF on the parent group, else xarray's dimension names and coordinate
            # arrays (geo and climate data), else legacy per-array attributes
            attrs = _node_attrs(kv)
            decode = _cf_decoding(attrs, store)
            dims = attrs.get("_ARRAY_DIMENSIONS") or (_read_json(kv, "zarr.json") or {}).get(
                "dimension_names"
            )
            meta = _ome_scale_metadata(group_attrs, name, ndim)
            if meta is None and isinstance(dims, list) and len(dims) == ndim and all(dims):
                meta = {"axes": tuple(str(d) for d in dims)}
                if parent is not None:
                    meta |= _cf_coordinates(parent, [str(d) for d in dims], shape)
            if meta is None:
                meta = _zarr_array_metadata(attrs, ndim)

        def pick(override, key, default):
            return tuple(override) if override is not None else meta.get(key, default)

        info = ArrayInfo(
            shape=shape,
            dtype=np.dtype(np.float32) if decode else store.dtype.numpy_dtype,
            chunk_shape=chunk_shape,
            voxel_size=pick(voxel_size, "voxel_size", (1.0,) * ndim),
            units=pick(units, "units", ("",) * ndim),
            axes=pick(axes, "axes", ArrayInfo.default_axes(ndim)),
            translation=pick(translation, "translation", (0.0,) * ndim),
        )
        return cls(store, info, key=f"ts:{path}", decode=decode)


def open_multiscale_tensorstore(path: str, *, cache_bytes: int = 0, **kw) -> MultiscaleSource:
    """Open ``path`` as a multiscale group, or a single array.

    Group levels come from ``multiscales[0].datasets[].path`` (OME-NGFF, COSEM N5) when
    present, else ``s0``, ``s1``, ... are probed.
    """
    kv = _open_kvstore(path)
    driver = _detect_driver(kv)
    if driver == "neuroglancer_precomputed":
        info = _read_json(kv, "info") or {}
        n = len(info.get("scales", [])) or 1
        levels = [
            TensorStoreSource.from_path(path, cache_bytes=cache_bytes, scale_index=i, **kw)
            for i in range(n)
        ]
        return MultiscaleSource(levels, name=path)
    if driver is not None and not driver.endswith("-group"):
        return MultiscaleSource(
            [TensorStoreSource.from_path(path, cache_bytes=cache_bytes, **kw)], name=path
        )
    root = path.rstrip("/")
    names = [str(d["path"]).strip("/") for d in _multiscale_datasets(_node_attrs(kv))]
    if not names:
        i = 0
        while True:
            subdriver = _detect_driver(_open_kvstore(f"{root}/s{i}"))
            if subdriver is None or subdriver.endswith("-group"):
                break
            names.append(f"s{i}")
            i += 1
    if not names:
        raise ValueError(f"{path}: no multiscales datasets or s0..sN levels (driver={driver})")
    levels = [
        TensorStoreSource.from_path(f"{root}/{n}", cache_bytes=cache_bytes, **kw) for n in names
    ]
    return MultiscaleSource(levels, name=path)
