"""tensorstore-backed sources (zarr v2/v3, n5, neuroglancer precomputed; file/s3/gcs/http).

tensorstore does all I/O in C++ threads (GIL released), supports async reads, and has its
own byte-bounded ``cache_pool`` so the *raw* chunks of a remote source are cached without
any extra work on our side.

Remote stores are read the way a public dataset most likely lets in (``kvstore_specs``):
``s3://`` anonymously first, then with AWS's default credentials; ``gs://`` with Google's
default credentials (anonymous without any), then through the bucket's public https URL.
Which way worked is remembered per bucket. Metadata reads give up after a few seconds.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import tensorstore as ts

from chunkmirage.core import ArrayInfo, Box, kind_for_dtype
from chunkmirage.sources.base import MultiscaleSource, Source

log = logging.getLogger("chunkmirage")
T = TypeVar("T")

#: How a metadata read retries a failing request: about 5 s in all. tensorstore's default,
#: 32 retries of up to 32 s each, keeps a mistyped host's dataset opening for many minutes.
METADATA_RETRIES = {"max_retries": 5, "initial_delay": "0.2s", "max_delay": "2s"}
GCS_PUBLIC_URL = "https://storage.googleapis.com"


def kvstore_specs(url_or_path: str) -> list[dict]:
    """The tensorstore kvstore specs that read the directory ``url_or_path``, in the order to
    try them. ``s3://``: anonymously, then with AWS's default credentials (S3 refuses even a
    public read signed with a stale or foreign key). ``gs://``: Google's default credentials
    (anonymous without any, tensorstore's rule), then the bucket's public https URL, for
    credentials that are there but broken (``TENSORSTORE_GCS_HTTP_URL`` moves it). Anything
    else: one way. Each path ends in '/': tensorstore concatenates keys literally."""
    root = url_or_path if url_or_path.endswith("/") else url_or_path + "/"
    if not re.match(r"^(s3|gs|http|https|file|memory)://", root):
        return [{"driver": "file", "path": root}]
    spec = ts.KvStore.Spec(root).to_json()
    if spec["driver"] == "s3":
        return [{**spec, "aws_credentials": {"type": t}} for t in ("anonymous", "default")]
    if spec["driver"] == "gcs":
        public = (os.environ.get("TENSORSTORE_GCS_HTTP_URL") or GCS_PUBLIC_URL).rstrip("/")
        path = f"/{spec['bucket']}/{spec.get('path', '')}"
        return [spec, {"driver": "http", "base_url": public, "path": path}]
    return [spec]


_admitted: dict[tuple[str, str], int] = {}  # (driver, bucket) -> the spec that was let in


def _refused(error: Exception) -> bool:
    return str(error).startswith(("PERMISSION_DENIED", "UNAUTHENTICATED"))


def with_access(specs: list[dict], attempt: Callable[[dict], T]) -> T:
    """``attempt(spec)`` with each of ``specs`` in turn, moving on only while access is
    refused; the one let in is tried first next time, for its bucket. Any other error, and
    the last refusal, is raised."""
    if len(specs) == 1:
        return attempt(specs[0])
    key = (specs[0]["driver"], specs[0].get("bucket", ""))
    first = _admitted.get(key, 0)
    order = [first] + [i for i in range(len(specs)) if i != first]
    for i in order[:-1]:
        try:
            result = attempt(specs[i])
        except Exception as e:
            if not _refused(e):
                raise
            log.info("%s refused a read (%s); trying the next way in", key, str(e).splitlines()[0][:200])
            continue
        _admitted[key] = i
        return result
    result = attempt(specs[order[-1]])
    _admitted[key] = order[-1]
    return result


def metadata_spec(spec: dict) -> dict:
    """``spec`` with its requests' retries capped (``METADATA_RETRIES``), for metadata reads."""
    if spec["driver"] not in ("s3", "gcs", "http"):
        return spec
    return {**spec, "context": {f"{spec['driver']}_request_retries": METADATA_RETRIES}}


@functools.lru_cache(maxsize=512)
def _metadata_kvstore(spec_json: str) -> ts.KvStore:
    return ts.KvStore.open(metadata_spec(json.loads(spec_json))).result()


class Location:
    """A directory (local, or an ``s3://``, ``gs://`` or ``http(s)://`` URL) and the ways to
    read it (``kvstore_specs``): what ``_open_kvstore`` returns."""

    def __init__(self, url_or_path: str):
        self.specs = kvstore_specs(url_or_path)

    def read(self, key: str) -> bytes | None:
        """The file ``key`` in the directory, None if it is not there."""

        def attempt(spec):
            r = _metadata_kvstore(json.dumps(spec, sort_keys=True)).read(key).result()
            return bytes(r.value) if r.state == "value" else None

        return with_access(self.specs, attempt)


def _open_kvstore(url_or_path: str) -> Location:
    """The directory ``url_or_path``, to read metadata from."""
    return Location(url_or_path)


def _read_json(kv: Location, key: str) -> dict | None:
    data = kv.read(key)
    return None if data is None else json.loads(data.decode())


# Compressor members tensorstore knows; it rejects any other (numcodecs >= 0.13 writes zstd's
# "checksum", for one)
_ZARR2_COMPRESSOR_FIELDS = {
    "zstd": {"id", "level"},
    "zlib": {"id", "level"},
    "gzip": {"id", "level"},
    "bz2": {"id", "level"},
    "blosc": {"id", "cname", "clevel", "shuffle", "blocksize"},
}


def cleaned_zarray(meta: dict | None) -> dict | None:
    """A zarr v2 ``.zarray`` without the compressor members tensorstore rejects, or None if
    it has none of them."""
    compressor = (meta or {}).get("compressor")
    if not isinstance(compressor, dict):
        return None
    allowed = _ZARR2_COMPRESSOR_FIELDS.get(compressor.get("id", ""))
    if allowed is None or set(compressor) <= allowed:
        return None
    log.info("ignoring compressor members %s tensorstore does not know", sorted(set(compressor) - allowed))
    return {**meta, "compressor": {k: v for k, v in compressor.items() if k in allowed}}


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
    extra: dict[str, Any] = {}
    options: dict[str, Any] = {}
    if driver == "neuroglancer_precomputed" and scale_index is not None:
        extra["scale_index"] = scale_index
    if driver == "zarr" and (cleaned := cleaned_zarray(_read_json(kv, ".zarray"))) is not None:
        extra["metadata"] = cleaned
        options = {"open": True, "assume_metadata": True}
    if context is None:
        context = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_bytes)}})

    def attempt(kvstore: dict) -> ts.TensorStore:
        spec = {"driver": driver, "kvstore": kvstore, **extra}
        return ts.open(spec, read=True, write=False, context=context, **options).result()

    return with_access(kv.specs, attempt)


# Per-level metadata is a dict with any of ``voxel_size``, ``units``, ``translation``, ``axes``
# (C order, length ndim); missing keys fall back to defaults in ``from_path``.
#
# ``translation`` is where voxel 0's centre is, as in OME-Zarr (and as Neuroglancer reads
# OME-Zarr: it moves each voxel back by half to its corner). Conventions that give voxel 0's
# corner are moved by half a voxel on reading: precomputed's ``voxel_offset`` and funlib's
# ``offset`` (a region's start), as Neuroglancer and funlib place them.


def corner_to_centre(corner, voxel_size, axes=None) -> tuple[float, ...]:
    """Voxel 0's centre from its corner, both C order and one entry per axis: half a voxel on
    along the spatial axes (those named z, y or x, else the last three), not along channels
    or time."""
    n = len(corner)
    spatial = {a for a in range(n) if (axes[a] in ("z", "y", "x") if axes else a >= n - 3)}
    return tuple(
        float(c) + (float(v) / 2 if a in spatial else 0.0)
        for a, (c, v) in enumerate(zip(corner, voxel_size))
    )


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
    level's ``downsamplingFactors``, plus ``offset``. Plain N5 lists are x-first.

    funlib's ``resolution``/``offset`` give voxel 0's corner (``offset``, 0 if absent).
    ``pixelResolution`` without an offset is BigDataViewer's convention, voxel centres on
    whole multiples of the full-resolution size: a level downsampled by ``f`` has its
    first centre ``(f - 1) / 2`` full-resolution voxels on."""
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
    offset = attrs.get("offset")
    offset = offset if isinstance(offset, list) else None
    factors = attrs.get("downsamplingFactors")
    level_res = None
    if isinstance(res, list):
        if isinstance(factors, list) and len(factors) == len(res):
            level_res = [float(r) * float(f) for r, f in zip(res, factors)]
        else:
            factors, level_res = None, [float(r) for r in res]
        meta["voxel_size"] = _floats(_per_axis(level_res, ndim, 1.0, reverse=True))
    meta["units"] = _per_axis(units, ndim, "", reverse=True)
    meta["axes"] = _per_axis(attrs.get("axes") or group_attrs.get("axes"), ndim, None, True)
    if offset is not None or "resolution" in base:  # funlib: the corner of voxel 0
        corner = _floats(_per_axis(offset or [], ndim, 0.0, reverse=True))
        if corner is not None:
            sizes = meta.get("voxel_size") or (1.0,) * ndim
            meta["translation"] = corner_to_centre(corner, sizes, meta["axes"])
    elif factors is not None:  # BigDataViewer's levels: centred on their blocks
        centre = [(float(f) - 1) / 2 * float(r) for r, f in zip(res, factors)]
        meta["translation"] = _floats(_per_axis(centre, ndim, 0.0, reverse=True))
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
    ``offset``, ``units``, ``axis_names``, or a COSEM-style ``transform``. funlib's
    ``offset`` is voxel 0's corner (0 if absent while a resolution is given)."""
    if isinstance(attrs.get("transform"), dict):
        return _transform_metadata(attrs["transform"], ndim)
    res = attrs.get("voxel_size", attrs.get("resolution"))
    units = attrs.get("units")
    if isinstance(units, str) and isinstance(res, list):
        units = [units] * len(res)
    voxel_size = _floats(_per_axis(res, ndim, 1.0))
    axes = _per_axis(attrs.get("axis_names"), ndim, None)
    translation = None
    if isinstance(attrs.get("offset"), list) or voxel_size is not None:
        corner = _per_axis(attrs.get("offset"), ndim, 0.0) or (0.0,) * ndim
        translation = corner_to_centre(corner, voxel_size or (1.0,) * ndim, axes)
    meta = {
        "voxel_size": voxel_size,
        "translation": translation,
        "units": _per_axis(units, ndim, ""),
        "axes": axes,
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
    """``resolution`` (nm) and ``voxel_offset`` (voxels) of one scale, x-first -> C order.
    ``voxel_offset`` is the first voxel's corner, in voxels, as Neuroglancer places it."""
    try:
        sc = info["scales"][scale_index or 0]
        res = [float(v) for v in sc["resolution"]]
    except (KeyError, IndexError, TypeError):
        return {}
    off = [(float(o) + 0.5) * r for o, r in zip(sc.get("voxel_offset", [0] * len(res)), res)]
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

        dtype = np.dtype(np.float32) if decode else store.dtype.numpy_dtype
        info = ArrayInfo(
            shape=shape,
            dtype=dtype,
            chunk_shape=chunk_shape,
            voxel_size=pick(voxel_size, "voxel_size", (1.0,) * ndim),
            units=pick(units, "units", ("",) * ndim),
            axes=pick(axes, "axes", ArrayInfo.default_axes(ndim)),
            translation=pick(translation, "translation", (0.0,) * ndim),
            kind=None if decode else kind_for_dtype(dtype),  # override: open_source(kind=)
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
