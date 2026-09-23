"""tensorstore-backed sources (zarr v2/v3, n5, neuroglancer precomputed; file/s3/gcs/http).

tensorstore does all I/O in C++ threads (GIL released), supports async reads, and has its
own byte-bounded ``cache_pool`` so the *raw* chunks of a remote source are cached without
any extra work on our side.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import tensorstore as ts

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

_SCALE_RE = re.compile(r"^s(\d+)$")


def _split_container(path: str) -> tuple[str, str]:
    """Split ``.../foo.zarr/a/b`` into (``.../foo.zarr``, ``a/b``); if no marker, (path, '')."""
    for marker in (".zarr", ".n5", ".precomputed"):
        idx = path.rfind(marker + "/")
        if idx >= 0:
            cut = idx + len(marker)
            return path[:cut], path[cut + 1 :].strip("/")
        if path.endswith(marker):
            return path, ""
    return path.rstrip("/"), ""


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
    if _read_json(kv, "zarr.json") is not None:
        return "zarr3"
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


def _n5_scale_metadata(
    attrs: dict, ndim: int
) -> tuple[tuple[float, ...], tuple[str, ...], tuple[float, ...]] | None:
    """Best-effort voxel size / units / translation from N5 attributes (C order)."""
    if "transform" in attrs and isinstance(attrs["transform"], dict):
        t = attrs["transform"]
        scale = list(t.get("scale", [1.0] * ndim))
        units = list(t.get("units", ["nm"] * ndim))
        trans = list(t.get("translate", [0.0] * ndim))
        # N5 lists axes in F order (x fastest first) -> reverse to C order.
        return tuple(scale[::-1]), tuple(units[::-1]), tuple(trans[::-1])
    if "pixelResolution" in attrs:
        pr = attrs["pixelResolution"]
        dims = list(pr.get("dimensions", [1.0] * ndim))
        unit = pr.get("unit", "nm")
        return tuple(dims[::-1]), (unit,) * ndim, (0.0,) * ndim
    if "resolution" in attrs:
        res = list(attrs["resolution"])
        return tuple(res[::-1]), ("nm",) * ndim, (0.0,) * ndim
    return None


def _ome_scale_metadata(group_attrs: dict, level: int, ndim: int):
    """Voxel size / units / translation from OME-NGFF multiscales metadata (C order already)."""
    ms = (group_attrs.get("multiscales") or [None])[0]
    if not ms:
        return None
    axes = ms.get("axes", [])
    units = tuple(a.get("unit", "") if isinstance(a, dict) else "" for a in axes) or ("",) * ndim
    names = tuple(a.get("name") if isinstance(a, dict) else str(a) for a in axes)
    try:
        ds = ms["datasets"][level]
    except (KeyError, IndexError):
        return None
    scale = [1.0] * ndim
    trans = [0.0] * ndim
    for ct in ds.get("coordinateTransformations", []):
        if ct.get("type") == "scale":
            scale = list(ct["scale"])
        elif ct.get("type") == "translation":
            trans = list(ct["translation"])
    if len(units) != ndim:
        units = ("",) * ndim
    return tuple(scale), units, tuple(trans), (names if len(names) == ndim else None)


class TensorStoreSource(Source):
    def __init__(self, store: ts.TensorStore, info: ArrayInfo, key: str):
        self.store = store
        self._info = info
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        return np.asarray(self.store[box.slices()].read().result(), dtype=self._info.dtype)

    async def read_async(self, box: Box) -> np.ndarray:
        return np.asarray(await self.store[box.slices()].read(), dtype=self._info.dtype)

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
        store = open_tensorstore(path, cache_bytes=cache_bytes, scale_index=scale_index)
        domain = store.domain
        ndim = store.rank
        shape = tuple(int(s) for s in domain.exclusive_max)  # assume origin 0 for now
        if any(int(o) != 0 for o in domain.inclusive_min):
            store = store[domain.translate_to[[0] * ndim]]
        chunk_shape = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)
        kv = _open_kvstore(path)
        driver = _detect_driver(kv)
        meta = None
        if driver == "n5":
            attrs = _read_json(kv, "attributes.json") or {}
            meta = _n5_scale_metadata(attrs, ndim)
        elif driver == "neuroglancer_precomputed":
            sp = store.spec().to_json()
            res = None
            try:
                info = _read_json(kv, "info") or {}
                scales = info.get("scales", [])
                res = scales[sp.get("scale_index", 0)]["resolution"]
            except Exception:
                pass
            if res:
                # precomputed arrays are (x, y, z[, c]) in tensorstore; we transpose to C order.
                store = store[ts.d[:].transpose[::-1]]
                shape = tuple(int(s) for s in store.domain.exclusive_max)
                chunk_shape = tuple(int(c) for c in store.chunk_layout.read_chunk.shape)
                r = [float(v) for v in res][::-1]
                if ndim == 4:
                    meta = ((1.0, *r), ("", "nm", "nm", "nm"), (0.0,) * 4)
                else:
                    meta = (tuple(r), ("nm",) * ndim, (0.0,) * ndim)
        found_axes = None
        if meta is None or driver in ("zarr", "zarr3"):
            # Look one level up for OME-NGFF multiscales and figure out our level index.
            container, inner = _split_container(path)
            parts = inner.split("/") if inner else []
            m = _SCALE_RE.match(parts[-1]) if parts else None
            level = int(m.group(1)) if m else 0
            parent = "/".join(parts[:-1])
            try:
                pkv = _open_kvstore(container if not parent else f"{container}/{parent}")
                gattrs = _read_json(pkv, ".zattrs")
                if gattrs is None:
                    zj = _read_json(pkv, "zarr.json") or {}
                    gattrs = zj.get("attributes", {}).get("ome", zj.get("attributes", {}))
                if gattrs:
                    ome = _ome_scale_metadata(gattrs, level, ndim)
                    if ome:
                        meta = ome[:3]
                        found_axes = ome[3]
            except Exception:
                pass
        if meta is None:
            meta = ((1.0,) * ndim, ("",) * ndim, (0.0,) * ndim)
        vs, un, tr = meta
        info = ArrayInfo(
            shape=shape,
            dtype=store.dtype.numpy_dtype,
            chunk_shape=chunk_shape,
            voxel_size=tuple(voxel_size) if voxel_size is not None else vs,
            units=tuple(units) if units is not None else un,
            axes=tuple(axes) if axes is not None else (found_axes or ArrayInfo.default_axes(ndim)),
            translation=tuple(translation) if translation is not None else tr,
        )
        return cls(store, info, key=f"ts:{path}")


def open_multiscale_tensorstore(path: str, *, cache_bytes: int = 0, **kw) -> MultiscaleSource:
    """Open ``path`` as a multiscale group (``s0``, ``s1``, ...), or a single array."""
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
    levels = []
    i = 0
    while True:
        sub = f"{path.rstrip('/')}/s{i}"
        subdriver = _detect_driver(_open_kvstore(sub))
        if subdriver is None or subdriver.endswith("-group"):
            break
        levels.append(TensorStoreSource.from_path(sub, cache_bytes=cache_bytes, **kw))
        i += 1
    if not levels:
        raise ValueError(f"{path}: no array or s0..sN levels found (driver={driver})")
    return MultiscaleSource(levels, name=path)
