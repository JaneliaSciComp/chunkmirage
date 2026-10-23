"""tensorstore-backed sources (zarr v2/v3, n5, neuroglancer precomputed; file/s3/gcs/http),
read through worldarray: it detects the format, opens the array in C order with its origin at
0, reads remote stores the way a public dataset most likely lets in, and says where the array
sits (voxel size, the centre of voxel 0, units, axes) by the conventions in
``docs/concepts/formats.md``. What chunkmirage adds on top: xarray coordinate arrays and CF
packing (geo and climate zarr), the overrides ``open_source`` takes, and the ``kind``.
"""

from __future__ import annotations

import re

import numpy as np
import tensorstore as ts
import worldarray as wa

from chunkmirage.core import ArrayInfo, Box, corner_to_centre, kind_for_dtype
from chunkmirage.sources.base import MultiscaleSource, Source

#: One tensorstore context per cache budget, shared by every source this process opens: the
#: levels of an image, two opens of it and several images all draw on one cache pool.
shared_context = wa.shared_context


def detect(url: str) -> wa.Node | None:
    """What is at ``url`` (its format, array or group, the documents found), None if nothing."""
    return wa.detect(url, missing_ok=True)


def node_attrs(node: wa.Node | None) -> dict:
    """User attributes of a zarr v2/v3 or N5 node (array or group); OME 0.5 is unwrapped."""
    return {} if node is None or node.format == "precomputed" else node.ome_attrs


def open_tensorstore(
    path: str, *, cache_bytes: int = 0, context: ts.Context | None = None, scale_index: int | None = None
) -> ts.TensorStore:
    """One array, C order, origin 0, its metadata not interpreted (a field's, a coordinate's)."""
    ctx = context if context is not None else shared_context(cache_bytes)
    return wa.open_tensorstore(path, level=scale_index, context=ctx)


def _cf_unit(units: str) -> tuple[str, float]:
    """A CF coordinate's ``units`` as (unit, factor to it): ``days since ...`` and the like
    become seconds, degrees of latitude or longitude unitless (no viewer has degrees)."""
    u = units.strip().lower()
    if m := re.match(r"(day|hour|minute|second)s?\s+since\b", u):
        return "s", wa.units.factor(m.group(1), "second")
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
    vs, un = [1.0] * len(dims), [""] * len(dims)
    tr = list(corner_to_centre((0.0,) * len(dims), vs, dims))  # unplaced: a corner at 0
    for a, (dim, n) in enumerate(zip(dims, shape)):
        try:
            store = wa.open_tensorstore(f"{parent}/{dim}")
        except Exception:
            continue
        if store.rank != 1 or store.shape[0] != n or n < 2:
            continue
        first = np.asarray(store[0:2].read().result(), dtype=np.float64)
        last = float(np.asarray(store[n - 1].read().result()))
        step = (last - first[0]) / (n - 1)  # the whole span: float32 coordinates are coarse
        if step <= 0 or abs(first[1] - first[0] - step) > 0.01 * step:
            continue
        unit, f = _cf_unit(str(node_attrs(detect(f"{parent}/{dim}")).get("units", "")))
        # to the coordinates' own precision: float32 degrees 0.01 apart are not 0.0099999998
        digits = 6 if store.dtype.numpy_dtype.itemsize <= 4 else 12
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


def _placed(meta: wa.ArrayMeta) -> dict:
    """Where ``meta``'s array sits; for an xarray array (named dimensions and no placement of
    its own), from the coordinate arrays beside it."""
    found = {
        "voxel_size": meta.voxel_size,
        "translation": meta.translation,
        "units": meta.units,
        "axes": meta.axes,
    }
    if meta.placed_by is None and meta.format in ("zarr2", "zarr3") and meta.axes_named and meta.parent:
        found |= _cf_coordinates(meta.parent, list(meta.axes), meta.shape)
    return found


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
    def from_array(
        cls, vol: wa.WorldArray, key: str, *, voxel_size=None, units=None, translation=None, axes=None
    ) -> TensorStoreSource:
        """``vol`` (opened in its own units) as a source; the overrides win over its metadata."""
        meta, store = vol.meta, vol.ts
        decode = _cf_decoding(meta.attrs, store) if meta.format in ("zarr2", "zarr3") else None
        found = _placed(meta)

        def pick(override, name):
            return tuple(override) if override is not None else found[name]

        dtype = np.dtype(np.float32) if decode else meta.dtype
        info = ArrayInfo(
            shape=meta.shape,
            dtype=dtype,
            chunk_shape=meta.chunk_shape,
            voxel_size=pick(voxel_size, "voxel_size"),
            units=pick(units, "units"),
            axes=pick(axes, "axes"),
            translation=pick(translation, "translation"),
            kind=None if decode else kind_for_dtype(dtype),  # override: open_source(kind=)
        )
        return cls(store, info, key=key, decode=decode)

    @classmethod
    def from_path(cls, path: str, *, cache_bytes: int = 0, **overrides) -> TensorStoreSource:
        vol = wa.open(path, units=None, context=shared_context(cache_bytes), metadata_conflicts="warn")
        return cls.from_array(vol, f"ts:{path}", **overrides)


def open_multiscale_tensorstore(path: str, *, cache_bytes: int = 0, **kw) -> MultiscaleSource:
    """Open ``path`` as a multiscale group (its ``multiscales[0].datasets``, else ``s0``,
    ``s1``, ...; a precomputed volume's scales), or a single array."""
    node = detect(path)
    if node is None:
        raise ValueError(f"{path}: no zarr, N5 or precomputed array or group there")
    if node.kind == "array" or node.scale_key is not None:
        return MultiscaleSource([TensorStoreSource.from_path(path, cache_bytes=cache_bytes, **kw)], name=path)
    context = shared_context(cache_bytes)
    pyramid = wa.open_pyramid(path, units=None, context=context, metadata_conflicts="warn")
    root = path.rstrip("/")
    levels = [
        TensorStoreSource.from_array(vol, f"ts:{root}/{name}", **kw)
        for name, vol in zip(pyramid.names, pyramid.levels)
    ]
    return MultiscaleSource(levels, name=path)
