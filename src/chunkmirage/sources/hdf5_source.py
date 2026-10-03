"""HDF5 source via h5py (optional dependency). Path syntax: ``/path/file.h5::/group/dataset``."""

from __future__ import annotations

import threading

import numpy as np

from chunkmirage.core import ArrayInfo, Box, kind_for_dtype
from chunkmirage.sources.base import MultiscaleSource, Source


class HDF5Source(Source):
    def __init__(
        self,
        filename: str,
        dataset: str,
        chunk_shape=None,
        voxel_size=None,
        units=None,
        axes=None,
        translation=None,
    ):
        import h5py

        self._f = h5py.File(filename, "r")
        self._d = self._f[dataset]
        self._lock = threading.Lock()  # h5py is not thread-safe
        ndim = self._d.ndim
        cs = tuple(chunk_shape or self._d.chunks or tuple(min(64, s) for s in self._d.shape))
        vs = voxel_size or self._d.attrs.get(
            "resolution", self._d.attrs.get("voxel_size", [1.0] * ndim)
        )
        offset = translation  # voxel 0's centre, as everywhere in chunkmirage
        given = self._d.attrs.get("offset")
        if offset is None and (given is not None or "resolution" in self._d.attrs or "voxel_size" in self._d.attrs):
            # funlib's offset is voxel 0's corner (0 if absent while a resolution is given)
            from chunkmirage.sources.tensorstore_source import corner_to_centre

            corner = list(given) if given is not None else [0.0] * ndim
            offset = corner_to_centre(corner, vs, tuple(axes) if axes else None)
        self._info = ArrayInfo(
            shape=self._d.shape,
            dtype=self._d.dtype,
            chunk_shape=cs,
            voxel_size=tuple(float(v) for v in vs),
            units=tuple(units) if units else ("nm",) * ndim,
            axes=tuple(axes) if axes else ArrayInfo.default_axes(ndim),
            translation=tuple(float(v) for v in offset) if offset is not None else None,
            kind=kind_for_dtype(self._d.dtype),
        )
        self._key = f"h5:{filename}::{dataset}"

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        with self._lock:
            return np.asarray(self._d[box.slices()])


def open_multiscale_hdf5(path: str, cache_bytes: int = 0, **kw) -> MultiscaleSource:
    filename, _, dataset = path.partition("::")
    if not dataset:
        raise ValueError("HDF5 paths must be 'file.h5::/dataset'")
    return MultiscaleSource([HDF5Source(filename, dataset, **kw)], name=path)
