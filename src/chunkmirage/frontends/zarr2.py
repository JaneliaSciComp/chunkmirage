from __future__ import annotations

import re

import numpy as np

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends._ome import multiscales
from chunkmirage.frontends.base import ChunkRequest, Frontend, Metadata, register_frontend
from chunkmirage.frontends.codecs import Compressor
from chunkmirage.pipeline import Pipeline

_LEVEL_RE = re.compile(r"^s(\d+)/(\.zarray|\.zattrs|\.zgroup)$")
_CHUNK_RE = re.compile(r"^s(\d+)/(\d+(?:[./]\d+)*)$")


@register_frontend
class Zarr2Frontend(Frontend):
    name = "zarr"
    neuroglancer_scheme = "zarr2"
    root_keys = (".zgroup", ".zattrs", ".zmetadata")
    level_keys = (".zarray", ".zattrs")

    def __init__(self, compressor: str = "blosc", level: int | None = None, separator: str = "/"):
        self.compressor = Compressor(compressor, level)
        self.separator = separator

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path == ".zgroup":
            return Metadata.json({"zarr_format": 2})
        if path == ".zattrs":
            return Metadata.json({"multiscales": [multiscales(pipeline, "0.4")]})
        if path in ("", ".zmetadata"):
            return Metadata.json(self._consolidated(pipeline))
        if m := _LEVEL_RE.match(path):
            lvl, what = int(m.group(1)), m.group(2)
            if lvl >= pipeline.num_levels:
                return None
            if what == ".zarray":
                return Metadata.json(self._zarray(pipeline.info(lvl)))
            if what == ".zattrs":
                return Metadata.json({})
            return None  # levels are arrays, not groups
        if m := _CHUNK_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            idx = tuple(int(i) for i in re.split(r"[./]", m.group(2)))
            info = pipeline.info(lvl)
            if len(idx) != info.ndim or any(i >= g for i, g in zip(idx, info.chunk_grid)):
                return None
            return ChunkRequest(lvl, idx)
        return None

    def _zarray(self, info: ArrayInfo) -> dict:
        return {
            "zarr_format": 2,
            "shape": list(info.shape),
            "chunks": list(info.chunk_shape),
            "dtype": info.dtype.newbyteorder("<").str
            if info.dtype.itemsize > 1
            else info.dtype.str.replace(">", "|").replace("<", "|"),
            "compressor": self.compressor.zarr2_meta(),
            "fill_value": 0,
            "order": "C",
            "filters": None,
            "dimension_separator": self.separator,
        }

    def _consolidated(self, pipeline: Pipeline) -> dict:
        meta = {
            ".zgroup": {"zarr_format": 2},
            ".zattrs": {"multiscales": [multiscales(pipeline, "0.4")]},
        }
        for lvl in range(pipeline.num_levels):
            meta[f"s{lvl}/.zarray"] = self._zarray(pipeline.info(lvl))
            meta[f"s{lvl}/.zattrs"] = {}
        return {"zarr_consolidated_format": 1, "metadata": meta}

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        full = self.pad_to_full(info, index, block)
        raw = np.ascontiguousarray(full).astype(full.dtype.newbyteorder("<"), copy=False).tobytes()
        return self.compressor.encode(raw)
