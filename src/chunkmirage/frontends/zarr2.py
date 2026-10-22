from __future__ import annotations

import re

import numpy as np
import worldarray as wa

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends._ome import group_attrs
from chunkmirage.frontends.base import Frontend, Metadata, register_frontend
from chunkmirage.frontends.codecs import Compressor
from chunkmirage.pipeline import Pipeline

_LEVEL_RE = re.compile(r"^s(\d+)/(\.zarray|\.zattrs|\.zgroup)$")


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
            return Metadata.json(group_attrs(pipeline, "0.4"))
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
        # either separator: a client that ignores dimension_separator still finds the chunk
        return self.chunk_request(
            pipeline, path, lambda key, info: wa.parse_chunk_key(key.replace("/", "."), "zarr2")
        )

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
            ".zattrs": group_attrs(pipeline, "0.4"),
        }
        for lvl in range(pipeline.num_levels):
            meta[f"s{lvl}/.zarray"] = self._zarray(pipeline.info(lvl))
            meta[f"s{lvl}/.zattrs"] = {}
        return {"zarr_consolidated_format": 1, "metadata": meta}

    def chunk_key(self, level: int, index) -> str:
        return f"s{level}/" + wa.chunk_key(index, "zarr2", self.separator)

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        full = self.pad_to_full(info, index, block)
        raw = np.ascontiguousarray(full).astype(full.dtype.newbyteorder("<"), copy=False).tobytes()
        return self.compressor.encode(raw)
