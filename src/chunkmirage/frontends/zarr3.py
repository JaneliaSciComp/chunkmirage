from __future__ import annotations

import re

import numpy as np
import worldarray as wa

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends._ome import group_attrs
from chunkmirage.frontends.base import Frontend, Metadata, register_frontend
from chunkmirage.frontends.codecs import Compressor
from chunkmirage.pipeline import Pipeline

_LEVEL_RE = re.compile(r"^s(\d+)/zarr\.json$")

_DTYPES = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


@register_frontend
class Zarr3Frontend(Frontend):
    name = "zarr3"
    neuroglancer_scheme = "zarr3"
    root_keys = ("zarr.json",)
    level_keys = ("zarr.json",)

    def __init__(self, compressor: str = "blosc", level: int | None = None):
        self.compressor = Compressor(compressor, level)

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path in ("zarr.json", ""):
            return Metadata.json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": group_attrs(pipeline, "0.5"),
                }
            )
        if m := _LEVEL_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            return Metadata.json(self._array_meta(pipeline.info(lvl)))
        return self.chunk_request(pipeline, path, lambda key, info: wa.parse_chunk_key(key, "zarr3"))

    def _array_meta(self, info: ArrayInfo) -> dict:
        return {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(info.shape),
            "data_type": _DTYPES[info.dtype.name],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(info.chunk_shape)},
            },
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0,
            "codecs": self.compressor.zarr3_codecs(info.dtype),
            "attributes": {},
            "dimension_names": list(info.axes),
        }

    def chunk_key(self, level: int, index) -> str:
        return f"s{level}/" + wa.chunk_key(index, "zarr3")

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        full = self.pad_to_full(info, index, block)
        raw = np.ascontiguousarray(full).astype(full.dtype.newbyteorder("<"), copy=False).tobytes()
        return self.compressor.encode(raw)
