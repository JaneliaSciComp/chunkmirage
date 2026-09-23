from __future__ import annotations

import re

import numpy as np

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends._ome import multiscales
from chunkmirage.frontends.base import ChunkRequest, Frontend, Metadata, register_frontend
from chunkmirage.frontends.codecs import Compressor
from chunkmirage.pipeline import Pipeline

_LEVEL_RE = re.compile(r"^s(\d+)/zarr\.json$")
_CHUNK_RE = re.compile(r"^s(\d+)/c/(\d+(?:/\d+)*)$")

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

    def __init__(self, compressor: str = "gzip", level: int | None = None):
        self.compressor = Compressor(compressor, level)

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path in ("zarr.json", ""):
            return Metadata.json(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "ome": {"version": "0.5", "multiscales": [multiscales(pipeline, "0.5")]}
                    },
                }
            )
        if m := _LEVEL_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            return Metadata.json(self._array_meta(pipeline.info(lvl)))
        if m := _CHUNK_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            idx = tuple(int(i) for i in m.group(2).split("/"))
            info = pipeline.info(lvl)
            if len(idx) != info.ndim or any(i >= g for i, g in zip(idx, info.chunk_grid)):
                return None
            return ChunkRequest(lvl, idx)
        return None

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

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        full = self.pad_to_full(info, index, block)
        raw = np.ascontiguousarray(full).astype(full.dtype.newbyteorder("<"), copy=False).tobytes()
        return self.compressor.encode(raw)
