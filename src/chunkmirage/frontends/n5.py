"""N5 frontend. N5 lists axes x-fastest-first, so every list is reversed vs. C order."""

from __future__ import annotations

import gzip
import re
import struct

import numpy as np

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends.base import ChunkRequest, Frontend, Metadata, register_frontend
from chunkmirage.frontends.codecs import Compressor
from chunkmirage.pipeline import Pipeline

_CHUNK_RE = re.compile(r"^s(\d+)/(\d+(?:/\d+)*)$")
_LEVEL_ATTR_RE = re.compile(r"^s(\d+)/attributes\.json$")


@register_frontend
class N5Frontend(Frontend):
    name = "n5"
    neuroglancer_scheme = "n5"

    def __init__(self, compressor: str = "gzip", level: int | None = None):
        if compressor not in ("gzip", "raw", "none"):
            raise ValueError("N5 frontend supports compressor 'gzip' or 'raw'")
        self.compressor = Compressor("none" if compressor in ("raw", "none") else "gzip", level)

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path in ("attributes.json", ""):
            return Metadata.json(self._root_attrs(pipeline))
        if m := _LEVEL_ATTR_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            return Metadata.json(self._level_attrs(pipeline, lvl))
        if m := _CHUNK_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            idx_f = [int(i) for i in m.group(2).split("/")]
            info = pipeline.info(lvl)
            if len(idx_f) != info.ndim:
                return None
            idx = tuple(idx_f[::-1])
            if any(i >= g for i, g in zip(idx, info.chunk_grid)):
                return None
            return ChunkRequest(lvl, idx)
        return None

    def _root_attrs(self, pipeline: Pipeline) -> dict:
        info = pipeline.info(0)
        return {
            "n5": "2.5.0",
            "multiScale": True,
            "axes": list(info.axes[::-1]),
            "units": list(info.units[::-1]),
            "pixelResolution": {
                "dimensions": list(info.voxel_size[::-1]),
                "unit": info.units[-1] or "nm",
            },
            "scales": [[int(round(s)) for s in sc[::-1]] for sc in self.relative_scales(pipeline)],
            "translate": list(info.translation[::-1]),
        }

    def _level_attrs(self, pipeline: Pipeline, lvl: int) -> dict:
        info = pipeline.info(lvl)
        return {
            "dataType": str(info.dtype.name),
            "dimensions": list(info.shape[::-1]),
            "blockSize": list(info.chunk_shape[::-1]),
            "compression": self.compressor.n5_meta(),
            "pixelResolution": {
                "dimensions": list(info.voxel_size[::-1]),
                "unit": info.units[-1] or "nm",
            },
            "transform": {
                "ordering": "C",
                "axes": list(info.axes[::-1]),
                "scale": list(info.voxel_size[::-1]),
                "units": list(info.units[::-1]),
                "translate": list(info.translation[::-1]),
            },
        }

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        # mode 0 = default; dims are F-order (x first) = reversed C shape; data big-endian.
        header = struct.pack(">HH", 0, block.ndim) + struct.pack(
            f">{block.ndim}I", *block.shape[::-1]
        )
        payload = (
            np.ascontiguousarray(block).astype(block.dtype.newbyteorder(">"), copy=False).tobytes()
        )
        if self.compressor.kind == "gzip":
            payload = gzip.compress(payload, compresslevel=self.compressor.level)
        return header + payload
