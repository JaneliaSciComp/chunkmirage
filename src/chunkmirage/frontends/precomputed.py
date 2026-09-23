"""Neuroglancer precomputed frontend ("raw" encoding: plain little-endian [c,z,y,x] bytes).

Only 3-D volumes (z,y,x) or 4-D (c,z,y,x) are representable. ``raw`` is uncompressed, so
the server applies HTTP ``Content-Encoding: gzip`` when the client accepts it.
"""

from __future__ import annotations

import re

import numpy as np

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends.base import ChunkRequest, Frontend, Metadata, register_frontend
from chunkmirage.pipeline import Pipeline

_CHUNK_RE = re.compile(r"^s(\d+)/(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$")
_UNIT_TO_NM = {"nm": 1.0, "um": 1e3, "µm": 1e3, "mm": 1e6, "m": 1e9, "": 1.0}
_SUPPORTED = {"uint8", "uint16", "uint32", "uint64", "float32"}


@register_frontend
class PrecomputedFrontend(Frontend):
    name = "precomputed"
    neuroglancer_scheme = "precomputed"

    def __init__(self, volume_type: str | None = None):
        self.volume_type = volume_type  # "image" | "segmentation" | None (auto)

    @staticmethod
    def _spatial(info: ArrayInfo) -> tuple[int, ...]:
        if info.ndim == 3:
            return (0, 1, 2)
        if info.ndim == 4:
            return (1, 2, 3)
        raise ValueError("precomputed needs 3-D (z,y,x) or 4-D (c,z,y,x) data")

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path in ("info", ""):
            return Metadata.json(self._info(pipeline))
        if m := _CHUNK_RE.match(path):
            lvl = int(m.group(1))
            if lvl >= pipeline.num_levels:
                return None
            info = pipeline.info(lvl)
            sp = self._spatial(info)
            x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups()[1:])
            starts = (z0, y0, x0)
            stops = (z1, y1, x1)
            cs = [info.chunk_shape[a] for a in sp]
            if any(s % c for s, c in zip(starts, cs)):
                return None
            idx_sp = tuple(s // c for s, c in zip(starts, cs))
            idx = idx_sp if info.ndim == 3 else (0, *idx_sp)
            if any(i >= g for i, g in zip(idx, info.chunk_grid)):
                return None
            box = info.chunk_box(idx)
            if tuple(box.stop[a] for a in sp) != stops:
                return None
            return ChunkRequest(lvl, idx)
        return None

    def _info(self, pipeline: Pipeline) -> dict:
        info0 = pipeline.info(0)
        sp = self._spatial(info0)
        if info0.dtype.name not in _SUPPORTED:
            raise ValueError(
                f"precomputed supports {sorted(_SUPPORTED)}, got {info0.dtype.name}; add a 'cast' op"
            )
        vt = self.volume_type or (
            "segmentation" if info0.dtype.kind == "u" and info0.dtype.itemsize >= 4 else "image"
        )
        scales = []
        for lvl in range(pipeline.num_levels):
            info = pipeline.info(lvl)
            res = [info.voxel_size[a] * _UNIT_TO_NM.get(info.units[a], 1.0) for a in sp][::-1]
            off = [
                int(round(info.translation[a] / info.voxel_size[a])) if info.voxel_size[a] else 0
                for a in sp
            ][::-1]
            scales.append(
                {
                    "key": f"s{lvl}",
                    "size": [info.shape[a] for a in sp][::-1],
                    "resolution": res,
                    "voxel_offset": off,
                    "chunk_sizes": [[info.chunk_shape[a] for a in sp][::-1]],
                    "encoding": "raw",
                }
            )
        return {
            "@type": "neuroglancer_multiscale_volume",
            "type": vt,
            "data_type": info0.dtype.name,
            "num_channels": 1 if info0.ndim == 3 else int(info0.shape[0]),
            "scales": scales,
        }

    def encode(self, info: ArrayInfo, index, block: np.ndarray) -> bytes:
        # raw: C-order [c, z, y, x]; our block is already (z,y,x) or (c,z,y,x).
        return (
            np.ascontiguousarray(block).astype(block.dtype.newbyteorder("<"), copy=False).tobytes()
        )
