"""Neuroglancer precomputed frontend ("raw" encoding: plain little-endian [c,z,y,x] bytes).

Only 3-D volumes (z,y,x) or 4-D (c,z,y,x) are representable. ``raw`` is uncompressed, so
the server applies HTTP ``Content-Encoding: gzip`` when the client accepts it.
"""

from __future__ import annotations

import math

import numpy as np
import worldarray as wa

from chunkmirage.core import ArrayInfo
from chunkmirage.frontends.base import Frontend, Metadata, register_frontend
from chunkmirage.neuroglancer import _UNIT_TO_M
from chunkmirage.pipeline import Pipeline

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

    @staticmethod
    def _offset(info: ArrayInfo, sp: tuple[int, ...]) -> list[int]:
        """``voxel_offset``, z, y, x: the first voxel's corner in whole voxels, translation
        (its centre) less half a voxel, to the nearest (an OME-style 0 rounds to 0)."""
        return [
            math.floor(info.translation[a] / info.voxel_size[a] + 1e-6) if info.voxel_size[a] else 0
            for a in sp
        ]

    def _key(self, key: str, info: ArrayInfo) -> tuple[int, ...] | None:
        """A chunk's index from its key, ``x0-x1_y0-y1_z0-z1`` counted from voxel_offset as
        Neuroglancer and tensorstore count it; the key must be the chunk's exact range."""
        sp = self._spatial(info)
        where = {
            "chunk_shape": [info.chunk_shape[a] for a in sp],
            "shape": [info.shape[a] for a in sp],
            "origin": self._offset(info, sp),
        }
        idx = wa.parse_chunk_key(key, "precomputed", **where)
        if idx is None or wa.chunk_key(idx, "precomputed", **where) != key:
            return None
        return idx if info.ndim == 3 else (0, *idx)

    def resolve(self, pipeline: Pipeline, path: str):
        path = path.strip("/")
        if path in ("info", ""):
            return Metadata.json(self._info(pipeline))
        return self.chunk_request(pipeline, path, self._key)

    def _info(self, pipeline: Pipeline) -> dict:
        info0 = pipeline.info(0)
        sp = self._spatial(info0)
        if info0.dtype.name not in _SUPPORTED:
            raise ValueError(
                f"precomputed supports {sorted(_SUPPORTED)}, got {info0.dtype.name}; add a 'cast' op"
            )
        if info0.kind is not None:
            auto = "segmentation" if info0.kind in ("label", "mask") else "image"
        else:
            auto = "segmentation" if info0.dtype.kind == "u" and info0.dtype.itemsize >= 4 else "image"
        vt = self.volume_type or auto
        scales = []
        for lvl in range(pipeline.num_levels):
            info = pipeline.info(lvl)
            # precomputed resolutions are always nanometres
            res = [info.voxel_size[a] * _UNIT_TO_M.get(info.units[a], 1e-9) * 1e9 for a in sp]
            res = res[::-1]
            off = self._offset(info, sp)[::-1]
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
