"""Chunk compressors shared by zarr v2 / v3 frontends, described in both metadata dialects."""

from __future__ import annotations

import numcodecs
import numpy as np


class Compressor:
    def __init__(self, kind: str = "gzip", level: int | None = None):
        self.kind = kind
        if kind == "gzip":
            self.level = 5 if level is None else level
            self._codec = numcodecs.GZip(level=self.level)
        elif kind == "zstd":
            self.level = 3 if level is None else level
            self._codec = numcodecs.Zstd(level=self.level)
        elif kind == "blosc":
            self.level = 3 if level is None else level
            self._codec = numcodecs.Blosc(
                cname="zstd", clevel=self.level, shuffle=numcodecs.Blosc.SHUFFLE
            )
        elif kind in ("none", "raw", None):
            self.kind = "none"
            self._codec = None
        else:
            raise ValueError(f"unknown compressor {kind!r}")

    def encode(self, data: bytes) -> bytes:
        if self._codec is None:
            return data
        return bytes(self._codec.encode(data))

    def zarr2_meta(self) -> dict | None:
        if self.kind == "none":
            return None
        if self.kind == "gzip":
            return {"id": "gzip", "level": self.level}
        if self.kind == "zstd":
            return {"id": "zstd", "level": self.level}
        return {"id": "blosc", "cname": "zstd", "clevel": self.level, "shuffle": 1, "blocksize": 0}

    def zarr3_codecs(self, dtype: np.dtype) -> list[dict]:
        codecs: list[dict] = [{"name": "bytes", "configuration": {"endian": "little"}}]
        if self.kind == "gzip":
            codecs.append({"name": "gzip", "configuration": {"level": self.level}})
        elif self.kind == "zstd":
            codecs.append(
                {"name": "zstd", "configuration": {"level": self.level, "checksum": False}}
            )
        elif self.kind == "blosc":
            codecs.append(
                {
                    "name": "blosc",
                    "configuration": {
                        "cname": "zstd",
                        "clevel": self.level,
                        "shuffle": "shuffle",
                        "typesize": int(dtype.itemsize),
                        "blocksize": 0,
                    },
                }
            )
        return codecs

    def n5_meta(self) -> dict:
        if self.kind == "gzip":
            return {"type": "gzip", "useZlib": False, "level": self.level}
        if self.kind == "zstd":
            return {"type": "zstd", "level": self.level}
        if self.kind == "blosc":
            return {
                "type": "blosc",
                "cname": "zstd",
                "clevel": self.level,
                "shuffle": 1,
                "blocksize": 0,
                "nthreads": 1,
            }
        return {"type": "raw"}
