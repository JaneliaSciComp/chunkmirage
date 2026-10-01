"""GeoTIFF sources, cloud-optimized GeoTIFFs (COGs) above all: the tiled TIFFs, with their
lower resolutions inside, that most geospatial and planetary rasters are published as.

    https://.../site.tif   (or s3://, gs://, a local path)

A COG is read where it is looked at: its header once (a few range requests), then each
512 x 512 tile the pipeline asks for, decoded by tifffile and kept in a small cache. Its
pages of decreasing size (the overviews) are the levels, so a zoomed-out view reads the
small ones. One sample per pixel; the axes are ``y, x`` in the projection's units (metres),
``y`` counting down the image as rows do (minus the northing, for a north-up raster), and
voxel ``(0, 0)`` sits at the tiepoint. Float data's no-data value (``GDAL_NODATA``) is NaN.

Needs the ``tiff`` extra (tifffile, imagecodecs): tensorstore's ``tiff`` driver reads only
uint8 images.
"""

from __future__ import annotations

import io
import os
import threading
import urllib.request

import numpy as np

from chunkmirage.cache import LRUCache
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source


def is_geotiff(path: str) -> bool:
    return path.split("?")[0].lower().endswith((".tif", ".tiff"))


class _Ranges(io.RawIOBase):
    """A file of ``size`` bytes read through a tensorstore kvstore in 256 KB blocks, each
    fetched once: the header and page directories tifffile reads are many small reads."""

    BLOCK = 1 << 18

    def __init__(self, kv, key: str, size: int):
        self.kv, self.key, self.size, self.pos = kv, key, size, 0
        self.blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def bytes(self, start: int, stop: int) -> bytes:
        got = self.kv.read(self.key, byte_range=slice(start, stop)).result()
        if got.state != "value" or len(got.value) != stop - start:
            raise OSError(f"{self.key}: bytes {start}-{stop} could not be read ({got.state})")
        return bytes(got.value)

    def readinto(self, b) -> int:
        n = 0
        while n < len(b) and self.pos < self.size:
            i, o = divmod(self.pos, self.BLOCK)
            if i not in self.blocks:
                stop = min((i + 1) * self.BLOCK, self.size)
                self.blocks[i] = self.bytes(i * self.BLOCK, stop)
            piece = self.blocks[i][o : o + len(b) - n]
            b[n : n + len(piece)] = piece
            n += len(piece)
            self.pos += len(piece)
        return n


def _size(kv, key: str, url: str) -> int:
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            return int(r.headers["Content-Length"])
    if "://" not in url or url.startswith("file://"):
        return os.path.getsize(url.removeprefix("file://"))
    return len(kv.read(key).result().value)  # a bucket object: read once, whole


class GeoTIFFLevel(Source):
    """One page of a GeoTIFF: tiles fetched by range and decoded where a read covers them."""

    def __init__(self, page, ranges: _Ranges, info: ArrayInfo, key: str, cache: LRUCache):
        self.page, self.ranges, self._info, self._key, self.cache = page, ranges, info, key, cache
        self.nodata = None
        if "GDAL_NODATA" in page.tags and info.dtype.kind == "f":
            self.nodata = float(page.tags["GDAL_NODATA"].value)

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return self._key

    def _tile(self, ty: int, tx: int) -> np.ndarray:
        th, tw = self._info.chunk_shape
        across = -(-self._info.shape[1] // tw)
        i = ty * across + tx
        hit = self.cache.get((self._key, i))
        if hit is not None:
            return hit
        start, count = self.page.dataoffsets[i], self.page.databytecounts[i]
        if count == 0:  # a tile the writer left out: nothing there
            tile = np.full(
                (th, tw), np.nan if self._info.dtype.kind == "f" else 0, self._info.dtype
            )
        else:
            data, _, _ = self.page.decode(self.ranges.bytes(start, start + count), i)
            tile = np.asarray(data).reshape(th, tw)
            if self.nodata is not None and not np.isnan(self.nodata):
                tile = np.where(tile == self.nodata, np.nan, tile).astype(self._info.dtype)
        self.cache.put((self._key, i), tile)
        return tile

    def read(self, box: Box) -> np.ndarray:
        out = np.empty(box.shape, self._info.dtype)
        th, tw = self._info.chunk_shape
        for ty in range(box.start[0] // th, (box.stop[0] - 1) // th + 1):
            for tx in range(box.start[1] // tw, (box.stop[1] - 1) // tw + 1):
                y0, x0 = ty * th, tx * tw
                ys = slice(max(box.start[0], y0), min(box.stop[0], y0 + th))
                xs = slice(max(box.start[1], x0), min(box.stop[1], x0 + tw))
                tile = self._tile(ty, tx)
                out[ys.start - box.start[0] : ys.stop - box.start[0],
                    xs.start - box.start[1] : xs.stop - box.start[1]] = tile[
                    ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0
                ]  # fmt: skip
        return out


def open_geotiff(path: str, *, cache_bytes: int = 0, **_) -> MultiscaleSource:
    import tensorstore as ts
    import tifffile

    head, _, key = path.rstrip("/").rpartition("/")
    root = head + "/"  # a kvstore's keys are appended to its root as they are
    kv = ts.KvStore.open(root if "://" in head else {"driver": "file", "path": root}).result()
    ranges = _Ranges(kv, key, _size(kv, key, path))
    with _lock:
        tif = tifffile.TiffFile(io.BufferedReader(ranges, 1 << 16), name=key)
    pages = [p for p in tif.pages if p.shape[:2] and (p.subfiletype & 4) == 0]  # no masks
    base = pages[0]
    if base.samplesperpixel != 1:
        raise ValueError(f"{path}: {base.samplesperpixel} samples per pixel; one is read")
    if not base.is_tiled:
        raise ValueError(f"{path}: stored in strips, not tiles; only tiled GeoTIFFs are read")
    tags = base.tags
    sx, sy = (
        (float(v) for v in tags["ModelPixelScaleTag"].value[:2])
        if "ModelPixelScaleTag" in tags
        else (1.0, 1.0)
    )
    tie = tags["ModelTiepointTag"].value if "ModelTiepointTag" in tags else (0, 0, 0, 0, 0, 0)
    # raster (i, j) -> model (x, y): the top-left corner of pixel (0, 0)
    x0, y0 = float(tie[3]) - float(tie[0]) * sx, float(tie[4]) + float(tie[1]) * sy
    geo = tif.geotiff_metadata or {}
    projected = int(geo.get("GTModelTypeGeoKey", 1)) == 1 and "ModelPixelScaleTag" in tags
    unit = "m" if projected else ""  # a geographic raster's degrees have no viewer unit
    cache = LRUCache(max(int(cache_bytes), 256 * 2**20))
    levels = []
    for i, page in enumerate(pages):
        h, w = page.shape[:2]
        fy, fx = base.shape[0] / h, base.shape[1] / w  # the overview's pixel, in the base's
        vy, vx = sy * fy, sx * fx
        info = ArrayInfo(
            shape=(h, w),
            dtype=page.dtype,
            chunk_shape=(page.tilelength, page.tilewidth),
            voxel_size=(vy, vx),
            units=(unit, unit),
            axes=("y", "x"),
            translation=(-y0 + vy / 2, x0 + vx / 2),  # pixel centres; y is minus the northing
        )
        levels.append(GeoTIFFLevel(page, ranges, info, f"geotiff:{path}:{i}", cache))
    return MultiscaleSource(levels, name=path)


_lock = threading.Lock()
