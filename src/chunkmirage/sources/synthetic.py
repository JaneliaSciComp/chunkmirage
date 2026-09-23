"""Procedurally generated sources: arbitrarily large, nothing on disk, exact at every scale.

URL form (accepted by ``open_source``)::

    synthetic://blobs?shape=4096,4096,4096&chunk=64,64,64&levels=5&seed=0&voxel_size=8
    synthetic://noise?...        fractal (fBm) value noise
    synthetic://julia?...        3-D slice of a quaternion Julia set
    synthetic://blobs+noise?...  sum of kinds

Every voxel is a deterministic function of its world coordinate, so level *i* is simply
the same function sampled with a 2**i voxel spacing: ``s1[z, y, x] == s0[2z, 2y, 2x]``.
That makes the pyramid free and exactly self-consistent, unlike averaging.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

_KINDS = ("blobs", "noise", "julia", "shells")


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint32 hash of integer lattice coordinates -> float in [0, 1)."""
    h = (
        x.astype(np.uint32) * np.uint32(73856093)
        ^ y.astype(np.uint32) * np.uint32(19349663)
        ^ z.astype(np.uint32) * np.uint32(83492791)
        ^ np.uint32(seed * 2654435761 & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h.astype(np.float32) * np.float32(1.0 / 4294967296.0)


def _value_noise(z, y, x, cell: float, seed: int) -> np.ndarray:
    """Trilinearly interpolated lattice noise with lattice spacing ``cell`` (world units)."""
    fz, fy, fx = (np.asarray(v, dtype=np.float32) / np.float32(cell) for v in (z, y, x))
    iz, iy, ix = (
        np.floor(fz).astype(np.int64),
        np.floor(fy).astype(np.int64),
        np.floor(fx).astype(np.int64),
    )
    tz, ty, tx = fz - iz, fy - iy, fx - ix
    # smoothstep for C1 continuity
    tz, ty, tx = (t * t * (3 - 2 * t) for t in (tz, ty, tx))
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=np.float32)
    for dz in (0, 1):
        wz = tz if dz else 1 - tz
        for dy in (0, 1):
            wy = ty if dy else 1 - ty
            for dx in (0, 1):
                wx = tx if dx else 1 - tx
                out += wz * wy * wx * _hash3(ix + dx, iy + dy, iz + dz, seed)
    return out


def _fbm(z, y, x, seed: int, base_cell: float = 256.0, octaves: int = 4) -> np.ndarray:
    out = np.zeros(np.broadcast(z, y, x).shape)
    amp, total, cell = 1.0, 0.0, base_cell
    for o in range(octaves):
        out += amp * _value_noise(z, y, x, cell, seed + o)
        total += amp
        amp *= 0.5
        cell /= 2
    return out / total


def _footprint(z, y, x, center, reach):
    """Index slices of the (broadcast) grid within ``reach`` of ``center``, or None if empty."""
    sl = []
    for coord, c in zip((z, y, x), center):
        v = coord.ravel()
        lo = int(np.searchsorted(v, c - reach, side="left"))
        hi = int(np.searchsorted(v, c + reach, side="right"))
        if hi <= lo:
            return None
        sl.append(slice(lo, hi))
    return tuple(sl)


def _cells(z, y, x, cell):
    rng = []
    for coord in (z, y, x):
        lo = int(np.floor(coord.min() / cell)) - 1
        hi = int(np.floor(coord.max() / cell)) + 1
        rng.append(range(lo, hi + 1))
    return rng


def _blobs(z, y, x, seed: int, cell: float = 96.0, per_cell: int = 2) -> np.ndarray:
    """Sum of Gaussian blobs, ``per_cell`` random blobs per lattice cell, radius 8..cell/3."""
    out = np.zeros(np.broadcast(z, y, x).shape)
    rz, ry, rx = _cells(z, y, x, cell)
    for gz in rz:
        for gy in ry:
            for gx in rx:
                g = (np.array([gx]), np.array([gy]), np.array([gz]))
                for k in range(per_cell):
                    s = seed * 131 + k * 7
                    u = [_hash3(*g, s + i)[0] for i in range(5)]
                    center = ((gz + u[0]) * cell, (gy + u[1]) * cell, (gx + u[2]) * cell)
                    r = 8 + u[3] * (cell / 3 - 8)
                    amp = 90 + u[4] * 130
                    fp = _footprint(z, y, x, center, 3 * r)
                    if fp is None:
                        continue
                    zz, yy, xx = z[fp[0]], y[:, fp[1]], x[:, :, fp[2]]
                    d2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
                    out[fp] += amp * np.exp(-d2 / (2 * r * r))
    return out


def _shells(z, y, x, seed: int, cell: float = 160.0) -> np.ndarray:
    """Hollow spheres (membrane-like), one per lattice cell: bright thin shell, darker interior."""
    out = np.zeros(np.broadcast(z, y, x).shape)
    rz, ry, rx = _cells(z, y, x, cell)
    for gz in rz:
        for gy in ry:
            for gx in rx:
                g = (np.array([gx]), np.array([gy]), np.array([gz]))
                u = [_hash3(*g, seed + i)[0] for i in range(4)]
                center = (
                    (gz + 0.5 + (u[0] - 0.5) * 0.4) * cell,
                    (gy + 0.5 + (u[1] - 0.5) * 0.4) * cell,
                    (gx + 0.5 + (u[2] - 0.5) * 0.4) * cell,
                )
                r = cell * (0.22 + 0.15 * u[3])
                fp = _footprint(z, y, x, center, r + 10)
                if fp is None:
                    continue
                zz, yy, xx = z[fp[0]], y[:, fp[1]], x[:, :, fp[2]]
                d = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
                out[fp] += 200 * np.exp(-((d - r) ** 2) / (2 * 2.5**2)) + 40 * (d < r)
    return out


def _julia(z, y, x, seed: int, scale: float, max_iter: int = 14) -> np.ndarray:
    """Escape-time of a quaternion Julia set sampled on a 3-D slice (w=0). Bright = slow escape."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.8, 0.4, 4) * np.array([1, 1, 1, 0.3])
    qx = (np.asarray(x) / scale - 1.4).astype(np.float64)
    qy = (np.asarray(y) / scale - 1.4).astype(np.float64)
    qz = (np.asarray(z) / scale - 1.4).astype(np.float64)
    qx, qy, qz = np.broadcast_arrays(qx, qy, qz)
    qx, qy, qz = qx.copy(), qy.copy(), qz.copy()
    qw = np.zeros_like(qx)
    count = np.zeros(qx.shape, dtype=np.float64)
    alive = np.ones(qx.shape, dtype=bool)
    for _ in range(max_iter):
        nx = qx * qx - qy * qy - qz * qz - qw * qw + c[0]
        ny = 2 * qx * qy + c[1]
        nz = 2 * qx * qz + c[2]
        nw = 2 * qx * qw + c[3]
        alive &= (nx * nx + ny * ny + nz * nz + nw * nw) < 16
        # freeze escaped points at 0 so they cannot overflow on later iterations
        qx, qy, qz, qw = (np.where(alive, v, 0.0) for v in (nx, ny, nz, nw))
        count += alive
    return 255.0 * count / max_iter


class SyntheticSource(Source):
    def __init__(
        self,
        kinds: list[str],
        shape,
        chunk_shape,
        level: int,
        seed: int,
        voxel_size: float,
        unit: str,
    ):
        self.kinds = kinds
        self.level = level
        self.seed = seed
        self.step = 2**level
        full = tuple(int(s) for s in shape)
        self._full = full
        self._info = ArrayInfo(
            shape=tuple(-(-s // self.step) for s in full),
            dtype=np.uint8,
            chunk_shape=tuple(chunk_shape),
            voxel_size=(voxel_size * self.step,) * 3,
            units=(unit,) * 3,
            axes=("z", "y", "x"),
        )

    @property
    def info(self) -> ArrayInfo:
        return self._info

    def cache_key(self) -> str:
        return f"synthetic:{'+'.join(self.kinds)}:{self._full}:{self.seed}:s{self.level}"

    def read(self, box: Box) -> np.ndarray:
        # world coordinates (in s0 voxels) of the requested voxels at this level
        z = (np.arange(box.start[0], box.stop[0]) * self.step)[:, None, None].astype(np.float64)
        y = (np.arange(box.start[1], box.stop[1]) * self.step)[None, :, None].astype(np.float64)
        x = (np.arange(box.start[2], box.stop[2]) * self.step)[None, None, :].astype(np.float64)
        out = np.zeros(box.shape)
        for kind in self.kinds:
            if kind == "blobs":
                out += _blobs(z, y, x, self.seed)
            elif kind == "noise":
                out += 80 * _fbm(z, y, x, self.seed)
            elif kind == "shells":
                out += _shells(z, y, x, self.seed)
            elif kind == "julia":
                out += _julia(z, y, x, self.seed, scale=max(self._full) / 2.8)
        return np.clip(out, 0, 255).astype(np.uint8)


def open_synthetic(url: str) -> MultiscaleSource:
    parts = urlsplit(url)
    kinds = [k for k in parts.netloc.split("+") if k]
    for k in kinds:
        if k not in _KINDS:
            raise ValueError(f"unknown synthetic kind {k!r}; choose from {_KINDS}")
    q = {k: v[-1] for k, v in parse_qs(parts.query).items()}
    shape = tuple(int(s) for s in q.get("shape", "1024,1024,1024").split(","))
    chunk = tuple(int(s) for s in q.get("chunk", "64,64,64").split(","))
    levels = int(q.get("levels", "0")) or max(1, int(np.ceil(np.log2(max(shape) / 256))) + 1)
    seed = int(q.get("seed", "0"))
    voxel_size = float(q.get("voxel_size", "8"))
    unit = q.get("unit", "nm")
    if len(shape) != 3:
        raise ValueError("synthetic sources are 3-D: shape=z,y,x")
    src = [
        SyntheticSource(kinds or ["blobs"], shape, chunk, lvl, seed, voxel_size, unit)
        for lvl in range(levels)
    ]
    return MultiscaleSource(src, name=url)
