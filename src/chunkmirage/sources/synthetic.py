"""Procedurally generated sources: arbitrarily large, nothing on disk, exact at every scale.

URL form (accepted by ``open_source``)::

    synthetic://blobs?shape=4096,4096,4096&chunk=64,64,64&levels=5&seed=0&voxel_size=8
    synthetic://noise?...        fractal (fBm) value noise
    synthetic://shells?...       hollow spheres (membrane-like)
    synthetic://julia?...        3-D slice of a quaternion Julia set
    synthetic://blobs+noise?...  sum of kinds

Every voxel is a deterministic function of its world coordinate, so level *i* is simply
the same function sampled with a 2**i voxel spacing: ``s1[z, y, x] == s0[2z, 2y, 2x]``.
That makes the pyramid free and exactly self-consistent, unlike averaging.

Generation is numpy on ~260k voxels per chunk; numpy releases the GIL for those
operations, so the server's threadpool already computes chunks on all cores.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source

_KINDS = ("blobs", "noise", "julia", "shells")
F32 = np.float32


def _hash3(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint32 hash of integer lattice coordinates -> float32 in [0, 1)."""
    h = (
        np.asarray(x).astype(np.uint32) * np.uint32(73856093)
        ^ np.asarray(y).astype(np.uint32) * np.uint32(19349663)
        ^ np.asarray(z).astype(np.uint32) * np.uint32(83492791)
        ^ np.uint32((seed * 2654435761) & 0xFFFFFFFF)
    )
    h ^= h >> np.uint32(16)
    h *= np.uint32(0x7FEB352D)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x846CA68B)
    h ^= h >> np.uint32(16)
    return h.astype(F32) * F32(1.0 / 4294967296.0)


def _lattice(z, y, x, cell: float):
    """Per-voxel lattice cell indices (int) and fractional offsets, plus the covered ranges."""
    fz, fy, fx = (np.asarray(v, dtype=F32) / F32(cell) for v in (z, y, x))
    iz, iy, ix = (np.floor(f).astype(np.int64) for f in (fz, fy, fx))
    tz, ty, tx = fz - iz, fy - iy, fx - ix
    rng = [(int(i.min()) - 1, int(i.max()) + 2) for i in (iz, iy, ix)]  # +2: room for corner+1
    return (iz, iy, ix), (tz, ty, tx), rng


def _grid(rng, seed):
    """Hash values on the small lattice grid covering ``rng`` (inclusive lows, exclusive highs)."""
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    return _hash3(gx, gy, gz, seed), (z0, y0, x0)


def _value_noise(z, y, x, cell: float, seed: int) -> np.ndarray:
    """Trilinear lattice noise. Hashes the (small) lattice once, then gathers 8 corners."""
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z, y, x, cell)
    vals, (z0, y0, x0) = _grid(rng, seed)
    tz, ty, tx = (t * t * (3 - 2 * t) for t in (tz, ty, tx))  # smoothstep
    iz, iy, ix = iz - z0, iy - y0, ix - x0
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    for dz in (0, 1):
        wz = tz if dz else 1 - tz
        for dy in (0, 1):
            wy = ty if dy else 1 - ty
            for dx in (0, 1):
                wx = tx if dx else 1 - tx
                out += wz * wy * wx * vals[iz + dz, iy + dy, ix + dx]
    return out


def _fbm(z, y, x, seed: int, base_cell: float = 256.0, octaves: int = 2) -> np.ndarray:
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    amp, total, cell = 1.0, 0.0, base_cell
    for o in range(octaves):
        out += F32(amp) * _value_noise(z, y, x, cell, seed + o)
        total += amp
        amp *= 0.5
        cell /= 2
    return out / F32(total)


# Blobs / shells: one lattice cell holds `per_cell` random spheres whose influence is cut off
# at `reach` <= cell/2. Two evaluation strategies with identical results: loop over cells and
# evaluate each sphere on its footprint (cheap when a chunk spans few cells, i.e. fine
# levels), or gather per voxel from the 8 nearest cells (bounded cost at coarse levels; a
# sphere reaching at most cell/2 can only touch voxels in the 2x2x2 cells nearest to them).
_CELL_LOOP_MAX = 216  # switch strategy above this many cells


def _cell_params(gz, gy, gx, seed: int, k: int, cell: float, kind: str):
    s = seed * 131 + k * 7
    u = [_hash3(gx, gy, gz, s + i) for i in range(5)]
    if kind == "blobs":
        center = ((gz + u[0]) * cell, (gy + u[1]) * cell, (gx + u[2]) * cell)
        radius = 8 + u[3] * (cell / 6 - 8)  # 8 .. cell/6, so reach = 3r <= cell/2
        amp = 90 + u[4] * 130
        reach = 3 * radius
    else:  # shells: centred, larger, one per cell; reach = r + 10 <= cell/2
        center = tuple((g + 0.5 + (uu - 0.5) * 0.4) * cell for g, uu in zip((gz, gy, gx), u[:3]))
        radius = cell * (0.22 + 0.15 * u[3])
        amp = np.full_like(radius, 200)
        reach = radius + 10
    f = lambda a: np.asarray(a, dtype=F32)  # noqa: E731
    return tuple(f(c) for c in center), f(radius), f(amp), f(reach)


def _sphere_field(d2, radius, amp, reach, kind):
    inside = d2 <= reach * reach
    if kind == "blobs":
        return inside * amp * np.exp(-d2 / (2 * radius * radius))
    d = np.sqrt(d2)
    return inside * (amp * np.exp(-((d - radius) ** 2) / F32(2 * 2.5**2)) + F32(40) * (d < radius))


def _spheres(z, y, x, seed: int, kind: str) -> np.ndarray:
    cell, per_cell = (128.0, 2) if kind == "blobs" else (160.0, 1)
    z32, y32, x32 = (np.asarray(v, dtype=F32) for v in (z, y, x))
    out = np.zeros(np.broadcast(z, y, x).shape, dtype=F32)
    (iz, iy, ix), (tz, ty, tx), rng = _lattice(z32, y32, x32, cell)
    (z0, z1), (y0, y1), (x0, x1) = rng
    gz, gy, gx = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    n_cells = gz.size
    # nearest-cell base index per voxel: floor(f - 0.5), relative to the grid origin
    nz, ny, nx = (i - (t < 0.5) for i, t in ((iz, tz), (iy, ty), (ix, tx)))
    for k in range(per_cell):
        (cz, cy, cx), rad, amp, reach = _cell_params(gz, gy, gx, seed, k, cell, kind)
        if n_cells <= _CELL_LOOP_MAX:
            # loop over cells, evaluate each sphere only on its footprint
            for i in np.ndindex(gz.shape):
                c0, c1, c2, r, a, rc = cz[i], cy[i], cx[i], rad[i], amp[i], reach[i]
                sl = []
                for coord, c in zip((z32.ravel(), y32.ravel(), x32.ravel()), (c0, c1, c2)):
                    lo = int(np.searchsorted(coord, c - rc))
                    hi = int(np.searchsorted(coord, c + rc, side="right"))
                    if hi <= lo:
                        break
                    sl.append(slice(lo, hi))
                if len(sl) < 3:
                    continue
                zz, yy, xx = z32[sl[0]], y32[:, sl[1]], x32[:, :, sl[2]]
                d2 = (zz - c0) ** 2 + (yy - c1) ** 2 + (xx - c2) ** 2
                out[tuple(sl)] += _sphere_field(d2, r, a, rc, kind)
        else:
            # gather from the 8 nearest cells for every voxel
            for dz in (0, 1):
                jz = nz - z0 + dz
                for dy in (0, 1):
                    jy = ny - y0 + dy
                    for dx in (0, 1):
                        jx = nx - x0 + dx
                        jz_, jy_, jx_ = np.broadcast_arrays(jz, jy, jx)
                        d2 = (
                            (z32 - cz[jz_, jy_, jx_]) ** 2
                            + (y32 - cy[jz_, jy_, jx_]) ** 2
                            + (x32 - cx[jz_, jy_, jx_]) ** 2
                        )
                        out += _sphere_field(
                            d2, rad[jz_, jy_, jx_], amp[jz_, jy_, jx_], reach[jz_, jy_, jx_], kind
                        )
    return out


def _julia(z, y, x, seed: int, scale: float, max_iter: int = 14) -> np.ndarray:
    """Escape-time of a quaternion Julia set sampled on a 3-D slice (w=0). Bright = slow escape."""
    rng = np.random.default_rng(seed)
    c = (rng.uniform(-0.8, 0.4, 4) * np.array([1, 1, 1, 0.3])).astype(F32)
    qx, qy, qz = np.broadcast_arrays(
        *(np.asarray(v, dtype=F32) / F32(scale) - F32(1.4) for v in (x, y, z))
    )
    qx, qy, qz = qx.copy(), qy.copy(), qz.copy()
    qw = np.zeros_like(qx)
    count = np.zeros(qx.shape, dtype=F32)
    alive = np.ones(qx.shape, dtype=bool)
    for _ in range(max_iter):
        nx = qx * qx - qy * qy - qz * qz - qw * qw + c[0]
        ny = 2 * qx * qy + c[1]
        nz = 2 * qx * qz + c[2]
        nw = 2 * qx * qw + c[3]
        alive &= (nx * nx + ny * ny + nz * nz + nw * nw) < 16
        qx, qy, qz, qw = (np.where(alive, v, F32(0)) for v in (nx, ny, nz, nw))
        count += alive
    return F32(255.0 / max_iter) * count


def generate(
    kinds: tuple[str, ...], full_shape: tuple[int, ...], level: int, seed: int, start, stop
) -> np.ndarray:
    """Pure function: voxels of ``kinds`` for box [start, stop) at ``level``."""
    step = 2**level
    z = (np.arange(start[0], stop[0]) * step)[:, None, None].astype(F32)
    y = (np.arange(start[1], stop[1]) * step)[None, :, None].astype(F32)
    x = (np.arange(start[2], stop[2]) * step)[None, None, :].astype(F32)
    shape = tuple(b - a for a, b in zip(start, stop))
    out = np.zeros(shape, dtype=F32)
    for kind in kinds:
        if kind == "blobs" or kind == "shells":
            out += _spheres(z, y, x, seed, kind)
        elif kind == "noise":
            out += F32(80) * _fbm(z, y, x, seed)
        elif kind == "julia":
            out += _julia(z, y, x, seed, scale=max(full_shape) / 2.8)
    return np.clip(out, 0, 255).astype(np.uint8)


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
        self.kinds = tuple(kinds)
        self.level = level
        self.seed = seed
        self.step = 2**level
        self._full = tuple(int(s) for s in shape)
        self._info = ArrayInfo(
            shape=tuple(-(-s // self.step) for s in self._full),
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
        return generate(
            self.kinds, self._full, self.level, self.seed, tuple(box.start), tuple(box.stop)
        )


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
