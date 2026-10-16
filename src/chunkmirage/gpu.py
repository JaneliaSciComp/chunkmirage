"""The resampler on a GPU: ``warp://`` and ``scene://`` push every output voxel through a
chain of affines and sampled displacement fields and interpolate the image there, which on
the CPU is most of a registration's live view (sampling the fields above all). With a device
set (``set_device``, ``serve --device``) that chain runs in PyTorch: the points, each field's
window and the image's window on the device, interpolated linearly as ``map_coordinates``
does (clamped at the window's edge), in single precision, so values agree with the CPU's to
within a grey level and the cache keys are the same. Only chains of affines and sampled
fields move; a slab whose chain has anything else (a swirl, an inverse estimated
numerically, coordinates read from a field) is resampled on the CPU as before.
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from chunkmirage.core import Box
from chunkmirage.transforms import Affine, Displacements, Sequence, Transform, VectorField

if TYPE_CHECKING:
    from chunkmirage.sources.scene import SceneLevelSource

_device: Any = None  # a torch.device, or None: numpy and scipy


def set_device(device: str) -> None:
    """Where the resampler runs: ``cpu`` (numpy and scipy, the default), ``cuda`` or
    ``cuda:N`` (PyTorch on that GPU), or ``auto`` (the first GPU if PyTorch sees one, else
    the CPU)."""
    global _device
    if device == "cpu":
        _device = None
        return
    try:
        import torch
    except ImportError as e:
        if device == "auto":
            _device = None
            return
        raise ValueError(f"device {device!r} needs PyTorch (the gpu extra)") from e
    if device == "auto":
        _device = torch.device("cuda") if torch.cuda.is_available() else None
        return
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(f"device {device!r}: expected cpu, auto, cuda or cuda:N, with a GPU PyTorch sees")
    _device = torch.device(device)


def device() -> Any:
    """The torch device the resampler runs on, or None for the CPU."""
    return _device


def _steps(mapping: Transform) -> list | None:
    parts = mapping.parts if isinstance(mapping, Sequence) else [mapping]
    for p in parts:
        sampled = isinstance(p, Displacements) and isinstance(p.field, VectorField) and p.field.order == 1
        if not (isinstance(p, Affine) or sampled):
            return None
    return parts


def _affine(a: Affine, pts):
    import torch

    lin, off = a.linear, a.offset
    cols = []
    for j in range(a.ndim_out):
        col = pts[:, 0] * float(lin[j, 0])
        for i in range(1, a.ndim_in):
            col = col + pts[:, i] * float(lin[j, i])
        cols.append(col + float(off[j]))
    return torch.stack(cols, 1)


def _linear(win, local):
    """``win`` (channels, *shape) at ``local`` (points, axes) in its voxels, linearly, the
    window's edge repeated beyond it: (channels, points)."""
    import torch

    shape = win.shape[1:]
    flat = win.reshape(win.shape[0], -1)
    i0 = torch.floor(local)
    frac = local - i0
    i0 = i0.long()
    hi = torch.tensor(shape, device=local.device) - 1
    lo_idx, hi_idx = torch.clamp(i0, min=0), torch.minimum(torch.clamp(i0 + 1, min=0), hi)
    lo_idx = torch.minimum(lo_idx, hi)
    strides = [math.prod(shape[a + 1 :]) for a in range(len(shape))]
    out = None
    for corner in product((0, 1), repeat=len(shape)):
        idx, w = None, None
        for a, c in enumerate(corner):
            part = (hi_idx if c else lo_idx)[:, a] * strides[a]
            wa = frac[:, a] if c else 1 - frac[:, a]
            idx = part if idx is None else idx + part
            w = wa if w is None else w * wa
        term = flat[:, idx] * w
        out = term if out is None else out + term
    return out


def _bounds(c) -> tuple[np.ndarray, np.ndarray]:
    """The smallest and largest of each axis over the finite rows of ``c``, on the CPU."""
    import torch

    finite = torch.isfinite(c).all(1)
    if not bool(finite.all()):
        c = c[finite]
    if c.shape[0] == 0:
        return np.zeros(0), np.zeros(0)
    return c.min(0).values.cpu().numpy(), c.max(0).values.cpu().numpy()


def _sample(field: VectorField, pts):
    """``field.sample`` on the device: (points, components)."""
    import torch

    g = _affine(field.to_grid, pts)
    top = torch.tensor(field.grid_shape, device=g.device, dtype=g.dtype) - 1
    g = torch.minimum(torch.clamp(g, min=0), top)
    gmin, gmax = _bounds(g)
    if gmin.size == 0:
        return torch.full((pts.shape[0], field.ncomp), float("nan"), device=g.device)
    lo = np.floor(gmin).astype(int)
    stop = np.minimum(np.floor(gmax).astype(int) + 2, field.grid_shape)
    slices = [slice(int(a), int(b)) for a, b in zip(lo, stop)]
    slices.insert(field.vector_axis, slice(None))
    win = np.moveaxis(np.asarray(field._read(tuple(slices)), dtype=np.float32), field.vector_axis, 0)
    local = g - torch.tensor(lo, device=g.device, dtype=g.dtype)
    return _linear(torch.from_numpy(np.ascontiguousarray(win)).to(g.device), local).T


def resample(level: SceneLevelSource, lead: Box, slab: Box) -> np.ndarray | None:
    """``level._resample(lead, slab)`` on the device, or None if its chain has a step that
    only runs on the CPU."""
    import torch

    from chunkmirage.sources.scene import MAX_WINDOW_BYTES

    steps = _steps(level.mapping)
    if steps is None:
        return None
    dev = _device
    axes = [torch.arange(a, b, device=dev, dtype=torch.float32) for a, b in zip(slab.start, slab.stop)]
    pts = torch.stack([g.reshape(-1) for g in torch.meshgrid(*axes, indexing="ij")], 1)
    for step in steps:
        pts = _affine(step, pts) if isinstance(step, Affine) else pts + _sample(step.field, pts)
    n = pts.shape[0]
    size = torch.tensor(level._src_spatial, device=dev, dtype=pts.dtype)
    inside = ((pts >= -0.5) & (pts < size - 0.5)).all(1)
    res = np.zeros((*lead.shape, n), dtype=level.info.dtype)
    if not bool(inside.any()):
        return res.reshape(*lead.shape, *slab.shape)
    c = pts[inside]
    if level.order == 0:
        c = torch.floor(c + 0.5)
    cmin, cmax = _bounds(c)
    lo = np.floor(cmin).astype(int)
    hi = np.floor(cmax).astype(int) + (1 if level.order == 0 else 2)
    lo = np.clip(lo, 0, level._src_spatial.astype(int))
    hi = np.clip(hi, 0, level._src_spatial.astype(int))
    window = Box(tuple(lead.start) + tuple(int(v) for v in lo), tuple(lead.stop) + tuple(int(v) for v in hi))
    nbytes = math.prod(window.shape) * level.src.info.dtype.itemsize
    if nbytes > MAX_WINDOW_BYTES:
        raise MemoryError(
            f"{level.cache_key()}: output region {slab} would read {nbytes / 2**30:.1f} GiB of the "
            f"source (window {window.shape}); use smaller output chunks (chunk=...)."
        )
    data = level.src.read(window)
    local = c - torch.tensor(lo, device=dev, dtype=c.dtype)
    mask = inside.cpu().numpy()
    if level.order == 0:  # the nearest voxel, gathered on the CPU in the image's own dtype
        idx = local.long().cpu().numpy()
        res[..., mask] = data[(Ellipsis, *idx.T)]
    else:
        k = len(slab.shape)
        win = torch.from_numpy(np.ascontiguousarray(data.reshape(-1, *data.shape[data.ndim - k :]), dtype=np.float32))
        vals = _linear(win.to(dev), local).cpu().numpy().reshape(*lead.shape, -1)
        if np.issubdtype(level.info.dtype, np.integer):
            lim = np.iinfo(level.info.dtype)
            vals = np.clip(np.rint(vals), lim.min, lim.max)
        res[..., mask] = vals
    return res.reshape(*lead.shape, *slab.shape)
