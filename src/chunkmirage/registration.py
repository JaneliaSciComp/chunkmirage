"""Deformable registration on a GPU: the solver behind ``register://``.

Finds a smooth displacement field ``u`` in the fixed image's physical space such that the
moving image sampled at ``affine(p + u(p))`` looks like the fixed image at ``p``. The fit
maximizes local normalized cross-correlation, penalizes the field's gradient, and runs
Adam from coarse levels to fine ones. ``u`` lives on a regular control grid a few voxels
apart, so it stays small (tens of megabytes for a whole organ), and the ``scene://``
resampler serves the registered image through it at any resolution.

Needs PyTorch (``chunkmirage[gpu]``); without a GPU it runs on the CPU, slowly.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("chunkmirage")

MAX_VOXELS = 1 << 24  # fixed voxels warped at once; bounds GPU memory to a few GB
FLAT = 1e-4  # windows with less intensity variance than this (in [0, 1] units) are ignored


@dataclass
class Level:
    """One resolution level of an image, spatial axes only (C order)."""

    data: np.ndarray
    voxel_size: np.ndarray  # physical units per voxel
    translation: np.ndarray  # physical position of voxel 0's centre


@dataclass
class Grid:
    """A displacement field on a regular grid: ``values[i, j, k]`` (physical units, C order
    components) is the displacement at ``origin + (i, j, k) * spacing``."""

    values: np.ndarray  # (z, y, x, 3) float32
    origin: np.ndarray
    spacing: np.ndarray


@dataclass
class Settings:
    iterations: tuple[int, ...]  # per level, coarse to fine
    smooth: float = 1.0  # weight of the penalty on the field's gradient
    grid: float = 4.0  # control-point spacing, in voxels of each level
    window: int = 7  # correlation window, voxels (odd)
    step: float = 0.5  # Adam learning rate, in voxels of each level


def pick_device(device: str = "auto"):
    """``auto``: the GPU with the most free memory, else the CPU."""
    import torch

    if device != "auto":
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def normalize(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Intensities to [0, 1] between ``lo`` and ``hi``, clipped (bright spots saturate)."""
    return np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-12), 0, 1)


def solve(
    fixed: list[Level],
    moving: list[Level],
    affine: np.ndarray,
    settings: Settings,
    *,
    box: tuple[np.ndarray, np.ndarray],
    device: str = "auto",
    snapshots: int = 1,
) -> list[Grid]:
    """Fit ``u`` level by level (``fixed[i]`` against ``moving[i]``, coarse to fine).

    ``affine`` (4x4 or 3x4) maps fixed physical points to moving physical points; the
    moving image is sampled at ``affine(p + u(p))``. The control grid spans ``box`` (the
    fixed image's physical extent, ``(low, high)``); points outside it use the nearest
    edge value. Returns ``snapshots`` fields evenly spaced over the iterations, from the
    zero field to the final one (just the final one when ``snapshots`` is 1).
    """
    lo, hi = (np.asarray(b, dtype=float) for b in box)
    total = sum(settings.iterations)
    if total == 0:  # nothing to fit: the affine alone
        return [_grid(np.zeros((2, 2, 2, 3), np.float32), lo, hi)] * max(1, snapshots)
    import torch
    import torch.nn.functional as F

    dev = pick_device(device)
    a = torch.as_tensor(np.asarray(affine, dtype=float)[:3], dtype=torch.float32, device=dev)
    u = torch.zeros((1, 3, *_control_shape(lo, hi, fixed[0].voxel_size, settings.grid)), device=dev)
    # the iteration count after which each snapshot is taken (repeated when there are more
    # snapshots than iterations)
    n = max(snapshots, 1)
    wanted = [round(k * total / (n - 1)) for k in range(n)] if n > 1 else [total]
    out: list[Grid] = []

    def record(step: int) -> None:
        while len(out) < n and wanted[len(out)] == step:
            out.append(_grid(u, lo, hi))

    record(0)
    step = 0
    started = time.time()
    fixed_range = _range(fixed[0].data)
    moving_range = _range(moving[0].data)
    for stage, (fl, ml, iters) in enumerate(zip(fixed, moving, settings.iterations), 1):
        if iters == 0:
            continue
        shape = _control_shape(lo, hi, fl.voxel_size, settings.grid)
        if tuple(u.shape[2:]) != shape:  # refine: both grids span the box exactly
            u = F.interpolate(u, size=shape, mode="trilinear", align_corners=True)
        u = u.detach().requires_grad_(True)
        spacing = (hi - lo) / (np.asarray(shape) - 1)
        opt = _Adam(u, lr=settings.step * float(np.mean(fl.voxel_size)))
        fix = torch.as_tensor(normalize(fl.data, *fixed_range), device=dev)[None, None]
        mov = torch.as_tensor(normalize(ml.data, *moving_range), device=dev)[None, None]
        warp = _Warp(fl, ml, lo, hi, a, dev)
        slabs = _slabs(fl.data.shape, settings.window // 2)
        # Only voxels whose correlation window the moving image covers count: elsewhere
        # there is nothing to match, and fitting would drag the moving image's edge over
        # whatever lies beyond it. The field there follows from its smoothness alone.
        seen = [warp.seen(z0, z1, settings.window)[:, :, c0:c1] for z0, z1, c0, c1 in slabs]
        n_vox = math.prod(fl.data.shape)
        t0 = time.time()
        for it in range(iters):
            u.grad = None
            sim = 0.0
            for (z0, z1, c0, c1), inside in zip(slabs, seen):
                warped = warp(mov, u, z0, z1)
                cc = _lncc(fix[:, :, z0:z1], warped, settings.window)[:, :, c0:c1] * inside
                loss = -cc.sum() / n_vox
                loss.backward()
                sim -= float(loss.detach())
            reg = settings.smooth * _diffusion(u, spacing)
            reg.backward()
            opt.step()
            step += 1
            if it == 0:
                first = sim
            record(step)
        log.info(
            "register: stage %d/%d, %s voxels, grid %s, %d iterations in %.1f s: "
            "similarity %.4f -> %.4f",
            stage,
            len(fixed),
            "x".join(map(str, fl.data.shape)),
            "x".join(map(str, shape)),
            iters,
            time.time() - t0,
            first,
            sim,
        )
        del fix, mov, warp, seen
    log.info("register: solved in %.1f s on %s", time.time() - started, dev)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _range(data: np.ndarray) -> tuple[float, float]:
    step = max(1, data.size // 1_000_000)
    lo, hi = np.percentile(data.reshape(-1)[::step], [0.5, 99.5])
    return float(lo), float(hi)


def _control_shape(lo, hi, voxel_size, grid: float) -> tuple[int, ...]:
    n = np.ceil((hi - lo) / (grid * np.asarray(voxel_size, dtype=float))).astype(int) + 1
    return tuple(int(v) for v in np.maximum(n, 2))


def _grid(u, lo, hi) -> Grid:
    values = u if isinstance(u, np.ndarray) else u.detach()[0].permute(1, 2, 3, 0).cpu().numpy()
    shape = np.asarray(values.shape[:3])
    return Grid(np.ascontiguousarray(values, np.float32), lo.copy(), (hi - lo) / (shape - 1))


def _slabs(shape, halo: int) -> list[tuple[int, int, int, int]]:
    """Runs of whole z planes of at most ``MAX_VOXELS`` voxels, each padded by ``halo``
    planes so the correlation windows at its core are complete: ``(start, stop, core
    start, core stop)``, the core relative to the padded run."""
    plane = math.prod(shape[1:])
    rows = max(1, MAX_VOXELS // plane - 2 * halo)
    out = []
    for z in range(0, shape[0], rows):
        z0, z1 = max(0, z - halo), min(shape[0], z + rows + halo)
        out.append((z0, z1, z - z0, min(z + rows, shape[0]) - z0))
    return out


class _Adam:
    """Adam (Kingma and Ba) on one tensor. ``torch.optim`` would do, but importing it loads
    ``torch._dynamo``: seconds of start-up from a network file system, for nothing here."""

    def __init__(self, p, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        import torch

        self.p, self.lr, self.b1, self.b2, self.eps = p, lr, b1, b2, eps
        self.m, self.v = torch.zeros_like(p), torch.zeros_like(p)
        self.t = 0
        self.torch = torch

    def step(self) -> None:
        g, b1, b2 = self.p.grad, self.b1, self.b2
        self.t += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        m_hat = self.m / (1 - b1**self.t)
        v_hat = self.v / (1 - b2**self.t)
        with self.torch.no_grad():
            self.p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))


class _Warp:
    """The moving image sampled at ``affine(p + u(p))`` for fixed voxel centres ``p``."""

    def __init__(self, fixed: Level, moving: Level, lo, hi, affine, dev):
        import torch

        def t(v):
            return torch.as_tensor(np.asarray(v, dtype=float), dtype=torch.float32, device=dev)

        self.torch = torch
        self.shape = fixed.data.shape
        self.vf, self.tf = t(fixed.voxel_size), t(fixed.translation)
        self.lo, self.span = t(lo), t(hi - lo)
        self.a = affine
        self.vm, self.tm = t(moving.voxel_size), t(moving.translation)
        self.mshape = t(np.maximum(np.asarray(moving.data.shape) - 1, 1))
        self.dev = dev

    def _points(self, z0: int, z1: int):
        """Physical positions of the fixed voxel centres in planes ``z0:z1``, (d, h, w, 3)."""
        torch = self.torch
        axes = [
            torch.arange(z0, z1, device=self.dev, dtype=torch.float32),
            torch.arange(self.shape[1], device=self.dev, dtype=torch.float32),
            torch.arange(self.shape[2], device=self.dev, dtype=torch.float32),
        ]
        p = [ax * self.vf[i] + self.tf[i] for i, ax in enumerate(axes)]
        return torch.stack(torch.meshgrid(*p, indexing="ij"), dim=-1)

    def seen(self, z0: int, z1: int, window: int):
        """Which fixed voxels in planes ``z0:z1`` have their whole ``window`` (as far as the
        fixed image reaches) inside the moving image under the affine alone: a
        (1, 1, d, h, w) float mask."""
        import torch.nn.functional as F

        idx = (self._points(z0, z1) @ self.a[:, :3].T + self.a[:, 3] - self.tm) / self.vm
        ok = ((idx >= 0) & (idx <= self.mshape)).all(dim=-1)[None, None].float()
        h = window // 2
        ok = F.pad(ok, (h,) * 6, value=1.0)  # beyond the fixed image: no data to miss
        for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # a box minimum
            ok = -F.max_pool3d(-ok, k, stride=1)
        return ok

    def __call__(self, mov, u, z0: int, z1: int):
        import torch.nn.functional as F

        d, h, w = z1 - z0, self.shape[1], self.shape[2]
        p = self._points(z0, z1)
        g = 2 * (p - self.lo) / self.span - 1  # into the control grid, grid_sample wants x,y,z
        disp = F.grid_sample(
            u, g.flip(-1)[None], mode="bilinear", padding_mode="border", align_corners=True
        )[0].permute(1, 2, 3, 0)
        q = (p + disp) @ self.a[:, :3].T + self.a[:, 3]  # moving physical
        idx = (q - self.tm) / self.vm
        g = 2 * idx / self.mshape - 1
        return F.grid_sample(  # 0 beyond the moving image, as the served image has
            mov, g.flip(-1)[None], mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(1, 1, d, h, w)


def _lncc(i, j, window: int):
    """Squared local normalized cross-correlation per voxel, in [0, 1]."""
    import torch
    import torch.nn.functional as F

    x = torch.cat([i, j, i * i, j * j, i * j], dim=1)
    h = window // 2
    for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # a box mean, one axis at a time
        pad = tuple(h if n > 1 else 0 for n in k)
        x = F.avg_pool3d(x, k, stride=1, padding=pad, count_include_pad=False)
    mi, mj, ii, jj, ij = x.unbind(1)
    cov = ij - mi * mj
    vi, vj = ii - mi * mi, jj - mj * mj
    # Windows without contrast in either image carry no signal and are left out (as in
    # ANTs), not damped with a constant: that would reward warps that add contrast.
    ok = (vi > FLAT) & (vj > FLAT)
    den = torch.where(ok, vi * vj, torch.ones_like(vi))
    return torch.where(ok, cov * cov / den, torch.zeros_like(vi))[:, None]


def _diffusion(u, spacing):
    """Mean squared gradient of the field (dimensionless: displacement per distance)."""
    total = 0.0
    for dim, sp in zip((2, 3, 4), spacing):
        d = u.diff(dim=dim) / float(sp)
        total = total + (d * d).mean()
    return total
