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
MAX_FIT_VOXELS = 300_000  # the affine search's finest copy of the images
MIN_FIT_SIZE = 12  # its coarsest copy: at least this many voxels on every axis
FIT_ITERATIONS = (150, 100)  # per copy, coarse to fine (the first alone when there is one)
LR_MATRIX = 2e-3  # Adam steps of the affine's matrix, and of its centre in voxels
LR_CENTRE = 0.4


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
    window: tuple[int, ...] = (7,)  # correlation window per level, voxels (odd)
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
    init: np.ndarray | None = None,
    ranges: tuple[tuple[float, float], tuple[float, float]] | None = None,
    label: str = "",
) -> list[Grid]:
    """Fit ``u`` level by level (``fixed[i]`` against ``moving[i]``, coarse to fine).

    ``affine`` (4x4 or 3x4) maps fixed physical points to moving physical points; the
    moving image is sampled at ``affine(p + u(p))``. The control grid spans ``box`` (the
    fixed image's physical extent, ``(low, high)``); points outside it use the nearest
    edge value. Returns ``snapshots`` fields evenly spaced over the iterations, from the
    starting field to the final one (just the final one when ``snapshots`` is 1).

    ``init`` starts from a field given at the first level's control points, ``(z, y, x,
    3)``, rather than from zero; ``ranges`` normalizes the images with the given
    intensity ranges (fixed, moving) rather than the first level's percentiles, so blocks
    of one image are all scaled alike; ``label`` names the fit in the log.
    """
    lo, hi = (np.asarray(b, dtype=float) for b in box)
    total = sum(settings.iterations)
    if total == 0:  # nothing to fit: the affine alone, or the starting field
        values = np.zeros((2, 2, 2, 3), np.float32) if init is None else init
        return [_grid(values, lo, hi)] * max(1, snapshots)
    if len(settings.iterations) != len(fixed) or len(settings.window) != len(fixed):
        raise ValueError(
            f"settings need one iteration count and one window per level ({len(fixed)})"
        )
    import torch
    import torch.nn.functional as F

    dev = pick_device(device)
    a = torch.as_tensor(np.asarray(affine, dtype=float)[:3], dtype=torch.float32, device=dev)
    if init is None:
        shape = _control_shape(lo, hi, fixed[0].voxel_size, settings.grid)
        u = torch.zeros((1, 3, *shape), device=dev)
    else:
        u = torch.as_tensor(init, dtype=torch.float32, device=dev).permute(3, 0, 1, 2)[None]
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
    fixed_range, moving_range = ranges or (_range(fixed[0].data), _range(moving[0].data))
    per_level = zip(fixed, moving, settings.iterations, settings.window)
    for stage, (fl, ml, iters, window) in enumerate(per_level, 1):
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
        slabs = _slabs(fl.data.shape, window // 2)
        # Only voxels whose correlation window the moving image covers (under the starting
        # field) count: elsewhere there is nothing to match, and fitting would drag the
        # moving image's edge over whatever lies beyond it. The field there follows from
        # its smoothness alone.
        with torch.no_grad():
            seen = [warp.seen(u, z0, z1, window)[:, :, c0:c1] for z0, z1, c0, c1 in slabs]
        # the fixed image's side of each window does not change as the field does
        stats = [_window_stats(fix[:, :, z0:z1], window) for z0, z1, _, _ in slabs]
        n_vox = math.prod(fl.data.shape)
        t0 = time.time()
        for it in range(iters):
            u.grad = None
            sim = torch.zeros((), device=dev)  # read back only when logged
            for (z0, z1, c0, c1), inside, st in zip(slabs, seen, stats):
                warped = warp(mov, u, z0, z1)
                cc = _lncc(fix[:, :, z0:z1], st, warped, window)[:, :, c0:c1] * inside
                loss = -cc.sum() / n_vox
                loss.backward()
                sim -= loss.detach()
            reg = settings.smooth * _diffusion(u, spacing)
            reg.backward()
            opt.step()
            step += 1
            if it == 0:
                first = float(sim)
            record(step)
        log.info(
            "register%s: stage %d/%d, %s voxels, grid %s, %d iterations in %.1f s: "
            "similarity %.4f -> %.4f",
            label,
            stage,
            len(fixed),
            "x".join(map(str, fl.data.shape)),
            "x".join(map(str, shape)),
            iters,
            time.time() - t0,
            first,
            float(sim),
        )
        del fix, mov, warp, seen, stats
    if not label:  # a whole image: its levels no longer need the GPU's memory
        log.info("register: solved in %.1f s on %s", time.time() - started, dev)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def find_affine(
    fixed: Level, moving: Level, device: str = "auto"
) -> tuple[np.ndarray, dict]:
    """The fixed-to-moving affine (4x4, physical units) to start from when none is given,
    as the browser page's ``affine.ts`` finds it. The images' intensity moments are
    matched first, centre to centre and principal axis to principal axis, which leaves
    each axis's sign open: of the orientations that do not mirror the image, the best
    correlated is kept. A mirror image (one axis reversed) is not searched for: on a nearly
    symmetric specimen (a brain) it correlates as well as the right rotation, so the images
    cannot tell, and a fit cannot reach one from a rotation; give it as an affine. Then the 12 numbers are fitted by gradient ascent on the
    normalized cross-correlation, on a copy of at most ``MAX_FIT_VOXELS`` voxels after a
    coarser one. The moments assume both images show the same whole object. Also returns
    the correlation with no affine, after the moments and after the fit."""
    import torch

    dev = pick_device(device)
    started = time.time()
    fs = [_normalized(fixed)]
    ms = [_normalized(moving)]
    # copies: the finest within MAX_FIT_VOXELS, after one coarser if that is big enough
    while math.prod(fs[-1].data.shape) > MAX_FIT_VOXELS:
        fs.append(_halve(fs[-1]))
        ms.append(_halve(ms[-1]))
    coarser = _halve(fs[-1])
    stages = [len(fs) - 1]
    if min(coarser.data.shape) >= MIN_FIT_SIZE:
        fs.append(coarser)
        ms.append(_halve(ms[-1]))
        stages.insert(0, len(fs) - 1)

    f0, m0, ff, mf = fs[stages[0]], ms[stages[0]], fs[stages[-1]], ms[stages[-1]]
    cf, sf = _moments(f0)
    cm, sm = _moments(m0)
    ef, vf = np.linalg.eigh(sf)  # ascending, so both images' axes pair up by extent
    em, vm = np.linalg.eigh(sm)
    best = None
    for s in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).reshape(3, -1).T:
        # A = Vm sqrt(em) S / sqrt(ef) Vf^T: takes the fixed second moments to the moving ones
        a = vm @ np.diag(np.sqrt(np.maximum(em, 1e-12) / np.maximum(ef, 1e-12)) * s) @ vf.T
        if np.linalg.det(a) < 0:  # a mirror: not searched for (see above)
            continue
        v = _ncc(f0, m0, a, cm, cf, dev)
        if best is None or v > best[0]:
            best = (v, a)
    assert best is not None  # four of the eight orientations have each handedness
    scores = {
        "identity": _ncc(ff, mf, np.eye(3), np.zeros(3), np.zeros(3), dev),
        "moments": _ncc(ff, mf, best[1], cm, cf, dev),
    }

    def t32(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev)

    a, u = t32(best[1]).requires_grad_(True), t32(cm).requires_grad_(True)  # u: the centre's image
    for k, s in enumerate(stages):
        f, m = fs[s], ms[s]
        iters = FIT_ITERATIONS[0 if len(stages) == 1 else k]
        opt_a = _Adam(a, lr=LR_MATRIX, eps=1e-12)
        opt_u = _Adam(u, lr=LR_CENTRE * float(np.mean(f.voxel_size)), eps=1e-12)
        fix, mov, p = _fit_tensors(f, m, cf, dev)
        for _ in range(iters):
            a.grad = u.grad = None
            (-_ncc_t(fix, mov, m, p, a, u)).backward()  # ascent
            opt_a.step()
            opt_u.step()
    a_np, u_np = a.detach().cpu().numpy().astype(float), u.detach().cpu().numpy().astype(float)
    scores["final"] = _ncc(ff, mf, a_np, u_np, cf, dev)
    affine = np.eye(4)
    affine[:3, :3] = a_np
    affine[:3, 3] = u_np - a_np @ cf  # q = A (p - cf) + u
    log.info(
        "register: affine found in %.1f s: correlation %.3f with none, %.3f after the moments,"
        " %.3f after the fit",
        time.time() - started,
        scores["identity"],
        scores["moments"],
        scores["final"],
    )
    return affine, scores


def _normalized(level: Level) -> Level:
    return Level(normalize(level.data, *_range(level.data)), level.voxel_size, level.translation)


def _halve(level: Level) -> Level:
    """2x2x2 means (partial blocks at the far faces averaged over what they hold); the
    grid's origin moves to the first block's centre. In float32 whatever the data's type:
    sums of uint8 voxels would wrap."""
    x = level.data.astype(np.float32)
    for axis in range(3):
        n = x.shape[axis]
        pairs = (x.take(range(0, n - n % 2, 2), axis) + x.take(range(1, n, 2), axis)) / 2
        x = np.concatenate([pairs, x.take([n - 1], axis)], axis) if n % 2 else pairs
    grew = np.asarray(level.data.shape) > 1
    return Level(
        x.astype(np.float32),
        np.where(grew, 2 * level.voxel_size, level.voxel_size),
        np.where(grew, level.translation + level.voxel_size / 2, level.translation),
    )


def _moments(level: Level) -> tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted centre and covariance of the tissue (above the 20th percentile
    of the non-zero voxels), physical units."""
    data = level.data
    nonzero = data[data > 0]
    floor = np.percentile(nonzero, 20) if nonzero.size else 0.0
    idx = np.argwhere(data > floor)
    w = data[data > floor].astype(float)
    p = level.translation + idx * level.voxel_size
    c = (w[:, None] * p).sum(0) / w.sum()
    d = p - c
    return c, (w[:, None, None] * d[:, :, None] * d[:, None, :]).sum(0) / w.sum()


def _fit_tensors(f: Level, m: Level, cf, dev):
    """The two images as tensors and the fixed voxel centres relative to ``cf``, (N, 3)."""
    import torch

    fix = torch.as_tensor(f.data, device=dev).reshape(-1)
    mov = torch.as_tensor(m.data, device=dev)[None, None]
    idx = np.indices(f.data.shape, dtype=np.float32).reshape(3, -1).T
    p = torch.as_tensor(idx * f.voxel_size + (f.translation - cf), dtype=torch.float32, device=dev)
    return fix, mov, p


def _ncc_t(fix, mov, m: Level, p, a, u):
    """Normalized cross-correlation of the fixed image with the moving one sampled at
    ``a p + u`` (trilinear, 0 beyond it), differentiable in ``a`` and ``u``."""
    import torch
    import torch.nn.functional as F

    q = (
        p @ a.T + u - torch.as_tensor(m.translation, dtype=p.dtype, device=p.device)
    ) / torch.as_tensor(m.voxel_size, dtype=p.dtype, device=p.device)
    g = (
        2
        * q
        / torch.as_tensor(
            np.maximum(np.asarray(m.data.shape) - 1, 1), dtype=p.dtype, device=p.device
        )
        - 1
    )
    v = F.grid_sample(
        mov,
        g.flip(-1).reshape(1, 1, 1, -1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(-1)
    fc, vc = fix - fix.mean(), v - v.mean()
    return (fc * vc).sum() / ((fc * fc).sum() * (vc * vc).sum()).clamp_min(1e-24).sqrt()


def _ncc(f: Level, m: Level, a, u, cf, dev) -> float:
    import torch

    fix, mov, p = _fit_tensors(f, m, cf, dev)
    with torch.no_grad():
        return float(
            _ncc_t(
                fix,
                mov,
                m,
                p,
                torch.as_tensor(a, dtype=p.dtype, device=dev),
                torch.as_tensor(u, dtype=p.dtype, device=dev),
            )
        )


def _range(data: np.ndarray) -> tuple[float, float]:
    step = max(1, data.size // 1_000_000)
    lo, hi = np.percentile(data.reshape(-1)[::step], [0.5, 99.5])
    return float(lo), float(hi)


def _control_shape(lo, hi, voxel_size, grid: float) -> tuple[int, ...]:
    """Control points spanning ``lo..hi`` at most ``grid`` voxels apart (exactly that far
    when the extent is a whole number of spacings, rounding error allowed for)."""
    n = np.ceil((hi - lo) / (grid * np.asarray(voxel_size, dtype=float)) - 1e-6).astype(int) + 1
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

    def _index(self, x):
        """Moving-image voxel indices of fixed-space physical points ``x`` under the affine."""
        return (x @ self.a[:, :3].T + self.a[:, 3] - self.tm) / self.vm

    def _displaced(self, u, z0: int, z1: int):
        """Moving-image voxel indices of the fixed voxel centres in planes ``z0:z1`` under
        the field ``u`` and the affine, (d, h, w, 3)."""
        import torch.nn.functional as F

        p = self._points(z0, z1)
        g = 2 * (p - self.lo) / self.span - 1  # into the control grid, grid_sample wants x,y,z
        disp = F.grid_sample(
            u, g.flip(-1)[None], mode="bilinear", padding_mode="border", align_corners=True
        )[0].permute(1, 2, 3, 0)
        return self._index(p + disp)

    def seen(self, u, z0: int, z1: int, window: int):
        """Which fixed voxels in planes ``z0:z1`` have their whole ``window`` (as far as the
        fixed image reaches) inside the moving image under the starting field ``u`` and the
        affine: a (1, 1, d, h, w) float mask."""
        import torch.nn.functional as F

        idx = self._displaced(u, z0, z1)
        ok = ((idx >= 0) & (idx <= self.mshape)).all(dim=-1)[None, None].float()
        h = window // 2
        ok = F.pad(ok, (h,) * 6, value=1.0)  # beyond the fixed image: no data to miss
        for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # a box minimum
            ok = -F.max_pool3d(-ok, k, stride=1)
        return ok

    def __call__(self, mov, u, z0: int, z1: int):
        import torch.nn.functional as F

        d, h, w = z1 - z0, self.shape[1], self.shape[2]
        g = 2 * self._displaced(u, z0, z1) / self.mshape - 1
        return F.grid_sample(  # 0 beyond the moving image, as the served image has
            mov, g.flip(-1)[None], mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(1, 1, d, h, w)


def _box_mean(x, window: int):
    """Mean over each voxel's window (clipped at the volume's faces), per channel: a
    separable box filter. A prefix-sum version, whose cost would not grow with the window,
    measured 4x slower at the default window (68 vs 17 ms on a 2^25-voxel level, RTX 2080
    Ti; cuDNN's pooling is that fast) and only wins past windows of about 60, so the
    browser engine's running sum is not mirrored here."""
    import torch.nn.functional as F

    h = window // 2
    for k in ((window, 1, 1), (1, window, 1), (1, 1, window)):  # one axis at a time
        pad = tuple(h if n > 1 else 0 for n in k)
        x = F.avg_pool3d(x, k, stride=1, padding=pad, count_include_pad=False)
    return x


def _window_stats(i, window: int):
    """The fixed image's window means and variances."""
    import torch

    mi, ii = _box_mean(torch.cat([i, i * i], dim=1), window).unbind(1)
    return mi, ii - mi * mi


def _lncc(i, stats, j, window: int):
    """Squared local normalized cross-correlation per voxel, in [0, 1]; ``stats`` are the
    fixed image ``i``'s window means and variances (``_window_stats``)."""
    import torch

    mi, vi = stats
    mj, jj, ij = _box_mean(torch.cat([j, j * j, i * j], dim=1), window).unbind(1)
    cov = ij - mi * mj
    vj = jj - mj * mj
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
