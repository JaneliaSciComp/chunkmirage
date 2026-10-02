"""Neighbourhood filters (most need scipy). These exercise the halo machinery."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Gaussian(Op):
    """Gaussian blur (smoothing). Reduces noise before thresholding; larger sigma = blurrier."""

    name = "gaussian"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        description="Blur width in voxels (standard deviation). 1 removes pixel noise; 3-5 merges small structures.",
    )
    truncate: float = Field(
        3.0,
        ge=1,
        le=6,
        description="Kernel radius in units of sigma. Rarely needs changing; 3 keeps 99.7% of the kernel. "
        "Determines the halo: ceil(sigma × truncate) voxels of neighbouring data are read on each side.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.truncate)

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            block.astype(np.float32), self.sigma, truncate=self.truncate, mode="nearest"
        )


@register
class Uniform(Op):
    """Box (mean) filter: each voxel becomes the average of a size³ cube around it."""

    name = "uniform"
    packages = ("scipy",)
    size: int = Field(
        3,
        ge=1,
        le=31,
        description="Edge length of the averaging cube, in voxels (odd values are centred).",
    )

    @property
    def halo(self):  # type: ignore[override]
        return self.size // 2 + 1

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        return uniform_filter(block.astype(np.float32), self.size, mode="nearest")


@register
class Diff(Op):
    """Change along one axis: each voxel minus the one ``lag`` steps before it on ``axis``
    (float32). Along time, what changed since the frame or day before: a hurricane's cold
    wake in sea temperature, a flare brightening the sun. The first ``lag`` steps of the
    array compare against its first."""

    name = "diff"
    axis: int = Field(
        0,
        ge=0,
        description="The axis to difference along, counted from the first (0: time in a t, y, x series).",
    )
    lag: int = Field(1, ge=1, le=64, description="How many steps back to compare with.")

    @property
    def halo(self):  # type: ignore[override]
        return self.lag

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        if self.axis >= ndim:
            raise ValueError(f"diff axis={self.axis}: the data has {ndim} axes")
        return tuple(self.lag if a == self.axis else 0 for a in range(ndim))

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = self.axis
        b = block.astype(np.float32)
        out = np.zeros_like(b)
        now, before = [slice(None)] * b.ndim, [slice(None)] * b.ndim
        now[a], before[a] = slice(self.lag, None), slice(None, -self.lag)
        out[tuple(now)] = b[tuple(now)] - b[tuple(before)]
        return out


@register
class Gradient(Op):
    """Rate of change along each of ``axes``, one channel each on a new leading ``c`` axis
    (float32), per unit of the axes (a level's voxel size): central differences. It returns
    only the interior it can compute, one voxel less on each side of those axes, as a valid
    convolution (or a model) does. Where temperature changes fastest at sea, ocean fronts;
    on terrain, the slope's components; in a volume, edges and their direction."""

    name = "gradient"
    axes: list[int] | None = Field(
        None,
        description="The axes to differentiate along, counted from the first (1, 2: latitude and "
        "longitude of a time, lat, lon series); default the last three, or all if fewer.",
    )
    _voxel: tuple[float, ...] | None = PrivateAttr(None)

    def _along(self, ndim: int) -> list[int]:
        axes = list(range(max(0, ndim - 3), ndim)) if self.axes is None else list(self.axes)
        if not axes or any(not 0 <= a < ndim for a in axes) or len(set(axes)) != len(axes):
            raise ValueError(f"gradient axes={self.axes}: the data has {ndim} axes")
        return axes

    @property
    def halo(self):  # type: ignore[override]
        return 1

    def halo_for(self, ndim: int) -> tuple[int, ...]:
        along = self._along(ndim)
        return tuple(1 if a in along else 0 for a in range(ndim))

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        n = len(self._along(info.ndim))
        return info.with_(
            shape=(n, *info.shape), chunk_shape=(n, *info.chunk_shape), dtype=np.dtype("float32"),
            voxel_size=(1.0, *info.voxel_size), units=("", *info.units), axes=("c", *info.axes),
            translation=(0.0, *info.translation), kind="image",
        )

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._voxel = tuple(float(v) for v in info.voxel_size)
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32)
        along = self._along(b.ndim)
        voxel = self._voxel[-b.ndim:] if self._voxel else (1.0,) * b.ndim
        inner = [slice(1, -1) if a in along else slice(None) for a in range(b.ndim)]
        out = []
        for a in along:
            hi, lo = list(inner), list(inner)
            hi[a], lo[a] = slice(2, None), slice(None, -2)
            out.append((b[tuple(hi)] - b[tuple(lo)]) / np.float32(2 * voxel[a]))
        return np.stack(out)


@register
class Downsample(Op):
    """Coarser voxels: each block of ``factor`` voxels becomes one, their mean, or for labels
    and masks their most common value. The grid changes with it: voxels ``factor`` times
    bigger, the shape divided (rounded up), each voxel's position the centre of its block.
    A pipeline makes the coarser levels of an op with an input voxel size this way."""

    name = "downsample"
    factor: list[int] = Field(
        [2, 2, 2],
        description="Voxels per output voxel along each of the data's last axes (z, y, x): "
        "2, 2, 2 halves each; 1 keeps an axis as it is.",
    )
    mode: str = Field(
        "auto",
        pattern="^(auto|mean|mode)$",
        description="mean of each block; mode, its most common value (labels, masks); auto: "
        "mode for labels and masks, mean for anything else.",
    )
    _mode: str = PrivateAttr("mean")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        f = self.factor
        if not f or len(f) > info.ndim or any(int(v) < 1 for v in f):
            raise ValueError(f"downsample factor={f}: one whole number of 1 or more per axis, of {info.ndim}")
        return info.rescaled([v * k for v, k in zip(info.voxel_size[-len(f) :], f)])

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._mode = self.mode if self.mode != "auto" else ("mode" if info.kind in ("label", "mask") else "mean")
        op._cache = self._cache
        return op

    def apply(self, block: np.ndarray) -> np.ndarray:
        f, k = [int(v) for v in self.factor], len(self.factor)
        lead, space = block.shape[:-k], block.shape[-k:]
        whole = [-(-n // v) * v for n, v in zip(space, f)]
        if list(space) != whole:  # the array's last block along an axis: its last voxel repeated
            block = np.pad(block, [(0, 0)] * len(lead) + [(0, w - n) for w, n in zip(whole, space)], "edge")
        shape = list(lead) + [x for w, v in zip(whole, f) for x in (w // v, v)]
        blocks = block.reshape(shape)
        inner = tuple(len(lead) + 2 * i + 1 for i in range(k))  # the axes within each block
        if self._mode == "mean":  # in the data's own type, integers rounded
            mean = blocks.mean(axis=inner, dtype=np.float64)
            return (np.rint(mean) if np.issubdtype(block.dtype, np.integer) else mean).astype(block.dtype)
        order = [a for a in range(blocks.ndim) if a not in inner] + list(inner)
        g = blocks.transpose(order).reshape(*lead, *(w // v for w, v in zip(whole, f)), -1)
        counts = (g[..., :, None] == g[..., None, :]).sum(-1)  # how often each value of a block occurs in it
        return np.take_along_axis(g, counts.argmax(-1)[..., None], -1)[..., 0]
