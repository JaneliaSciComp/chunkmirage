"""Neighbourhood filters (need scipy). These exercise the halo machinery."""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

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
