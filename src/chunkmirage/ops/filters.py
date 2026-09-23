"""Neighbourhood filters (need scipy). These exercise the halo machinery."""

from __future__ import annotations

import math

import numpy as np

from chunkmirage.ops.base import Op, register


@register
class Gaussian(Op):
    """Gaussian blur with ``sigma`` voxels (isotropic). Output float32."""

    name = "gaussian"
    sigma: float = 1.0
    truncate: float = 3.0

    @property
    def halo(self):  # type: ignore[override]
        return int(math.ceil(self.sigma * self.truncate))

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(
            block.astype(np.float32), self.sigma, truncate=self.truncate, mode="nearest"
        )


@register
class Uniform(Op):
    """Mean filter over a ``size``-voxel cube. Output float32."""

    name = "uniform"
    size: int = 3

    @property
    def halo(self):  # type: ignore[override]
        return self.size // 2 + 1

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        return uniform_filter(block.astype(np.float32), self.size, mode="nearest")
