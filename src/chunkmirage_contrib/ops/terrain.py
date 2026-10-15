"""Terrain from elevation: slope and hillshade, the first things one computes from a digital
elevation model. Both work on the last two axes (``y, x``, rows down the image) in the
level's own pixel spacing, so a coarse level's slope is the slope of its coarser grid."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


class _Terrain(Op):
    halo = 1
    _spacing: tuple[float, float] = PrivateAttr((1.0, 1.0))

    def for_level(self, info: ArrayInfo) -> Op:
        op = self.model_copy()
        op._spacing = (float(info.voxel_size[-2]), float(info.voxel_size[-1]))
        op._cache = self._cache
        return op

    def _gradient(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d(elevation)/dy and /dx, per unit of the axes (y counting down the image)."""
        z = block.astype(np.float64) * self.z_factor
        dy, dx = np.gradient(z, *self._spacing, axis=(-2, -1))
        return dy, dx


@register
class Slope(_Terrain):
    """Slope of an elevation model in degrees (0 flat, 90 a cliff), from the level's pixel
    spacing; output float32, NaN where the elevation is."""

    name = "slope"
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing (1 when both are metres).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


@register
class Hillshade(_Terrain):
    """Shaded relief: the brightness of the terrain lit by a distant sun at ``azimuth`` and
    ``altitude`` (local illumination only: no shadows cast across the terrain). Output
    uint8, 1 (unlit) to 255 (facing the sun), 0 where the elevation is NaN."""

    name = "hillshade"
    azimuth: float = Field(
        315.0,
        ge=0,
        le=360,
        description="Direction the sun shines from, degrees clockwise from the top of the image.",
    )
    altitude: float = Field(
        45.0, gt=0, le=90, description="Height of the sun above the horizon, in degrees."
    )
    z_factor: float = Field(
        1.0,
        gt=0,
        description="Elevation units per unit of the pixel spacing; above 1 exaggerates relief.",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        dy, dx = self._gradient(block)
        az, alt = np.radians(self.azimuth), np.radians(self.altitude)
        # the surface normal (-dz/dx, dz/dy up the image, 1) against the sun's direction
        sun = (np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt))
        lit = (-dx * sun[0] + dy * sun[1] + sun[2]) / np.sqrt(dx * dx + dy * dy + 1.0)
        out = (1 + 254 * np.clip(lit, 0.0, 1.0)).round()
        return np.where(np.isfinite(lit), out, 0).astype(np.uint8)
