from __future__ import annotations

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class Threshold(Op):
    """Binary mask: voxels with ``low <= value < high`` become ``value``, everything else 0."""

    name = "threshold"
    low: float = Field(
        0.0,
        description="Lower bound (inclusive), in the source's intensity units. Voxels at or above it pass.",
    )
    high: float | None = Field(
        None,
        description="Upper bound (exclusive). Leave empty for no upper bound. Use it to select an intensity band.",
    )
    value: int = Field(
        1,
        ge=1,
        le=255,
        description="Label written for voxels that pass (1 shows as one segment in Neuroglancer).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        mask = block >= self.low
        if self.high is not None:
            mask &= block < self.high
        return mask.astype(np.uint8) * np.uint8(self.value)


@register
class Cast(Op):
    """Convert to another data type, e.g. float32 → uint8 for viewers that need integers."""

    name = "cast"
    dtype: str = Field(
        "uint8",
        description="Target numpy dtype name: uint8, uint16, uint32, uint64, int16, float32, ...",
    )
    clip: bool = Field(
        True,
        description="Clip values to the target integer range first (avoids wrap-around, e.g. 300 → 255 not 44).",
    )

    def output_dtype(self, in_dtype):
        return np.dtype(self.dtype)

    def apply(self, block: np.ndarray) -> np.ndarray:
        out = np.dtype(self.dtype)
        if self.clip and np.issubdtype(out, np.integer):
            ii = np.iinfo(out)
            block = np.clip(block, ii.min, ii.max)
        return block.astype(out)


@register
class Scale(Op):
    """Linear intensity rescale ``value * factor + offset`` (output is float32)."""

    name = "scale"
    factor: float = Field(1.0, description="Multiply every voxel by this (contrast).")
    offset: float = Field(0.0, description="Then add this (brightness).")

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        return block.astype(np.float32) * np.float32(self.factor) + np.float32(self.offset)
