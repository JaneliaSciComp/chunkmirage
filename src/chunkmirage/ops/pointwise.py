from __future__ import annotations

import numpy as np

from chunkmirage.ops.base import Op, register


@register
class Threshold(Op):
    """Binary threshold: ``low <= x < high`` -> ``value``, else 0 (uint8)."""

    name = "threshold"
    low: float = 0.0
    high: float | None = None
    value: int = 1

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        mask = block >= self.low
        if self.high is not None:
            mask &= block < self.high
        return mask.astype(np.uint8) * np.uint8(self.value)


@register
class Cast(Op):
    """Cast to another dtype, optionally clipping to its range first."""

    name = "cast"
    dtype: str = "uint8"
    clip: bool = True

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
    """Affine intensity map ``x * factor + offset`` in float32."""

    name = "scale"
    factor: float = 1.0
    offset: float = 0.0

    def output_dtype(self, in_dtype):
        return np.dtype("float32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        return block.astype(np.float32) * np.float32(self.factor) + np.float32(self.offset)
