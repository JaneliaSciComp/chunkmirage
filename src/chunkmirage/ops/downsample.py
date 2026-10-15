"""Coarser voxels, as a stored pyramid level is made: the one op the engine itself runs, to
make the coarser levels of an op bound to one resolution (``chunkmirage.pipeline``). It
imports nothing beyond numpy, so the browser engine runs it too."""

from __future__ import annotations

import numpy as np
from pydantic import Field, PrivateAttr

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


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
