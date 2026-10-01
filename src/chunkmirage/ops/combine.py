"""Ops over the channels of a ``stack://`` source: results that need two images at once.

Such an op takes a block whose first axis is the stack's channel axis and returns one
without it; the pipeline reads every channel and pads only the spatial axes by the halo.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.core import ArrayInfo
from chunkmirage.ops.base import Op, register


@register
class Contacts(Op):
    """Contact sites between two structures, the first two channels of a ``stack://`` source:
    the voxels within ``radius`` of both. A mask; follow with ``label`` to colour and
    size-filter the sites."""

    name = "contacts"
    packages = ("scipy",)
    radius: float = Field(
        3.0,
        gt=0,
        le=32,
        description="Reach, in voxels: a voxel is a contact site when both structures lie within "
        "this distance of it (Euclidean). The halo is radius + 1.",
    )
    a_low: float = Field(
        128.0,
        description="Values at or above this in the first channel are the first structure: 128 "
        "for a uint8 probability map, 1 for a segmentation.",
    )
    b_low: float = Field(
        128.0,
        description="The same threshold for the second channel, the second structure.",
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.radius) + 1

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        if info.ndim < 4 or info.axes[0] != "c" or info.shape[0] < 2:
            raise ValueError(
                "contacts needs a source whose first axis is a channel axis holding two "
                f"structures, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:],
            dtype=self.output_dtype(info.dtype),
            chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:],
            units=info.units[1:],
            axes=info.axes[1:],
            translation=info.translation[1:],
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        a = block[0] >= self.a_low
        b = block[1] >= self.b_low
        return (self._within_reach(a) & self._within_reach(b)).astype(np.uint8)

    def _within_reach(self, mask: np.ndarray) -> np.ndarray:
        """The voxels within ``radius`` of ``mask``: the distance to it, thresholded."""
        from scipy.ndimage import distance_transform_edt

        if not mask.any():
            return np.zeros(mask.shape, dtype=bool)
        if mask.all():
            return np.ones(mask.shape, dtype=bool)
        return distance_transform_edt(~mask) <= self.radius
