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
    distance: float | None = Field(
        None,
        gt=0,
        description="Reach in the data's units (nm, say), instead of radius: each level counts "
        "it in its own voxels, so a contact means the same at every zoom (a level whose voxels "
        "are bigger than it keeps the voxels in both structures).",
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

    def for_level(self, info: ArrayInfo) -> Op:
        """With ``distance``, the radius in this level's voxels (its finest spatial axis)."""
        if self.distance is None:
            return self
        op = self.model_copy(update={"radius": self.distance / min(info.voxel_size[-3:])})
        op._cache = self._cache
        return op

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
            kind="mask",
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


@register
class NormalizedDifference(Op):
    """``(a - b) / (a + b)`` of two channels of a ``stack://`` source (float32): the indices
    remote sensing reads plants, water and burn scars from (NDVI is near infrared and red,
    NBR near and shortwave infrared). With ``minus``, a second pair's index is subtracted
    from the first's: a change between two dates. Burn severity (dNBR) is the NBR before a
    fire minus the NBR after, the before and after images' bands stacked as four channels.
    Where a band is zero or less, or the two sum to under ``floor`` (water), the index is NaN."""

    name = "normalized_difference"
    pair: list[int] = Field(
        [0, 1], min_length=2, max_length=2, description="The channels a and b, counted from the first."
    )
    minus: list[int] | None = Field(
        None,
        min_length=2,
        max_length=2,
        description="Two more channels, whose index is subtracted from the first pair's: [2, 3] "
        "for the after image of a stack of before and after.",
    )
    offset: float = Field(
        0.0,
        description="Added to every channel first, to make reflectances of stored numbers: "
        "Sentinel-2 since 2022 stores them plus 1000 (offset -1000).",
    )
    nodata: float | None = Field(
        None, description="A stored value meaning no data (Sentinel-2: 0): the index there is NaN."
    )
    floor: float = Field(
        0.0,
        ge=0,
        description="Where a pair's two values sum to less than this (after offset), its index is "
        "noise, a ratio of near zeros (water reflects almost no infrared), and is NaN.",
    )
    output_kind = "image"

    def _channels(self) -> list[int]:
        return [*self.pair, *(self.minus or [])]

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        need = max(self._channels()) + 1
        if info.ndim < 3 or info.axes[0] != "c" or info.shape[0] < need:
            raise ValueError(
                f"normalized_difference needs a source whose first axis holds {need} or more "
                f"channels, such as stack://<a>|<b>; got axes {info.axes}, shape {info.shape}"
            )
        return ArrayInfo(
            shape=info.shape[1:], dtype=np.dtype("float32"), chunk_shape=info.chunk_shape[1:],
            voxel_size=info.voxel_size[1:], units=info.units[1:], axes=info.axes[1:],
            translation=info.translation[1:], kind="image",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        def index(a: int, b: int) -> np.ndarray:
            x, y = (block[c].astype(np.float32) + np.float32(self.offset) for c in (a, b))
            with np.errstate(divide="ignore", invalid="ignore"):
                out = (x - y) / (x + y)
            out[(x <= 0) | (y <= 0) | (x + y < self.floor)] = np.nan  # positive, and not near zero
            if self.nodata is not None:
                out[(block[a] == self.nodata) | (block[b] == self.nodata)] = np.nan
            return out

        out = index(*self.pair)
        return out - index(*self.minus) if self.minus else out
