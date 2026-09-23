"""Showcase ops beyond pointwise: multi-scale filtering, morphology and labelling (need scipy).

These demonstrate what a viewer shader cannot do: neighbourhood context (halos), binary
morphology, and connected-component labelling with size filtering.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import Field

from chunkmirage.ops.base import Op, register


@register
class DoG(Op):
    """Difference of Gaussians: enhances blob-like structures of a chosen size, suppresses background."""

    name = "dog"
    sigma: float = Field(
        2.0, gt=0, description="Size of structures to enhance, in voxels (smaller blur)."
    )
    ratio: float = Field(
        1.6,
        gt=1,
        le=4,
        description="Larger blur = sigma × ratio. 1.6 approximates a Laplacian of Gaussian.",
    )
    gain: float = Field(
        4.0, gt=0, description="Multiply the difference so the result uses the 0..255 range."
    )

    @property
    def halo(self):  # type: ignore[override]
        return math.ceil(self.sigma * self.ratio * 3)

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        b = block.astype(np.float32)
        d = gaussian_filter(b, self.sigma, mode="nearest") - gaussian_filter(
            b, self.sigma * self.ratio, mode="nearest"
        )
        return np.clip(d * self.gain + 128, 0, 255).astype(np.uint8)


@register
class Morphology(Op):
    """Binary morphology on a mask: remove specks (open), fill holes (close), shrink or grow."""

    name = "morphology"
    operation: str = Field(
        "open",
        pattern="^(open|close|erode|dilate)$",
        description="open = erode then dilate (removes small objects); close = dilate then erode (fills small gaps); erode; dilate.",
    )
    radius: int = Field(
        2, ge=1, le=16, description="Radius of the spherical structuring element, in voxels."
    )

    @property
    def halo(self):  # type: ignore[override]
        return 2 * self.radius + 1

    def output_dtype(self, in_dtype):
        return np.dtype("uint8")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

        r = self.radius
        zz, yy, xx = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = (zz * zz + yy * yy + xx * xx) <= r * r
        mask = block > 0
        fn = {
            "open": binary_opening,
            "close": binary_closing,
            "erode": binary_erosion,
            "dilate": binary_dilation,
        }[self.operation]
        return fn(mask, structure=ball).astype(np.uint8)


@register
class Label(Op):
    """Connected components of a mask, coloured as segments. Labels are unique per chunk, so one
    object spanning several chunks gets several colours; that is the honest per-chunk preview."""

    name = "label"
    min_size: int = Field(
        0,
        ge=0,
        description="Drop components smaller than this many voxels (counted within the chunk + halo).",
    )
    connectivity: int = Field(
        1,
        ge=1,
        le=3,
        description="1 = faces only (6-connected), 2 = +edges (18), 3 = +corners (26).",
    )

    @property
    def halo(self):  # type: ignore[override]
        # Some context so size filtering near chunk borders sees more of each object.
        return 4 if self.min_size else 0

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def apply(self, block: np.ndarray) -> np.ndarray:
        from scipy.ndimage import generate_binary_structure, label

        structure = generate_binary_structure(3, self.connectivity)
        labels, n = label(block > 0, structure=structure)
        labels = labels.astype(np.uint32)
        if self.min_size and n:
            sizes = np.bincount(labels.ravel(), minlength=n + 1)
            keep = sizes >= self.min_size
            keep[0] = False
            labels = np.where(keep[labels], labels, 0).astype(np.uint32)
        return labels

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        labels = self.apply(block)
        # Offset labels by a salt derived from the chunk position so colours differ between
        # chunks (labels stay < 2**16 per chunk; the salt occupies the high 16 bits).
        z, y, x = (int(v) for v in box.start[-3:])
        salt = np.uint32(((z * 73856093) ^ (y * 19349663) ^ (x * 83492791)) & 0x7FFF) << np.uint32(
            16
        )
        return np.where(labels > 0, labels + salt, 0).astype(np.uint32)
