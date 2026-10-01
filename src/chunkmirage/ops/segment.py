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
    packages = ("scipy",)
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
    packages = ("scipy",)
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
    packages = ("scipy",)
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


@register
class Spots(Op):
    """Bright diffraction-limited spots, such as single mRNA molecules in smFISH or EASI-FISH:
    a difference of Gaussians, its local maxima above ``threshold``, each drawn as a small
    ball. A spot's id comes from its position in the volume, so one spot keeps its id and
    colour whichever chunk finds it."""

    name = "spots"
    packages = ("scipy",)
    sigma: float = Field(
        1.0,
        gt=0,
        le=8,
        description="Spot size, in voxels along y and x: the smaller blur of the difference of "
        "Gaussians (the larger is 1.6 times it). About the spot's standard deviation.",
    )
    sigma_z: float = Field(
        0.6,
        gt=0,
        le=8,
        description="The same along z, in z voxels: smaller than sigma when z voxels are "
        "coarser, as in most light microscopy.",
    )
    threshold: float = Field(
        10.0,
        ge=0,
        description="Least difference-of-Gaussians response a spot needs, in the image's "
        "intensity units: higher finds fewer, brighter spots.",
    )
    separation: int = Field(
        2,
        ge=1,
        le=8,
        description="Spots closer than this, in y-x voxels, are one: the radius of the local "
        "maximum search.",
    )
    radius: int = Field(
        1,
        ge=0,
        le=8,
        description="Radius of the ball drawn for each spot, in y-x voxels (0 marks one voxel).",
    )

    def _z(self, v: float) -> int:
        """``v`` y-x voxels in z voxels, as far as sigma_z / sigma says."""
        return int(math.ceil(v * self.sigma_z / self.sigma))

    @property
    def halo(self):  # type: ignore[override]
        xy = math.ceil(3 * 1.6 * self.sigma) + self.separation + self.radius
        z = math.ceil(3 * 1.6 * self.sigma_z) + self._z(self.separation) + self._z(self.radius)
        return (z, xy, xy)

    def output_dtype(self, in_dtype):
        return np.dtype("uint32")

    def output_info(self, info):
        if info.ndim != 3:
            raise ValueError(
                f"spots finds spots in a z, y, x volume, not axes {info.axes}: pin the other "
                'axes with the spec\'s select, e.g. {"c": 1, "t": 0} (--select c=1,t=0)'
            )
        return super().output_info(info)

    def apply(self, block: np.ndarray) -> np.ndarray:  # pragma: no cover - apply_at is used
        return self.apply_at(block, None)

    def apply_at(self, block: np.ndarray, box) -> np.ndarray:
        from scipy.ndimage import gaussian_filter, grey_dilation, maximum_filter

        b = block.astype(np.float32)
        s = (self.sigma_z, self.sigma, self.sigma)
        dog = gaussian_filter(b, s, mode="nearest") - gaussian_filter(
            b, tuple(1.6 * v for v in s), mode="nearest"
        )
        sz, sxy = self._z(self.separation), self.separation
        peaks = (dog == maximum_filter(dog, size=(2 * sz + 1, 2 * sxy + 1, 2 * sxy + 1))) & (
            dog >= self.threshold
        )
        out = np.zeros(b.shape, dtype=np.uint32)
        where = np.nonzero(peaks)
        if not len(where[0]):
            return out
        origin = np.asarray(box.start[-3:] if box is not None else (0, 0, 0), dtype=np.int64)
        z, y, x = (w.astype(np.int64) + o for w, o in zip(where, origin))
        h = (z * 73856093) ^ (y * 19349663) ^ (x * 83492791)
        out[where] = (h % 0xFFFFFFFE + 1).astype(np.uint32)
        if self.radius:
            rz, r = self._z(self.radius), self.radius
            zz, yy, xx = np.ogrid[-rz : rz + 1, -r : r + 1, -r : r + 1]
            ball = (zz / max(rz, 1)) ** 2 * (rz > 0) + (yy * yy + xx * xx) / (r * r) <= 1
            out = grey_dilation(out, footprint=ball)
        return out
