"""Core value types shared by sources, ops, frontends and the server.

Conventions
-----------
* Arrays are handled in numpy C order. For a 3-D volume that means axes ``(z, y, x)``
  with ``x`` varying fastest in memory. Frontends that use the opposite convention
  (N5, precomputed) reverse axis lists when emitting metadata / parsing chunk keys.
* ``Box`` is half-open ``[start, stop)`` in voxel coordinates of a given scale level.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Half-open voxel box ``[start, stop)`` in C-order axes."""

    start: tuple[int, ...]
    stop: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.start) != len(self.stop):
            raise ValueError("start and stop must have same length")

    @property
    def ndim(self) -> int:
        return len(self.start)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(b - a for a, b in zip(self.start, self.stop))

    @property
    def empty(self) -> bool:
        return any(s <= 0 for s in self.shape)

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop))

    def pad(self, halo: Sequence[int]) -> Box:
        return Box(
            tuple(a - h for a, h in zip(self.start, halo)),
            tuple(b + h for b, h in zip(self.stop, halo)),
        )

    def clip(self, shape: Sequence[int]) -> Box:
        return Box(
            tuple(min(max(a, 0), s) for a, s in zip(self.start, shape)),
            tuple(min(max(b, 0), s) for b, s in zip(self.stop, shape)),
        )

    def relative_to(self, other: Box) -> Box:
        """Express this box in coordinates whose origin is ``other.start``."""
        return Box(
            tuple(a - o for a, o in zip(self.start, other.start)),
            tuple(b - o for b, o in zip(self.stop, other.start)),
        )

    @staticmethod
    def from_chunk(index: Sequence[int], chunk_shape: Sequence[int]) -> Box:
        start = tuple(i * c for i, c in zip(index, chunk_shape))
        stop = tuple(s + c for s, c in zip(start, chunk_shape))
        return Box(start, stop)


KINDS = (None, "image", "label", "mask")


def kind_for_dtype(dtype) -> str | None:
    """What values of ``dtype`` most likely are when nothing says (a stored array): ``mask``
    for booleans, ``label`` for integers of 32 bits or more (segment ids; intensities are
    rarely stored that wide), else unknown. Labels and masks are resampled by nearest voxel
    and downsampled by their most common value, never averaged."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return "mask"
    if dtype.kind in "iu" and dtype.itemsize >= 4:
        return "label"
    return None


def spatial_axes(axes: Sequence[str]) -> tuple[int, ...]:
    """The axes named z, y or x; without any, the last three (worldarray's rule)."""
    named = tuple(i for i, a in enumerate(axes) if a in ("z", "y", "x"))
    return named or tuple(range(max(0, len(axes) - 3), len(axes)))


def corner_to_centre(corner, voxel_size, axes) -> tuple[float, ...]:
    """Voxel 0's centre from its corner, both C order and one entry per axis: half a voxel on
    along the spatial axes (``spatial_axes``), not along channels or time."""
    spatial = set(spatial_axes(axes))
    return tuple(
        float(c) + (float(v) / 2 if a in spatial else 0.0)
        for a, (c, v) in enumerate(zip(corner, voxel_size))
    )


@dataclass(frozen=True)
class ArrayInfo:
    """Static description of one scale level of a chunked array (C-order axes). ``kind`` is
    what its values are, for viewers choosing how to show it: ``image`` (intensities),
    ``label`` (segment ids) or ``mask`` (inside or not); ``None`` if unknown.
    ``computed_from`` is the input an op's values were computed on when that is finer than
    their own grid, as an op writing one vector per lattice node sets it in ``output_info``:
    its values were fitted on the input's voxels, not its nodes'. ``warp://`` then takes a
    field's level by the voxels it was fitted on, and the op's ``for_level`` sees the
    input's exact extent (the output's shape is rounded up). ``None``: its own grid.
    ``translation`` is voxel 0's centre; an array nothing places has its corner at the
    origin (``voxel_size / 2`` along the spatial axes), as Neuroglancer draws one."""

    shape: tuple[int, ...]
    dtype: np.dtype
    chunk_shape: tuple[int, ...]
    voxel_size: tuple[float, ...]
    units: tuple[str, ...]
    axes: tuple[str, ...]
    translation: tuple[float, ...] = field(default=None)  # type: ignore[assignment]
    kind: str | None = None
    computed_from: ArrayInfo | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "chunk_shape", tuple(int(s) for s in self.chunk_shape))
        n = len(self.shape)
        for name in ("chunk_shape", "voxel_size", "units", "axes"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must have length {n}, got {getattr(self, name)}")
        if self.translation is None:
            centre = corner_to_centre((0.0,) * n, self.voxel_size, self.axes)
            object.__setattr__(self, "translation", centre)
        if len(self.translation) != n:
            raise ValueError(f"translation must have length {n}, got {self.translation}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def chunk_grid(self) -> tuple[int, ...]:
        return tuple(-(-s // c) for s, c in zip(self.shape, self.chunk_shape))

    def chunk_box(self, index: Sequence[int], clip: bool = True) -> Box:
        box = Box.from_chunk(index, self.chunk_shape)
        return box.clip(self.shape) if clip else box

    def chunks_covering(self, box: Box) -> Iterator[tuple[int, ...]]:
        """Yield chunk indices whose extent intersects ``box`` (assumed already clipped)."""
        if box.empty:
            return
        lo = [a // c for a, c in zip(box.start, self.chunk_shape)]
        hi = [(b - 1) // c for b, c in zip(box.stop, self.chunk_shape)]
        for idx in np.ndindex(*[h - lo_i + 1 for lo_i, h in zip(lo, hi)]):
            yield tuple(int(lo_i + i) for lo_i, i in zip(lo, idx))

    def with_(self, **changes) -> ArrayInfo:
        from dataclasses import replace

        return replace(self, **changes)

    def rescaled(self, voxel_size: Sequence[float]) -> ArrayInfo:
        """This array's extent on voxels of ``voxel_size`` along its last axes (as many as
        given): the shape rounded up, chunks as they were (in voxels), and the translation
        moved so each voxel's position is its centre, as OME-Zarr's is (a voxel twice as big
        covers two, its centre half an old voxel on)."""
        k = len(voxel_size)
        old, new = self.voxel_size[-k:], tuple(float(v) for v in voxel_size)
        shape = tuple(math.ceil(n * o / v - 1e-9) for n, o, v in zip(self.shape[-k:], old, new))
        shift = tuple(t + (v - o) / 2 for t, o, v in zip(self.translation[-k:], old, new))
        fitted = self.computed_from
        if fitted is not None:  # a coarser copy of a lattice: as if fitted on voxels as much coarser
            fitted = fitted.rescaled([f * v / o for f, o, v in zip(fitted.voxel_size[-k:], old, new)])
        return self.with_(
            shape=self.shape[:-k] + shape,
            voxel_size=self.voxel_size[:-k] + new,
            translation=self.translation[:-k] + shift,
            computed_from=fitted,
        )

    @staticmethod
    def default_axes(ndim: int) -> tuple[str, ...]:
        names = ("t", "c", "z", "y", "x")
        return names[-ndim:] if ndim <= 5 else tuple(f"d{i}" for i in range(ndim))
