"""Coordinate transforms: the internal model every registration format is read into.

A transform maps *points* to points. Points are ``(n, d)`` float arrays whose columns follow
the axis order of the coordinate system (C order for OME-Zarr). Readers for each format
(OME-Zarr 0.6 in ``chunkmirage.ngff``; bigstream, BigWarp, ITK later) build these few
classes, and the resampler in ``chunkmirage.sources.scene`` only ever calls ``apply``.

``simplify`` folds runs of linear pieces into one matrix, so a typical registration chain
(grid → physical → field → affine → source index) costs two matrix products and one field
lookup per point.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence as Seq

import numpy as np


class Transform(ABC):
    """A function from points in one coordinate system to points in another."""

    ndim_in: int | None = None
    ndim_out: int | None = None

    @abstractmethod
    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Map ``(n, ndim_in)`` points to ``(n, ndim_out)`` points."""

    def inverse(self) -> Transform | None:
        """The closed-form inverse, or None if there is none."""
        return None

    def approximate_inverse(self) -> Transform | None:
        """An inverse that may be estimated numerically; defaults to the exact one."""
        return self.inverse()

    @abstractmethod
    def describe(self) -> dict:
        """JSON-able identity; equal descriptions mean equal functions (used in cache keys)."""

    def digest(self) -> str:
        payload = json.dumps(self.describe(), sort_keys=True, default=str).encode()
        return hashlib.sha1(payload).hexdigest()[:12]


class Affine(Transform):
    """``y = A @ x + t``, stored as the ``(d_out, d_in + 1)`` matrix ``[A | t]``."""

    def __init__(self, matrix):
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[1] < 2:
            raise ValueError(f"affine matrix must be (d_out, d_in + 1), got shape {m.shape}")
        self.matrix = m
        self.ndim_out, self.ndim_in = m.shape[0], m.shape[1] - 1

    @classmethod
    def identity(cls, n: int) -> Affine:
        return cls(np.hstack([np.eye(n), np.zeros((n, 1))]))

    @classmethod
    def from_linear(cls, linear, offset=None) -> Affine:
        a = np.asarray(linear, dtype=float)
        t = np.zeros(a.shape[0]) if offset is None else np.asarray(offset, dtype=float)
        return cls(np.hstack([a, t[:, None]]))

    @classmethod
    def scale_translation(cls, scale, translation=None) -> Affine:
        s = np.asarray(scale, dtype=float)
        return cls.from_linear(np.diag(s), translation)

    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:, :-1]

    @property
    def offset(self) -> np.ndarray:
        return self.matrix[:, -1]

    def apply(self, pts):
        return pts @ self.linear.T + self.offset

    def then(self, other: Affine) -> Affine:
        """``other ∘ self``: apply self first."""
        return Affine.from_linear(
            other.linear @ self.linear, other.linear @ self.offset + other.offset
        )

    def inverse(self):
        if self.ndim_in != self.ndim_out:
            return None
        a = self.linear
        if abs(np.linalg.det(a)) < 1e-12:
            return None
        ai = np.linalg.inv(a)
        return Affine.from_linear(ai, -ai @ self.offset)

    def describe(self):
        return {"affine": np.round(self.matrix, 12).tolist()}


class VectorField:
    """A regularly sampled vector field: an array with one vector axis, plus the affine that
    maps its other (grid) axes to points in the transform's input space.

    ``sample(pts)`` interpolates the vectors at arbitrary points, reading only the window of
    the array those points touch. Outside the field, values are clamped to the nearest edge.
    """

    def __init__(self, read, shape, vector_axis: int, grid_to_space: Affine, key: str, order=1):
        self._read = read  # callable(tuple[slice, ...]) -> ndarray
        self.shape = tuple(int(s) for s in shape)
        self.vector_axis = vector_axis
        self.grid_shape = tuple(s for i, s in enumerate(self.shape) if i != vector_axis)
        self.ncomp = self.shape[vector_axis]
        self.to_grid = grid_to_space.inverse()
        if self.to_grid is None:
            raise ValueError(f"{key}: field grid transform is not invertible")
        self.spacing = np.linalg.norm(grid_to_space.linear, axis=0)  # space units per grid step
        self.key = key
        self.order = order

    @property
    def ndim(self) -> int:
        return len(self.grid_shape)

    def sample(self, pts: np.ndarray) -> np.ndarray:
        """Vectors at ``pts`` (NaN rows, e.g. unconvergent inverse points, stay NaN)."""
        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            out = np.full((pts.shape[0], self.ncomp), np.nan)
            if finite.any():
                out[finite] = self.sample(pts[finite])
            return out
        from scipy.ndimage import map_coordinates

        g = self.to_grid.apply(pts)
        hi = np.asarray(self.grid_shape, dtype=float) - 1
        g = np.clip(g, 0.0, hi)
        lo = np.floor(g.min(axis=0)).astype(int)
        top = np.minimum(np.floor(g.max(axis=0)).astype(int) + 2, self.grid_shape)
        slices = [slice(int(a), int(b)) for a, b in zip(lo, top)]
        slices.insert(self.vector_axis, slice(None))
        win = np.moveaxis(np.asarray(self._read(tuple(slices)), dtype=float), self.vector_axis, 0)
        local = (g - lo).T
        out = np.empty((pts.shape[0], self.ncomp))
        for c in range(self.ncomp):
            out[:, c] = map_coordinates(
                win[c], local, order=self.order, mode="nearest", prefilter=False
            )
        return out


class SwirlField:
    """A procedural displacement field: a twist about an axis, computed exactly from
    coordinates, so nothing is stored and any parameter can change at any time.

    Points at distance ``r`` from the axis through ``centre`` turn about it by
    ``angle * exp(-(r / radius)**2)`` degrees: fully on the axis, fading out over about
    ``radius``. Only the two ``plane`` axes move. Same interface as ``VectorField``.
    """

    order = 1

    def __init__(self, centre, radius: float, angle: float, plane=(-2, -1)):
        self.centre = np.asarray(centre, dtype=float)
        self.ndim = self.ncomp = len(self.centre)
        self.radius = float(radius)
        self.angle = float(angle)
        self.plane = tuple(int(p) % self.ndim for p in plane)
        if self.radius <= 0 or len(set(self.plane)) != 2:
            raise ValueError("swirl needs radius > 0 and two distinct plane axes")
        self.spacing = np.full(self.ndim, self.radius / 1000)  # tolerance scale for inverses
        c = ",".join(f"{v:.6g}" for v in self.centre)
        self.key = f"swirl({c};r={self.radius:.6g};a={self.angle:.6g};plane={self.plane})"

    def sample(self, pts: np.ndarray) -> np.ndarray:
        a, b = self.plane
        u, v = pts[:, a] - self.centre[a], pts[:, b] - self.centre[b]
        theta = np.deg2rad(self.angle) * np.exp(-(u * u + v * v) / self.radius**2)
        cos, sin = np.cos(theta), np.sin(theta)
        d = np.zeros(pts.shape, dtype=float)
        d[:, a] = cos * u - sin * v - u
        d[:, b] = sin * u + cos * v - v
        return d


class Swirls:
    """Several 3-D swirls, their displacements added: swirl ``k`` turns points about the
    axis ``axes[k]`` (a direction, C order) through ``centres[k]`` by
    ``angles[k] * exp(-(d / radii[k])**2)`` degrees, ``d`` the distance from its centre.
    So each is a ball of twist fading in every direction, and a tilted axis moves points
    along all three axes. Where swirls overlap the sum is smooth but no longer a pure
    rotation. Same interface as ``VectorField``.
    """

    order = 1
    CUTOFF = 5.3  # beyond this many radii a swirl moves points by < 1e-12 of its twist

    def __init__(self, centres, axes, radii, angles):
        self.centres = np.asarray(centres, dtype=float).reshape(-1, 3)
        axes = np.asarray(axes, dtype=float).reshape(-1, 3)
        self.axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        self.radii = np.asarray(radii, dtype=float).reshape(-1)
        self.angles = np.asarray(angles, dtype=float).reshape(-1)
        k = len(self.centres)
        if not (len(self.axes) == len(self.radii) == len(self.angles) == k) or k == 0:
            raise ValueError("swirls need one axis, radius and angle per centre")
        if (self.radii <= 0).any() or not np.isfinite(self.axes).all():
            raise ValueError("swirls need radii > 0 and non-zero axes")
        self.ndim = self.ncomp = 3
        self.spacing = np.full(3, self.radii.min() / 1000)  # tolerance scale for inverses
        params = np.concatenate(
            [self.centres, self.axes, self.radii[:, None], self.angles[:, None]], 1
        )
        self.key = "swirls(" + hashlib.sha1(np.round(params, 9).tobytes()).hexdigest()[:16] + ")"

    def sample(self, pts: np.ndarray) -> np.ndarray:
        d = np.zeros(pts.shape, dtype=float)
        lo, hi = np.nanmin(pts, axis=0), np.nanmax(pts, axis=0)
        for c, n, r, a in zip(self.centres, self.axes, self.radii, self.angles):
            if np.linalg.norm(np.clip(c, lo, hi) - c) > self.CUTOFF * r:
                continue  # too far from every point to matter
            v = pts - c
            theta = np.deg2rad(a) * np.exp(-np.einsum("ij,ij->i", v, v) / r**2)
            cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
            # Rodrigues: v turned about n by theta
            d += v * (cos - 1) + np.cross(n, v) * sin + np.outer(v @ n, n) * (1 - cos)
        return d


class Displacements(Transform):
    """``y = x + d(x)`` with ``d`` a vector field in the input space (sampled, or
    procedural like ``SwirlField``)."""

    def __init__(self, field: VectorField):
        self.field = field
        self.ndim_in = self.ndim_out = field.ndim

    def apply(self, pts):
        return pts + self.field.sample(pts)

    def approximate_inverse(self):
        return InverseDisplacements(self.field)

    def describe(self):
        return {"displacements": self.field.key, "order": self.field.order}


class InverseDisplacements(Transform):
    """Numerical inverse of ``x -> x + d(x)``: solves ``x = y - d(x)`` by fixed-point
    iteration, which converges wherever the field's Jacobian has norm below one (true for
    the smooth, non-folding warps registration produces). Points that do not converge
    (the warp folds there, so no unique inverse exists) come back as NaN, which the
    resampler renders as empty rather than as a wrong value."""

    def __init__(self, field: VectorField, max_iter: int = 50, tol: float | None = None):
        self.field = field
        self.ndim_in = self.ndim_out = field.ndim
        self.max_iter = max_iter
        self.tol = tol if tol is not None else 1e-3 * float(np.min(field.spacing))

    def apply(self, pts):
        x = pts - self.field.sample(pts)
        err = np.full(pts.shape[0], np.inf)
        for _ in range(self.max_iter):
            residual = x + self.field.sample(x) - pts
            err = np.max(np.abs(residual), axis=1, initial=0.0)
            if not np.any(err >= self.tol):
                break
            x = x - residual
        else:  # out of iterations: judge the points being returned
            err = np.max(np.abs(x + self.field.sample(x) - pts), axis=1, initial=0.0)
        x[~(err < 10 * self.tol)] = np.nan
        return x

    def approximate_inverse(self):
        return Displacements(self.field)

    def describe(self):
        return {"inverse_displacements": self.field.key, "order": self.field.order}


class Coordinates(Transform):
    """``y = c(x)``: the field stores absolute output positions."""

    def __init__(self, field: VectorField):
        self.field = field
        self.ndim_in, self.ndim_out = field.ndim, field.ncomp

    def apply(self, pts):
        return self.field.sample(pts)

    def describe(self):
        return {"coordinates": self.field.key, "order": self.field.order}


class Sequence(Transform):
    """Apply ``parts`` in order (the first one first)."""

    def __init__(self, parts: Seq[Transform]):
        flat: list[Transform] = []
        for p in parts:
            flat.extend(p.parts if isinstance(p, Sequence) else [p])
        if not flat:
            raise ValueError("empty sequence")
        for a, b in zip(flat, flat[1:]):
            if a.ndim_out is not None and b.ndim_in is not None and a.ndim_out != b.ndim_in:
                raise ValueError(f"sequence dimension mismatch: {a.ndim_out} -> {b.ndim_in}")
        self.parts = flat
        self.ndim_in, self.ndim_out = flat[0].ndim_in, flat[-1].ndim_out

    def apply(self, pts):
        for p in self.parts:
            pts = p.apply(pts)
        return pts

    def inverse(self):
        inv = [p.inverse() for p in reversed(self.parts)]
        return None if any(i is None for i in inv) else Sequence(inv)

    def approximate_inverse(self):
        inv = [p.approximate_inverse() for p in reversed(self.parts)]
        return None if any(i is None for i in inv) else Sequence(inv)

    def describe(self):
        return {"sequence": [p.describe() for p in self.parts]}


class ByDimension(Transform):
    """Lower-dimensional transforms applied to subsets of axes: ``parts`` are
    ``(transform, input_axes, output_axes)``."""

    def __init__(
        self, parts: Seq[tuple[Transform, Seq[int], Seq[int]]], ndim_in: int, ndim_out: int
    ):
        self.parts = [(t, list(map(int, ia)), list(map(int, oa))) for t, ia, oa in parts]
        covered = sorted(a for _, _, oa in self.parts for a in oa)
        if covered != list(range(ndim_out)):
            raise ValueError(f"byDimension output axes {covered} must cover 0..{ndim_out - 1} once")
        self.ndim_in, self.ndim_out = ndim_in, ndim_out

    def apply(self, pts):
        out = np.zeros((pts.shape[0], self.ndim_out))
        for t, ia, oa in self.parts:
            out[:, oa] = t.apply(pts[:, ia])
        return out

    def _invert(self, approx: bool):
        if sorted(a for _, ia, _ in self.parts for a in ia) != list(range(self.ndim_in)):
            return None
        inv = []
        for t, ia, oa in self.parts:
            ti = t.approximate_inverse() if approx else t.inverse()
            if ti is None:
                return None
            inv.append((ti, oa, ia))
        return ByDimension(inv, self.ndim_out, self.ndim_in)

    def inverse(self):
        return self._invert(False)

    def approximate_inverse(self):
        return self._invert(True)

    def as_affine(self) -> Affine | None:
        if not all(isinstance(t, Affine) for t, _, _ in self.parts):
            return None
        m = np.zeros((self.ndim_out, self.ndim_in + 1))
        for t, ia, oa in self.parts:
            m[np.ix_(oa, ia)] = t.linear
            m[oa, -1] = t.offset
        return Affine(m)

    def describe(self):
        return {"byDimension": [[t.describe(), ia, oa] for t, ia, oa in self.parts]}


class Bijection(Transform):
    """A transform with an explicitly stored inverse."""

    def __init__(self, forward: Transform, inverse: Transform):
        self.forward, self._inverse = forward, inverse
        self.ndim_in, self.ndim_out = forward.ndim_in, forward.ndim_out

    def apply(self, pts):
        return self.forward.apply(pts)

    def inverse(self):
        return Bijection(self._inverse, self.forward)

    def describe(self):
        return {"bijection": [self.forward.describe(), self._inverse.describe()]}


def simplify(t: Transform) -> Transform:
    """Unwrap bijections, flatten sequences and fold adjacent linear pieces into one affine."""
    if isinstance(t, Bijection):
        return simplify(t.forward)
    if isinstance(t, ByDimension):
        parts = [(simplify(p), ia, oa) for p, ia, oa in t.parts]
        by = ByDimension(parts, t.ndim_in, t.ndim_out)
        return by.as_affine() or by
    if not isinstance(t, Sequence):
        return t
    out: list[Transform] = []
    for p in t.parts:
        p = simplify(p)
        for q in p.parts if isinstance(p, Sequence) else [p]:
            if out and isinstance(out[-1], Affine) and isinstance(q, Affine):
                out[-1] = out[-1].then(q)
            else:
                out.append(q)
    return out[0] if len(out) == 1 else Sequence(out)
