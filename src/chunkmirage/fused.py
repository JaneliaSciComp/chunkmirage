"""Ops run back to back on one block: what a pipeline stage computes for a chunk.

Imports nothing but numpy and the ops, so the browser engine runs this same code in
Pyodide: the page reads the padded input, this computes the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op


def _ratio(a: ArrayInfo, b: ArrayInfo, k: int) -> tuple[float, ...]:
    """Input voxels per output voxel along the last ``k`` axes, from ``a``'s grid to ``b``'s."""
    return tuple(float(w) / float(v) for v, w in zip(a.voxel_size[-k:], b.voxel_size[-k:]))


def _infos(info: ArrayInfo, ops: Sequence[Op]) -> tuple[list[ArrayInfo], int]:
    """Each op's output, and how many trailing axes every op keeps."""
    infos, kept = [info], info.ndim
    for op in ops:
        infos.append(op.output_info(infos[-1]))
        kept = min(kept, infos[-1].ndim)
    return infos, kept


def scale(info: ArrayInfo, ops: Sequence[Op]) -> tuple[float, ...]:
    """Input voxels each output voxel spans along the axes ``ops`` keep: 1 for ops that keep
    the grid. Only a stage's first op may change it (``downsample``, or a model that reads 8
    nm voxels and writes 16 nm ones)."""
    infos, kept = _infos(info, ops)
    for i, (a, b) in enumerate(zip(infos[1:], infos[2:]), 1):
        if any(abs(r - 1) > 1e-9 for r in _ratio(a, b, kept)):
            raise ValueError(f"op {ops[i].name!r} changes the voxel size: only a stage's first op may")
    return _ratio(infos[0], infos[-1], kept) if ops else (1.0,) * kept


def plan(info: ArrayInfo, ops: Sequence[Op]) -> tuple[ArrayInfo, int, tuple[int, ...]]:
    """What ``ops`` make of input ``info``: the output's info (its chunks still the input's),
    how many leading axes they consume (the channels of a ``stack://`` source, which are
    read whole), and the halo the axes they keep need, in input voxels: the sum of the ops',
    those after a first op that changes the grid counted in its output's voxels (``scale``).
    An op may also add leading axes (a model's channels; read whole too): the output's axes
    are those it added, then those kept. Planned on the finest level, the halo is the widest
    any level needs."""
    infos, kept = _infos(info, ops)
    out, r = infos[-1], scale(info, ops)
    lead = info.ndim - kept
    # each op as it runs on this level (an op in physical units counts its halo in its voxels)
    halos = [op.for_level(out).halo_for(kept) for op in ops]
    halo = tuple(
        int(np.ceil((halos[0][a] if halos else 0) + r[a] * sum(h[a] for h in halos[1:]) - 1e-9))
        for a in range(kept)
    )
    return out, lead, halo


def _whole(values, what: str) -> tuple[int, ...]:
    out = tuple(int(round(v)) for v in values)
    if any(abs(v - o) > 1e-6 for v, o in zip(values, out)):
        raise ValueError(f"{what} {tuple(values)} is not whole voxels of the input: choose chunks the voxel ratio divides")
    return out


def input_box(info: ArrayInfo, out_box: Box, lead: int, halo: Sequence[int],
              scale: Sequence[float] | None = None) -> Box:
    """The input to read for output box ``out_box``: every leading axis, the kept ones
    (the last ``len(halo)`` of the output's), on the input's grid (``scale`` input voxels
    to each output voxel), padded."""
    k = len(halo)
    r = scale or (1.0,) * k
    start = _whole([o * f for o, f in zip(out_box.start[-k:], r)], "an output chunk's start")
    stop = _whole([o * f for o, f in zip(out_box.stop[-k:], r)], "an output chunk's end")
    padded = Box(start, stop).pad(halo)
    return Box((0,) * lead + padded.start, info.shape[:lead] + padded.stop)


def run(ops: Sequence[Op], block: np.ndarray, in_box: Box, out_box: Box, out: ArrayInfo,
        halo: Sequence[int] | None = None, scale: Sequence[float] | None = None):
    """``ops`` on ``block`` (the input over ``in_box``), cropped to ``out_box``. ``halo`` and
    ``scale`` are ``plan``'s and ``scale``'s: the halo's length is how many trailing axes the
    ops keep (by default all the output's). Each op returns its block as it came, or shaved
    by its own halo on every side (as a valid convolution does), and may drop leading axes
    or add some; a first op that changes the grid returns its output's voxels over either."""
    k = len(halo) if halo is not None else min(in_box.ndim, out.ndim)
    r = tuple(scale) if scale is not None else (1.0,) * k
    space = Box(in_box.start[-k:], in_box.stop[-k:])  # where the block sits in the kept axes
    box = in_box  # and in all the axes it has
    for i, op in enumerate(ops):
        level_op = op.for_level(out)
        result = level_op.apply_at(block, box)
        h = level_op.halo_for(k)
        got = result.shape[-k:] if result.ndim >= k else None
        if i == 0 and any(abs(f - 1) > 1e-9 for f in r):  # onto the output's grid
            for cut in (h, (0,) * k):  # shaved by its halo first, as a model's output is
                lo = [(a + c) / f for a, c, f in zip(space.start, cut, r)]
                hi = [(b - c) / f for b, c, f in zip(space.stop, cut, r)]
                if all(abs(v - round(v)) < 1e-6 for v in lo + hi) and got == tuple(round(b - a) for a, b in zip(lo, hi)):
                    space = Box(tuple(round(v) for v in lo), tuple(round(v) for v in hi))
                    break
            else:
                raise ValueError(
                    f"op {op.name!r} returned {result.shape} from {block.shape}: on a grid {r} times "
                    f"coarser, the block's voxels or those within its halo {h}"
                )
        elif got != block.shape[-k:]:
            if got != tuple(s - 2 * a for s, a in zip(block.shape[-k:], h)):
                raise ValueError(
                    f"op {op.name!r} changed block shape {block.shape} -> {result.shape}: it may "
                    f"drop or add leading axes, and shave its halo {h} off the last {k}"
                )
            space = space.pad([-a for a in h])  # a valid convolution: it returned the interior
        lead = result.shape[: result.ndim - k]  # leading axes are read whole
        box = Box(box.start[: len(lead)] if result.ndim == block.ndim else (0,) * len(lead),
                  box.stop[: len(lead)] if result.ndim == block.ndim else tuple(lead))
        box = Box(box.start + space.start, box.stop + space.stop)
        block = result
    if block.ndim != out.ndim:
        raise ValueError(f"ops left a block of {block.ndim} axes for an output of {out.ndim}")
    crop = Box(out_box.start[-k:], out_box.stop[-k:]).relative_to(space)
    lead = Box(out_box.start[: out.ndim - k], out_box.stop[: out.ndim - k])
    return np.asarray(block[lead.slices() + crop.slices()], dtype=out.dtype)


def pad_edge(data: np.ndarray, box: Box, shape: Sequence[int]) -> np.ndarray:
    """``data``, read over ``box`` clipped to an array of ``shape``, extended to all of
    ``box`` by repeating the nearest voxel inside (``Source.read_padded(edge=True)``)."""
    clipped = box.clip(shape)
    if clipped == box:
        return data
    inner = clipped.relative_to(box)
    return np.pad(data, [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)], "edge")
