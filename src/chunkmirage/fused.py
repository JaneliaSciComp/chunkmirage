"""Ops run back to back on one block: what a pipeline stage computes for a chunk.

Imports nothing but numpy and the ops, so the browser engine runs this same code in
Pyodide: the page reads the padded input, this computes the chunk.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op


def plan(info: ArrayInfo, ops: Sequence[Op]) -> tuple[ArrayInfo, int, tuple[int, ...]]:
    """What ``ops`` make of input ``info``: the output's info (its chunks still the input's),
    how many leading axes they consume (the channels of a ``stack://`` source, which are
    read whole), and the halo the axes they keep need, the sum of the ops'. An op may also
    add leading axes (a model's channels; read whole too): the output's axes are those it
    added, then those kept. Planned on the finest level, the halo is the widest any level
    needs."""
    out, kept = info, info.ndim
    for op in ops:
        out = op.output_info(out)
        kept = min(kept, out.ndim)  # the trailing axes every op so far kept
    lead = info.ndim - kept
    # each op as it runs on this level (an op in physical units counts its halo in its voxels)
    halo = tuple(sum(op.for_level(out).halo_for(kept)[a] for op in ops) for a in range(kept))
    return out, lead, halo


def input_box(info: ArrayInfo, out_box: Box, lead: int, halo: Sequence[int]) -> Box:
    """The input to read for output box ``out_box``: every leading axis, the kept ones
    (the last ``len(halo)`` of the output's) padded."""
    k = len(halo)
    padded = Box(out_box.start[-k:], out_box.stop[-k:]).pad(halo)
    return Box((0,) * lead + padded.start, info.shape[:lead] + padded.stop)


def run(ops: Sequence[Op], block: np.ndarray, in_box: Box, out_box: Box, out: ArrayInfo,
        halo: Sequence[int] | None = None):
    """``ops`` on ``block`` (the input over ``in_box``), cropped to ``out_box``. ``halo`` is
    ``plan``'s: its length is how many trailing axes the ops keep (by default all the
    output's). Each op returns its block as it came, or shaved by its own halo on every
    side (as a valid convolution does), and may drop leading axes or add some."""
    k = len(halo) if halo is not None else min(in_box.ndim, out.ndim)
    space = Box(in_box.start[-k:], in_box.stop[-k:])  # where the block sits in the kept axes
    box = in_box  # and in all the axes it has
    for op in ops:
        level_op = op.for_level(out)
        result = level_op.apply_at(block, box)
        if result.ndim < k or result.shape[-k:] != block.shape[-k:]:
            h = level_op.halo_for(k)
            if result.ndim < k or result.shape[-k:] != tuple(s - 2 * a for s, a in zip(block.shape[-k:], h)):
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
