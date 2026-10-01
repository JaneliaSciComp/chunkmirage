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
    read whole), and the halo the output's axes need, the sum of the ops'."""
    out = info
    for op in ops:
        out = op.output_info(out)
    lead = info.ndim - out.ndim
    if lead < 0:
        raise ValueError(
            f"ops {[op.name for op in ops]} add axes ({info.ndim} -> {out.ndim}), which a"
            " stage cannot serve yet"
        )
    halo = tuple(sum(op.halo_for(out.ndim)[a] for op in ops) for a in range(out.ndim))
    return out, lead, halo


def input_box(info: ArrayInfo, out_box: Box, lead: int, halo: Sequence[int]) -> Box:
    """The input to read for output box ``out_box``: every leading axis, the rest padded."""
    padded = out_box.pad(halo)
    return Box((0,) * lead + padded.start, info.shape[:lead] + padded.stop)


def run(ops: Sequence[Op], block: np.ndarray, in_box: Box, out_box: Box, out: ArrayInfo):
    """``ops`` on ``block`` (the input over ``in_box``), cropped to ``out_box``."""
    ndim = out.ndim
    box = in_box  # where the block sits, in the axes it currently has
    for op in ops:
        result = op.apply_at(block, box)
        if result.ndim < ndim or result.shape[-ndim:] != block.shape[-ndim:]:
            raise ValueError(f"op {op.name!r} changed block shape {block.shape} -> {result.shape}")
        if result.ndim < block.ndim:  # this op consumed the leading axes
            drop = block.ndim - result.ndim
            box = Box(box.start[drop:], box.stop[drop:])
        block = result
    if block.ndim != ndim:
        raise ValueError(f"ops left a block of {block.ndim} axes for an output of {ndim}")
    padded = Box(in_box.start[-ndim:], in_box.stop[-ndim:])
    crop = out_box.relative_to(padded)
    return np.asarray(block[crop.slices()], dtype=out.dtype)


def pad_edge(data: np.ndarray, box: Box, shape: Sequence[int]) -> np.ndarray:
    """``data``, read over ``box`` clipped to an array of ``shape``, extended to all of
    ``box`` by repeating the nearest voxel inside (``Source.read_padded(edge=True)``)."""
    clipped = box.clip(shape)
    if clipped == box:
        return data
    inner = clipped.relative_to(box)
    return np.pad(data, [(a, s - b) for a, b, s in zip(inner.start, inner.stop, box.shape)], "edge")
