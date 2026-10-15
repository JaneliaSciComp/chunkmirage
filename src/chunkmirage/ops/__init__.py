"""Ops: pure functions on blocks with declared halo, output dtype and cacheability.

The core has two: ``downsample``, which the pipeline itself runs to make the coarser levels
of an op bound to one resolution, and ``python``, code carried in the spec. Every other op
(filters, thresholds, detectors, terrain) is in ``chunkmirage_contrib.ops``, found through
the ``chunkmirage.ops`` entry point as any other package's ops are.
"""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.code import PythonOp, python_op
from chunkmirage.ops.downsample import Downsample

__all__ = [
    "Downsample",
    "Op",
    "PythonOp",
    "get_op",
    "list_ops",
    "op_from_spec",
    "python_op",
    "register",
]
