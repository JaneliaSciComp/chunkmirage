"""Ops: pure functions on blocks with declared halo, output dtype and cacheability."""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.filters import Gaussian, Uniform
from chunkmirage.ops.pointwise import Cast, Scale, Threshold

__all__ = [
    "Cast",
    "Gaussian",
    "Op",
    "Scale",
    "Threshold",
    "Uniform",
    "get_op",
    "list_ops",
    "op_from_spec",
    "register",
]
