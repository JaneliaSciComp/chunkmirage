"""Ops: pure functions on blocks with declared halo, output dtype and cacheability."""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.combine import Contacts
from chunkmirage.ops.filters import Gaussian, Uniform
from chunkmirage.ops.pointwise import Cast, Scale, Threshold
from chunkmirage.ops.segment import DoG, Label, Morphology, Spots

__all__ = [
    "Cast",
    "Contacts",
    "DoG",
    "Label",
    "Morphology",
    "Gaussian",
    "Op",
    "Scale",
    "Spots",
    "Threshold",
    "Uniform",
    "get_op",
    "list_ops",
    "op_from_spec",
    "register",
]
