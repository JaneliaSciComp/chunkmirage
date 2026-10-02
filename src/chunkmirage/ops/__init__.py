"""Ops: pure functions on blocks with declared halo, output dtype and cacheability."""

from chunkmirage.ops.base import Op, get_op, list_ops, op_from_spec, register
from chunkmirage.ops.combine import Contacts, NormalizedDifference
from chunkmirage.ops.filters import Diff, Downsample, Gaussian, Gradient, Uniform
from chunkmirage.ops.pointwise import Cast, Scale, Threshold
from chunkmirage.ops.segment import DoG, Label, Morphology, Spots
from chunkmirage.ops.terrain import Hillshade, Slope

__all__ = [
    "Cast",
    "Contacts",
    "Diff",
    "Downsample",
    "Gradient",
    "DoG",
    "Label",
    "Morphology",
    "NormalizedDifference",
    "Gaussian",
    "Hillshade",
    "Op",
    "Scale",
    "Slope",
    "Spots",
    "Threshold",
    "Uniform",
    "get_op",
    "list_ops",
    "op_from_spec",
    "register",
]
