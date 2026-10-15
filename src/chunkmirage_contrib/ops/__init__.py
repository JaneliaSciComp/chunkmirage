"""The ops shipped with chunkmirage: pointwise, filters, segmentation, combining channels
and terrain. Importing this registers them (``@register``); an installed package finds them
through the ``chunkmirage.ops`` entry point without importing anything."""

from chunkmirage_contrib.ops.combine import Contacts, NormalizedDifference
from chunkmirage_contrib.ops.filters import Diff, Gaussian, Gradient, Uniform
from chunkmirage_contrib.ops.pointwise import Cast, Scale, Threshold
from chunkmirage_contrib.ops.segment import DoG, Label, Morphology, Spots
from chunkmirage_contrib.ops.terrain import Hillshade, Slope

__all__ = [
    "Cast",
    "Contacts",
    "Diff",
    "DoG",
    "Gaussian",
    "Gradient",
    "Hillshade",
    "Label",
    "Morphology",
    "NormalizedDifference",
    "Scale",
    "Slope",
    "Spots",
    "Threshold",
    "Uniform",
]
