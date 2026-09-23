"""chunkmirage: spoof chunked array formats over HTTP with on-the-fly processing."""

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.pipeline import Pipeline, PipelineSpec
from chunkmirage.server import create_app
from chunkmirage.sources import open_source

__all__ = ["ArrayInfo", "Box", "Pipeline", "PipelineSpec", "create_app", "open_source"]
__version__ = "0.1.0"
