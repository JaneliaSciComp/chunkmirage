"""Ops that change the length of a channel axis they keep: the channels are read whole, so
an op may select or combine them, and only the spatial axes are chunked."""

import numpy as np
from pydantic import Field

from chunkmirage import Pipeline, open_source
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.ops.base import Op

A = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1&seed=1"
B = "synthetic://blobs?shape=16,16,16&chunk=8,8,8&levels=1&seed=2"


class Pick(Op):
    """Channels ``start:stop``, kept as a channel axis."""

    name = "test_pick"
    start: int = Field(1, description="first channel")
    stop: int = Field(2, description="one past the last")

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        n = self.stop - self.start
        return info.with_(shape=(n, *info.shape[1:]), chunk_shape=(n, *info.chunk_shape[1:]))

    def apply(self, block):
        return block[self.start : self.stop]


class Sum(Op):
    """All channels added into one, with a halo, so the spatial axes are padded."""

    name = "test_sum"
    halo = 1

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        return info.with_(shape=(1, *info.shape[1:]), chunk_shape=(1, *info.chunk_shape[1:]), dtype=np.dtype("uint16"))

    def apply(self, block):
        return block.astype(np.uint16).sum(axis=0, keepdims=True)


def _whole(p):
    return p.read(0, Box((0,) * p.info(0).ndim, p.info(0).shape))


def test_an_op_selecting_a_channel_gets_every_channel():
    stack = open_source(f"stack://{A}|{B}")
    p = Pipeline(stack, [Pick()])
    assert p.info(0).shape == (1, 16, 16, 16)
    np.testing.assert_array_equal(_whole(p)[0], _whole(Pipeline(open_source(B), [])))


def test_an_op_combining_channels_with_a_halo_and_ops_after_it():
    stack = open_source(f"stack://{A}|{B}")
    p = Pipeline(stack, [Sum(), {"op": "threshold", "low": 200}])
    a, b = (_whole(Pipeline(open_source(s), [])).astype(np.uint16) for s in (A, B))
    np.testing.assert_array_equal(_whole(p)[0], (a + b >= 200).astype(np.uint8) * 255)
