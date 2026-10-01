"""chunkmirage.fused, the code a pipeline stage and the browser engine's Pyodide workers share:
a chunk computed the browser's way (read the clipped box, pad it here, run the ops) is the
pipeline's chunk."""

import numpy as np
import pytest

from chunkmirage import Pipeline, fused, open_source
from chunkmirage.ops import op_from_spec

pytest.importorskip("scipy")

A = "synthetic://blobs+noise?shape=40,72,72&chunk=16,32,32&levels=1&seed=2"


@pytest.mark.parametrize(
    "source, specs",
    [
        (A, [{"op": "gaussian", "sigma": 2}, {"op": "threshold", "low": 120}]),
        (A, [{"op": "spots", "threshold": 10}]),
        (
            f"stack://{A}|warp://{A}?field=swirl&angle=30",
            [{"op": "contacts", "radius": 2, "a_low": 120, "b_low": 120}, {"op": "label"}],
        ),
    ],
)
def test_a_chunk_computed_the_browsers_way_is_the_pipelines(source, specs):
    ms = open_source(source)
    p = Pipeline(ms, specs, chunk_shape=(16, 32, 32))
    ops = [op_from_spec(s) for s in specs]
    src = ms.levels[0]
    out, lead, halo = fused.plan(src.info, ops)
    out = out.with_(chunk_shape=p.info(0).chunk_shape)
    for idx in [(0, 0, 0), (1, 1, 1), tuple(g - 1 for g in out.chunk_grid)]:  # edges included
        out_box = out.chunk_box(idx)
        in_box = fused.input_box(src.info, out_box, lead, halo)
        clipped = in_box.clip(src.info.shape)
        block = fused.pad_edge(src.read(clipped), in_box, src.info.shape)
        np.testing.assert_array_equal(fused.run(ops, block, in_box, out_box, out), p.chunk(0, idx))
