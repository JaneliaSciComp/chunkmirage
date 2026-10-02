"""A model trained at one resolution, served at every zoom without running it at any other:
the shape a live-inference consumer's op takes, with a stand-in for the network.

    uv run python examples/model_at_one_resolution.py [--check]

``Membranes`` stands in for a network trained on 8 nm voxels: it reads 8 nm EM with 6 voxels
of context on each side, and writes two channels of 16 nm voxels (dark membranes, and
everything else) for only the interior it can compute, as a valid convolution does. It says
so with ``input_voxel_size`` and its ``output_info``. OpenOrganelle's HeLa cell is 5.24 x 4
x 4 nm, with no level at 8 nm isotropic, so the pipeline reads its finest level resampled to
8 nm, runs the op there, caches what it makes, and serves the coarser levels downsampled from
that: zoomed out, the viewer gets a pyramid the op never ran on. A coarse chunk is made
from the finer ones under it, so a zoomed-out view costs the op its whole region at 16 nm,
once (it is cached). ``--check`` computes one chunk of the next level, made from eight of
the op's, and says how often the op ran, without serving.

A real consumer ships its op in a package of its own, registered through the
``chunkmirage.ops`` entry point, and its forward pass goes where ``apply`` is here.
"""

from __future__ import annotations

import argparse

import numpy as np

from chunkmirage import Pipeline, create_app, open_source
from chunkmirage.core import ArrayInfo
from chunkmirage.neuroglancer import source_url, viewer_link
from chunkmirage.ops import Op, register

EM = "https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8"
CONTEXT = 6  # input voxels each side the "network" needs
RUNS = [0]


@register
class Membranes(Op):
    """Stand-in for a network trained on 8 nm voxels: dark membranes, and the rest, as two
    channels of 16 nm voxels."""

    name = "membranes"
    cache = True
    halo = CONTEXT

    def input_voxel_size(self):
        return (8.0, 8.0, 8.0)

    def output_info(self, info: ArrayInfo) -> ArrayInfo:
        out = info.rescaled((16.0, 16.0, 16.0))  # coarser voxels, centres where OME puts them
        return out.with_(
            shape=(2, *out.shape), chunk_shape=(2, *out.chunk_shape), dtype=np.dtype("uint8"),
            voxel_size=(1.0, *out.voxel_size), units=("", *out.units), axes=("c", *out.axes),
            translation=(0.0, *out.translation), kind="image",
        )

    def apply(self, block: np.ndarray) -> np.ndarray:
        RUNS[0] += 1
        from scipy.ndimage import uniform_filter

        b = block.astype(np.float32)
        dark = uniform_filter(b, 2 * CONTEXT + 1) - uniform_filter(b, 3)  # darker than around it
        inner = dark[CONTEXT:-CONTEXT, CONTEXT:-CONTEXT, CONTEXT:-CONTEXT]  # what it can compute
        z, y, x = (n // 2 for n in inner.shape)
        p = 1 / (1 + np.exp(-inner[: 2 * z, : 2 * y, : 2 * x].reshape(z, 2, y, 2, x, 2).mean((1, 3, 5)) / 6))
        return (np.stack([p, 1 - p]) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compute one coarse chunk and stop")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    pipe = Pipeline(open_source(EM, cache_bytes=1 << 30), [Membranes()], chunk_shape=(32, 64, 64))
    for k in range(pipe.num_levels):
        i = pipe.info(k)
        print(f"s{k}: {i.shape[1:]} voxels of {i.voxel_size[1:]} nm, from {i.translation[1:]}", flush=True)
    if args.check:
        info = pipe.info(1)
        idx = tuple(g // 2 for g in info.chunk_grid)
        chunk = pipe.read(1, info.chunk_box(idx))
        print(f"s1 chunk {idx}: {chunk.shape}, membranes {chunk[0].mean():.0f}/255 on average; "
              f"the op ran {RUNS[0]} times, all on 8 nm voxels")
        return
    import uvicorn

    app = create_app({"membranes": pipe})
    src = source_url(f"http://localhost:{args.port}", "membranes", "zarr3", "zarr3", pipe.digest())
    print("neuroglancer:", viewer_link({"membranes": pipe}, {"membranes": src}))
    uvicorn.run(app, port=args.port)


if __name__ == "__main__":
    main()
