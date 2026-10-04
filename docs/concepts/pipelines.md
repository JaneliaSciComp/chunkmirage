# Pipelines, ops and halos

A **pipeline** is a multiscale source followed by an ordered list of **ops**. Each scale
level is processed independently (see the caveat at the end).

```python
Pipeline(open_source(path), [Gaussian(sigma=2), Threshold(low=120)], chunk_shape=(64, 64, 64))
```

or, as the JSON the REST API and CLI exchange (`PipelineSpec`):

```json
{"source": "/data/vol.zarr/em", "chunk_shape": [64, 64, 64],
 "ops": [{"op": "gaussian", "sigma": 2}, {"op": "threshold", "low": 120}]}
```

## Every stage is a chunked array

Internally, stage 0 is the raw source re-chunked to the output chunk shape. Ops are grouped
into *segments*, each ending at an op with `cache=True` (or the end of the list), and each
segment is one `ChunkedSource` stage whose `compute_chunk(index)`:

1. computes the output chunk's box;
2. pads it by the **sum of the segment's halos**;
3. reads that padded box from the previous stage via `read_padded`, voxels beyond the volume
   repeating the nearest one inside, so a filter or detector sees no step at its border
   (the spec's `"padding": "zero"` pads with zeros instead, as a model trained on
   zero-padded blocks expects; the browser engine always repeats the edge);
4. calls each op's `apply_at(block, box)` in turn on the whole padded block;
5. crops the padding off and casts to the last op's declared output dtype.

Fusing uncached ops keeps the read footprint to a few upstream chunks per output chunk; see
[Caching](caching.md#fusion-or-why-uncached-ops-are-cheap).

Reading an arbitrary box from a stage gathers the covering chunks (cached or computed) and
slices, so a halo read of stage *k-1* costs at most a few chunk lookups.

## Ops

An op is a pydantic model. Its fields are its parameters, which gives every op a JSON
schema for free (`GET /api/ops`).

| attribute              | meaning                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `name`                 | identifier used in specs and the CLI (`--op name:key=val`)              |
| `halo`                 | voxels of context needed on every side; int, per-axis tuple, or a property computed from parameters (e.g. `Gaussian` uses `ceil(sigma * truncate)`) |
| `cache`                | whether this stage's output chunks are memoized; see [Caching](caching.md) |
| `slots`                | at most this many `apply` calls of the op at once, across the process (one per GPU, say, or a memory-hungry step); waiting chunks are queued finest level first and dropped when no request wants them any more ([order of work](caching.md#order-of-work-and-requests-given-up-on)). The op runs as a stage of its own, so reading its input holds no slot. Default `None`: as many as the server computes at once |
| `packages`             | packages `apply` imports beyond numpy, e.g. `("scipy",)`; the schema carries them (`x-packages`) so the browser engine loads them with Python, only for pages whose ops need them |
| `output_dtype(dtype)`  | result dtype; default unchanged                                         |
| `output_kind`          | what the result's values are, `image`, `label` (segment ids) or `mask` (inside or not), carried as the output's `ArrayInfo.kind`: viewers show labels and masks as segmentations. Default `None` (not said); `threshold`, `morphology` and `contacts` make masks, `label` and `spots` labels, `cast` keeps its input's |
| `output_info(info)`    | the result's `ArrayInfo`; default the input's with `output_dtype`. An op that adds leading axes (channels) prepends them here |
| `apply(block)`         | the computation: returns the block's own shape, or the block shaved by `halo` on every side (a valid convolution), with leading axes dropped or added as `output_info` says |
| `apply_at(block, box)` | optional; same but told the block's (halo-padded) position, for position-dependent results such as unique per-chunk labels |
| `input_voxel_size()`   | optional; the voxel size the op must read (its last axes, in the source's units), as a model trained at one resolution does. The pipeline then runs it on one level and makes the coarser ones from its output; see [ops at one resolution](#ops-at-one-resolution). Default `None`: it runs on every level |
| `for_level(info)`      | optional; the op as it runs on a scale level, given that level's `ArrayInfo`; ops in physical units take its voxel size (`slope` and `hillshade` their pixel spacing, which doubles from level to level). Default: the op itself |

What an op returns may differ from what it was given in two ways, the two a model's
output differs from its input:

* **Shaved by its halo.** A valid convolution computes only where its whole kernel lies
  inside the block, so it returns the block `2 × halo` smaller on each axis. Return that
  interior and the stage crops the rest; return the full block and it crops as before.
  `gradient` returns its interior.
* **New leading axes.** An op whose `output_info` prepends axes (a model's channels, the
  components of `gradient`) returns them first; the halo pads only the axes every op keeps,
  and the stage's chunks span the new axes whole (one chunk along `c`). Ops after it in the
  pipeline see the channels as the block's first axis.
* **Channels selected or combined.** A channel axis (named `c`, `channel` or `channels`),
  and any axis before it, is always read whole and never padded, so an op may change its
  length: pick channel 1 of 2, turn affinities into labels, flows into masks. Only the
  axes after the last channel axis are chunked and padded by the halo.

Ops are discovered through the `chunkmirage.ops` entry point, so plugins ship as ordinary
packages. A plugin that fails to import is skipped with a warning naming it and the error,
and the other ops still load. See [Contributing](../contributing.md#adding-an-op) for a template and the
[ops reference](../reference/ops.md) for what is built in.

## Chaining and branching

Chaining is the `ops` list: cellmap-flow's *model then post-processors* is one inference op
with `cache=True` followed by its post-processors.

Branching (one model output served through several post-processors as separate layers) is
not expressible in a single spec yet. Define one pipeline per branch: because cache keys are
prefix hashes, the branches share the cached model output automatically. A DAG spec with
named stages is on the [roadmap](../roadmap.md).

## Transforms

Affine transforms are a Neuroglancer client feature (per-layer, live, no refetch); do not put
them in a pipeline. Non-affine resampling (registration through displacement fields) is a
[`scene://` source](formats.md#scene-sources-ome-zarr-06-transformations), so ops apply to
the registered image. See [Interactivity](interactivity.md#transforms-on-the-fly).

## Re-chunking

`chunk_shape` sets the *output* chunk shape independently of the source. Viewers prefer
around 64³; sources often use 128³ or larger; inference block sizes differ again. Stage 0
reads whatever source chunks cover the requested output chunk.

## Ops at one resolution

Ops run on every level with the same parameters. That is right for thresholds and filters
in voxels, and for ops that measure in physical units through `for_level` (a slope in
degrees is a slope in degrees at every level). It is wrong for a model trained at one
resolution, so an op can say which it reads (`input_voxel_size`), and may write another
(its `output_info` changes the voxel size, through `ArrayInfo.rescaled`). From the first
such op in a pipeline:

* it runs on one level: the source's at that voxel size (`MultiscaleSource.level_for`, to
  within the spec's `level_rtol`, 1% by default), or if none is, the coarsest finer one
  resampled to it: along an axis it shrinks by a whole factor of 2 or more, each voxel is
  the mean of the block it covers (rounded back to an integer dtype, as a stored pyramid
  level is); along the others, linearly (the `scene://` resampler); labels and masks by
  nearest voxel. With
  the spec's `"input_level": "nearest"` it reads the nearest level as it is instead, a
  cheap preview: the op then sees voxels of another size than it asked for, and the output
  is on that level's grid. The ops before it run there too. `GET /api/datasets/{name}`
  says what was read (`reads`: the level, its voxel size, and whether it was resampled);
* its output is cached, whatever its `cache` flag says, since the coarser levels are made
  from it: each by `downsample` from the one above, by the source pyramid's own factors
  (mean for images, the most common value for labels and masks), cached too. So the op runs
  once per chunk of one level, whichever levels the viewer asks for;
* the ops after it run on every level, as usual.

The output has as many levels as the source has from the one read down. A coarse chunk is
made from the finer ones under it, so a zoomed-out view costs the op its whole region at its
own resolution, once, since that is cached; the work for a view the viewer has left is
dropped as usual ([caching](caching.md#order-of-work-and-requests-given-up-on)). A voxel's position
is its centre, as in OME-Zarr: a level of voxels twice as big starts half an old voxel on,
and the served translations say so. An op that changes the grid starts a stage of its own,
and its output chunks times the voxel ratio must be whole input voxels (checked when the
pipeline is built). The browser engine runs ops on every level of their own grid, so ops
like these run from Python.

