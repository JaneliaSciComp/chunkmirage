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

Internally, stage 0 is the raw source re-chunked to the output chunk shape, and stage *k*
is a `ChunkedSource` whose `compute_chunk(index)`:

1. computes the output chunk's box;
2. pads it by the op's **halo**;
3. reads that padded box from stage *k-1* via `read_padded` (out-of-volume voxels are 0);
4. calls `op.apply(block)`;
5. crops the halo off and casts to the op's declared output dtype.

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
| `output_dtype(dtype)`  | result dtype; default unchanged                                         |
| `apply(block)`         | the computation; must return an array of the same spatial shape        |

Ops are discovered through the `chunkmirage.ops` entry point, so plugins ship as ordinary
packages. See [Contributing](../contributing.md#adding-an-op) for a template and the
[ops reference](../reference/ops.md) for what is built in.

## Chaining and branching

Chaining is the `ops` list: cellmap-flow's *model then post-processors* is one inference op
with `cache=True` followed by its post-processors.

Branching (one model output served through several post-processors as separate layers) is
not expressible in a single spec yet. Define one pipeline per branch: because cache keys are
prefix hashes, the branches share the cached model output automatically. A DAG spec with
named stages is on the [roadmap](../roadmap.md).

## Re-chunking

`chunk_shape` sets the *output* chunk shape independently of the source. Viewers prefer
around 64³; sources often use 128³ or larger; inference block sizes differ again. Stage 0
reads whatever source chunks cover the requested output chunk.

## Caveat: scale levels

Ops run per level with the same parameters. That is right for thresholding and filters in
voxel units, and wrong for models trained at a specific resolution. Until per-op level
declarations land (roadmap item 2), serve only the levels your op is valid for, or apply it
to a single-level source.
