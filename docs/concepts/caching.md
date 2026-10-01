# Caching

## What is cached

Each pipeline stage can memoize its output chunks. By default:

| stage                           | cached | why                                                     |
| ------------------------------- | ------ | ------------------------------------------------------- |
| stage 0: raw source             | yes    | refetching from disk or S3 is the expensive part        |
| ops with `cache = True`         | yes    | expensive to recompute (inference)                      |
| ops with `cache = False`        | no     | fused with their neighbours into one stage and recomputed from the cached upstream stage |
| encoded bytes (gzip, blosc, ...)| no     | encoding is fast; caching arrays serves all formats     |

So if you threshold, look, and threshold again with a different level, the raw chunk is
read **once**. The threshold stage's hash changes, the raw stage's does not, and the next
request hits the cache. `tests/test_pipeline.py::test_stage_cache_keys_isolate_downstream_edits`
asserts exactly this.

Set `cache_source=False` in the spec to disable raw caching (for example when the source is
a fast local NVMe and memory is scarce).

## Fusion, or why uncached ops are cheap

Consecutive uncached ops are *fused*: one output chunk reads the upstream box once, padded
by the sum of their halos, and runs the ops back to back. A four-op chain with halos
5 + 0 + 5 + 8 on 64³ chunks reads a 100³ box, about four raw chunks. Giving each op its own
cached chunk grid instead would pull 343 raw chunks for the first output chunk, because
each halo widens the footprint at every stage. So `cache=True` is a deliberate cut in the
chain for stages worth keeping (inference), not a speed knob to sprinkle on filters.

## Where it lives

In RAM, inside the server process: a Python dict of decoded numpy chunk arrays with
byte-bounded LRU eviction ([`chunkmirage.cache.LRUCache`](../reference/python.md)).
Default budget is 2 GB (`--cache-gb`). It disappears on exit, and each uvicorn worker has
its own copy. A disk-backed cache that persists and is shared across workers is on the
[roadmap](../roadmap.md).

Separately, tensorstore keeps its own in-RAM cache of *compressed source bytes*
(`--source-cache-gb`, default 0.5 GB). It is mostly redundant with stage 0 for a single
pipeline and matters when several pipelines read the same source with different chunk
shapes.

## Keys

Every cache entry is keyed by `(stage_hash, chunk_index)` where `stage_hash` folds in:

* the source identity (path or URL) and scale level; for a `scene://` source, also the
  whole transformation chain, so an edited registration gets fresh keys,
* the output chunk shape,
* the spec of every op up to and including this stage.

Consequences:

* Editing op *k* changes the hashes of stages *k..n* only. Upstream entries stay valid with
  no explicit invalidation.
* Two pipelines with the same prefix share entries, including a second dataset branching
  off the same model output.
* A pipeline's overall digest appears in served URLs (`/{name}/@{digest}/{format}`) so
  viewers, which cache by URL, refetch after an edit.

## Why not cache every stage?

The cache has a fixed budget and every cached stage competes for it. Caching all stages of
a five-stage pipeline stores five copies per chunk and evicts the valuable ones sooner;
intermediate float32 stages are also four times the size of uint8 raw. The rule: cache a
stage when recomputing it costs more than storing it. Raw I/O and inference qualify; a
threshold that takes a millisecond does not.

The `cache` flag is a heuristic set by the op author and can be overridden per pipeline.
Adaptive caching based on measured compute time is on the roadmap.

## Concurrent requests for the same chunk

Two requests for the same chunk that arrive while it is being computed share one
computation (the second waits for the first, then reads the cache). Viewers retry and
re-request aggressively, so without this a slow chunk would be computed several times.

## Order of work, and requests given up on

Whatever a pipeline computes, the server bounds the work, shares it, and drops what clients
stop waiting for (`chunkmirage.demand`); nothing of it is specific to one viewer, op or
source. At most `--threads` requests compute at once; the rest wait their turn on a plain
semaphore, in no particular order. Every chunk request holds a claim on the work it needs,
cancelled when its client disconnects; a request whose client left before its turn came
costs nothing. Of the clients that read chunkmirage, Neuroglancer is the one that aborts
requests, and it does so when a chunk it still wants needs a download slot (100 at once by
default) held by one that scrolled out of view, so the claim earns its keep on expensive
chunks; Fiji, napari, webKnossos, dask and tensorstore never abort, they simply stop
asking, and for them the bound, the shared computation above and the cache are what keep
the server responsive. Expensive work inside a request goes through a queue of its own
with a few slots (`demand.Queue`; a GPU-bound one has as many as the GPU fits at once): a
job is shared by every request that needs it, dropped unrun once none of them waits any
more, and kept if it had started. Among waiting jobs the finest level goes first, then the
latest burst of requests, then the client's own order within it. A request waiting on
queued work gives its compute slot up meanwhile, so requests that need nothing expensive
never wait behind ones that do. Work asked for from Python, with no request behind it, is
never dropped. `GET /api/queue` reports what runs, waits and was dropped, per level.

## Inspecting and clearing

```bash
curl localhost:8000/api/cache            # {"entries":..,"bytes":..,"hits":..,"misses":..}
curl -X DELETE localhost:8000/api/cache  # drop everything
```
