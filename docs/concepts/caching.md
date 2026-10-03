# Caching

## What is cached

Each pipeline stage can memoize its output chunks. By default:

| stage                           | cached | why                                                     |
| ------------------------------- | ------ | ------------------------------------------------------- |
| stage 0: raw source             | yes    | refetching from disk or S3 is the expensive part        |
| ops with `cache = True`         | yes    | expensive to recompute (inference)                      |
| an op with an input voxel size, and the levels made from it | yes | it runs on one level, and every coarser level is made from its output ([ops at one resolution](pipelines.md#ops-at-one-resolution)) |
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

Separately, tensorstore keeps its own in-RAM cache of *decoded source chunks*
(`--source-cache-gb`, default 0.5 GB), one pool for the whole process: every source it
opens shares one tensorstore context, so the levels of an image, several images, and two
opens of one image (a before and an after dataset, or the images a registration reads
while its output serves them) draw on one budget and share what is decoded. It is what
`register://`'s solves and refined blocks read through, since they read the images
directly rather than through the stage cache above. On a local zlib zarr, reading a region
again from the pool takes 1 to 2 ms against 30 ms decoding it, and a second open of the
store in its own context would decode it again. Opened from Python with `open_source`
alone, sources get no pool (`cache_bytes=0`); pass `cache_bytes` to have one.

## Who caches what

A viewer caches what it shows; chunkmirage caches what it computes from. The two are not
copies of each other:

| cache | holds | whose job |
| ----- | ----- | --------- |
| output chunks | what a client displays | the client's (Neuroglancer keeps its own) |
| inputs | source chunks that many outputs read: overlapping halos, neighbouring chunks, several ops or views of one image | chunkmirage's: no client ever sees them |
| expensive intermediates | inference outputs, solved fields, refined blocks | chunkmirage's |

Outputs are cached only when they are expensive (the `cache` flag above), not because a
particular client may or may not keep them, so nothing depends on which client asks: a
dask array reading each chunk once is served as well as a viewer.

## In the browser engine

The [browser page](../design.md#client-side-browser-roadmap) keeps the same three, in the
browser:

* **Fetched source bytes**, in its service worker (512 MB, still compressed): every read of
  the images' stores, by the page, its chunk workers and the viewer alike, goes through it,
  and requests for a chunk already on its way wait for that fetch instead of starting
  another. At most 24 fetches run at once; with more pending, Chrome refuses some.
* **Decoded pieces**, aligned to the store's own chunks, in each context that reads (the
  page for its block fits, each chunk worker for resampling; 128 to 192 MB each), so a store
  chunk is decoded once per context however many regions overlap it.
* **Fitted blocks** of `refine`d fields, on the page.

Before the first two, zooming the EASI-FISH pair's link to full resolution fetched 11.7 GB
for 0.9 GB of distinct store chunks, each read 13 times on average, and the network, not the
GPU, set the pace. The same zoom on the same computer (an RTX 2080 Ti) since:

| | separate caches per context | shared |
| --- | --- | --- |
| fetched | 11.7 GB | 1.1 GB |
| a block's reads (median) | 37 s fixed, 52 s moving | 0.5 s, 0.9 s |
| the zoomed view complete | 192 s | 98 s |

The GPU is now what the view waits for (busy 90% of the time). Blending refined blocks, added since, fits a ring of blocks more around the view: the same zoom now completes in 115 s.

## Keys

Every cache entry is keyed by `(stage_hash, chunk_index)` where `stage_hash` folds in:

* the source identity (path or URL) and scale level; for a `scene://` source, also the
  whole transformation chain, so an edited registration gets fresh keys,
* the output chunk shape,
* the spec of every op up to and including this stage, with each op's `cache_token()` if
  it has one (below).

Consequences:

* Editing op *k* changes the hashes of stages *k..n* only. Upstream entries stay valid with
  no explicit invalidation.
* Two pipelines with the same prefix share entries, including a second dataset branching
  off the same model output.
* A pipeline's overall digest appears in served URLs (`/{name}/@{digest}/{format}`) so
  viewers, which cache by URL, refetch after an edit.

### State outside the parameters

An op's output can depend on more than its parameters: model weights replaced in place, a
lookup file that is rewritten. Such an op returns a string from `cache_token()` that
changes when that state does, such as the file's modification time or a version number:

```python
class Model(Op):
    name = "model"
    cache = True
    weights: str

    def cache_token(self):
        return str(os.path.getmtime(self.weights))
```

The token is part of the op's identity, so it enters the stage hash and the digest in
served URLs. A pipeline reads it when it is built, so a running server picks up a change
when the dataset is rebuilt: `POST /api/datasets/{name}/refresh`, or
`DatasetRegistry.refresh(name)` (every dataset without a name). Only datasets whose digest
moved change: they get new keys, new links and a `change` event, and viewers refetch. The
old entries are evicted as the cache fills. Ops without a token keep the digests they had.

## Why not cache every stage?

The cache has a fixed budget and every cached stage competes for it. Caching all stages of
a five-stage pipeline stores five copies per chunk and evicts the valuable ones sooner;
intermediate float32 stages are also four times the size of uint8 raw. The rule: cache a
stage when recomputing it costs more than storing it. Raw I/O and inference qualify; a
threshold that takes a millisecond does not.

The `cache` flag is a heuristic set by the op author, and a pipeline overrides it per op
with a `cache` key in the op's spec: `{"op": "gaussian", "sigma": 4, "cache": true}`, or
`--op gaussian:sigma=4,cache=true`. Editing an op after it then reruns only the stages
after the cached one. The flag is not part of the op's identity, since it does not change
what the op computes; `GET /api/datasets/{name}` reports each op's setting as `cached`.
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
with a few slots (three for `register://`'s refined blocks, whose fits share the GPU; an
op's `slots` for an op that declares them, a model on a GPU say, in a queue named `op
<name>`): a job is shared by every request that needs it, dropped unrun once none of them waits any
more, and kept if it had started. Among waiting jobs the finest level goes first, then the
first asked for. A request waiting on
queued work gives its compute slot up meanwhile, so requests that need nothing expensive
never wait behind ones that do. Work asked for from Python, with no request behind it, is
never dropped. Ten requests for
full-resolution chunks of the fly templates' refined registration, each given up on after
half a second, had all their block fits dropped; a patient request for another chunk was
served in 8.5 s. `GET /api/queue` reports what runs, waits and was dropped, per level. The
browser page keeps the same rules (`web/src/demand.ts`); it learns of a given-up request
from its streamed reply being cancelled, a service worker being told of nothing else.

## Inspecting and clearing

```bash
curl localhost:8000/api/cache            # {"entries":..,"bytes":..,"hits":..,"misses":..}
curl -X DELETE localhost:8000/api/cache  # drop everything
```
