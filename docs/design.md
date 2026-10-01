# chunkmirage design

## What it is for

Viewers such as Neuroglancer, BigDataViewer, vizarr and napari, and libraries such as
dask and tensorstore, can read chunked array formats over plain HTTP. chunkmirage answers
those HTTP requests itself, so the "dataset" can be anything computable per chunk. It is
meant to be the reusable, general form of a trick that has so far been re-implemented per
project. Two existing examples of that pattern:

* [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5): a Flask app that
  synthesises N5 metadata and chunks. Proves the idea, single format, single dataset.
* [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow): the same trick wrapped
  around live model inference, with a UI for swapping models and post-processing.

chunkmirage factors out the part both share and generalises it to any format, source and
computation:

| concern            | example-virtual-n5 | cellmap-flow            | chunkmirage                                   |
| ------------------ | ------------------ | ----------------------- | --------------------------------------------- |
| served formats     | N5                 | N5                      | N5, Zarr v2, Zarr v3, precomputed, at once     |
| source formats     | none (synthetic)   | zarr/n5 via tensorstore | zarr v2/v3, n5, precomputed, hdf5; file/s3/gcs/http |
| processing         | hard-coded         | inference + post-proc   | pipeline of pluggable ops with halos          |
| caching            | none               | per request             | per-stage LRU keyed by pipeline hash          |
| reconfiguration    | restart            | custom UI/API           | REST (and MCP) with cache-busting URLs        |
| deployment         | Flask              | Flask + gunicorn        | ASGI (Starlette) library + CLI                |

Projects like cellmap-flow are natural consumers (it could shrink to "a set of inference
ops plus a UI" on top of chunkmirage), and serve as a realistic test of the design, but
chunkmirage is not built around any one of them.

### Where it does *not* add value

Neuroglancer already runs GLSL shaders per voxel on the GPU. Pointwise operations on a
single source (thresholding, windowing, colormaps, channel mixing) are better done there,
client side, with zero latency. chunkmirage earns its keep when the op is:

* **non-local**: filters, morphology, connected components, distance transforms, meshes;
* **learned**: model inference (cellmap-flow's case), where GPU + weights live server-side;
* **geometric**: resampling under affine or deformation-field registration, on-the-fly
  downsampling for sources lacking a pyramid, reslicing;
* **multi-source**: masking one volume by another, differencing two conditions;
* **a format bridge**: exposing HDF5, a TIFF stack, or a proprietary reader as zarr to any
  viewer, or exposing zarr v3 to a viewer that only speaks N5.

Thresholding is still the right *demo* because it is instantly legible, and combined with a
cached expensive upstream stage (inference) it shows the caching story: change the level,
nothing upstream recomputes.

## Architecture

```
                 ┌────────────────────────────────────────────────────────────┐
  HTTP request   │ server.py (Starlette)                                      │
 ───────────────►│  /{name}/{fmt}/…  → Frontend.resolve → Metadata | ChunkReq │
                 │  /api/…           → DatasetRegistry (live edit)            │
                 └───────────────┬────────────────────────────────────────────┘
                                 │ pipeline.chunk(level, index)
                 ┌───────────────▼────────────────────────────────────────────┐
                 │ Pipeline (per scale level)                                 │
                 │  ChunkedSource(raw) ─► ChunkedSource(op₁) ─► … ─► op_n     │
                 │       ▲ cache            ▲ cache if op.cache               │
                 │  read_padded(box + halo) between every stage               │
                 └───────────────┬────────────────────────────────────────────┘
                                 │ Source.read(box)
                 ┌───────────────▼────────────────────────────────────────────┐
                 │ Sources: TensorStoreSource (zarr2/3, n5, precomputed over  │
                 │ file/s3/gcs/http, own byte cache), HDF5Source (h5py)       │
                 └────────────────────────────────────────────────────────────┘
```

### Core abstraction: every stage is a chunked, cacheable array

`ChunkedSource` wraps a `compute_chunk(index)` function and an LRU. `read(box)` gathers
covering chunks and slices. The raw source is stage 0 (re-chunked to the output chunk
shape). Each op becomes stage *k*, whose `compute_chunk` reads the halo-padded box from
stage *k-1*, applies the op, and crops. This one mechanism gives:

* **raw caching** (never re-download a chunk while the user fiddles downstream),
* **stage caching** (cache inference output; threshold it for free),
* **halos** (neighbourhood ops read padded regions from the previous stage, which itself
  serves from its cache where possible),
* **re-chunking** (output chunk shape independent of source chunk shape; useful because
  inference block sizes and viewer-friendly chunk sizes differ).

Cache keys are `(stage_hash, chunk_index)`; `stage_hash` folds in the source identity,
chunk shape, and every op spec up to that stage. Editing op *k* changes hashes for stages
≥ *k* only, so upstream entries stay valid without any explicit invalidation logic.
This is the same idea as tensorstore's `virtual_chunked` driver, kept in numpy so the ops
stay trivially writable.

### Viewer cache busting

Viewers cache chunks by URL. The API hands out source URLs of the form
`/{name}/@{digest}/{fmt}` where `digest` is the pipeline hash. After an edit the digest
changes, the viewer sees a new URL, and refetches. The un-versioned path also works.

### Coordinate transformations and registration

Viewers apply affine transforms themselves, but none applies a displacement field (as of
OME-Zarr 0.6, where they became standard), and before 0.6 every registration tool stored
warps in its own format. So a registration is read into one small internal model,
`chunkmirage.transforms` (affine, displacement and coordinate fields, sequence,
byDimension, bijection), by one reader per format; `chunkmirage.ngff` reads OME-Zarr 0.6,
and bigstream, BigWarp or ITK readers can be added without touching the resampler.
A field only has to answer `sample(points)`, so procedural fields fit too: `warp://`
sources twist an image through swirls computed from coordinates, which makes a
deformation you can change continuously without storing anything, or sweep along a `t`
axis of frames that a viewer plays without refetching anything it has already shown.

The resampler is a **source** (`scene://`), not an op, for three reasons: the output grid
belongs to another image, not to the input; each output level should read a different
source level (chosen through the transform); and the region an output chunk needs is the
exact bounding box of its transformed voxel centres, which no fixed halo describes. As a
source it is stage 0 of the pipeline, so its chunks are cached like raw chunks and every
op downstream sees the registered image. The chain for each level is simplified once
(runs of linear pieces fold into one matrix), so a typical grid → field → affine → source
index chain costs two small matrix products and one field lookup per voxel.

Inverses are taken only in closed form or from a stored `bijection`, unless the URL asks
for `inverse=approx`; an estimated inverse of a folding field would silently show wrong
data, so unconvergent points are left empty instead.

Solving a registration is a source too (`register://`): opening it fits a displacement
field on a GPU from a few coarse levels, keeps the field in memory and serves every level
through the same resampler. The split follows the costs. The field is global (any output
chunk may depend on all of it) but cheap to fit at coarse levels and small to keep, while
resampling full resolution is expensive and local, which is exactly what chunk-by-chunk
serving does lazily. So the fit happens once, when the source opens, and nothing of the
registered volume is ever written. That split holds while the field can be coarse. Where
it cannot, `refine=` moves the fit into the on-demand path too: the field of each finer
level becomes a chunked source of its own, a lattice `grid` voxels apart in blocks of
`block` voxels, fitted from the solved field over the block plus a `halo` of context when a
chunk it covers is first requested, coarse to fine within the block, and cached like any
stage. Blocks are sized on their own, not as the output's chunks: a chunk's field reaches
into the lattice beyond it, so a thin chunk straddles two thin blocks, and each block pays
for its context and correlation windows on every side. On the EASI-FISH pair, blocks of
64×128×128 voxels (the default) align full-resolution chunks better than blocks of
16×128×128 (correlation 0.851 against 0.829 on one), since they are deep enough to fit
coarse to fine, and in Python they take half the GPU time. (Fitting each level's blocks from the level above's instead was tried first: on a
pyramid whose z axis is never downsampled, one full-resolution chunk pulled in 48 blocks
across three levels, most of them for context, and a minute of GPU time.) The requests then
schedule the registration, so its cost follows what is looked at rather than the volume,
and a block can afford what a whole-image fit cannot (a finer lattice, more iterations).
Blocks fitted separately disagree where they meet, and kept side by side they made the
field step at their edges (up to 0.9 µm per voxel on the EASI-FISH pair, three times the
steepest 1% inside a block). So each block keeps its fit over its context too, and
neighbouring blocks are blended across that overlap, as bigstream's distributed pipeline
does, with weights that are 1 over a block's own points and fall off linearly into its
context; the field then changes no faster across block edges than inside them. Blending
the fits, rather than fitting each block against its neighbours' results, keeps every
block's fit a function of the solved field and the images alone, so the result is the
same whatever order a client's requests fit blocks in. Nothing of this is viewer-specific: any client reading chunks
drives it. Measured on two rounds of a public EASI-FISH fly brain, the blocks raise the
correlation of full-resolution chunks well above the coarse field's, the more the wider
their window (see [register sources](concepts/formats.md#register-sources-deformable-registration-on-a-gpu)).
PyTorch supplies autograd and grid sampling, as an
optional dependency; the objective is local cross-correlation with flat windows left out,
because damping them with a constant instead rewards warps that add contrast (the tests
catch that). Adam is written out rather than taken from `torch.optim`, whose import loads
`torch._dynamo`: about 9 s from a network file system.

## Language and stack

**Server: Python.** The whole point is that scientists write ops in numpy/torch/scipy;
that ecosystem, plus the tensorstore binding, decides it. The per-request Python overhead
(~0.1 ms routing) is negligible next to a chunk read, a codec, or a network round trip.
The hot paths (tensorstore I/O, numcodecs/blosc/gzip, numpy kernels, torch) all release
the GIL, so one process serves many chunks concurrently from a threadpool. Multiple
uvicorn workers scale further at the cost of per-process caches.

**Why not Rust for the server.** A Rust server (axum + `zarrs`) would win on pure
pass-through/transcoding throughput, but the interesting work is user code, and embedding
Python for it would recreate the same GIL story with more moving parts. If a pure
transcoding proxy ever becomes a bottleneck, the format layer (frontends/codecs) is the
natural piece to reimplement, ideally in a crate compiled both to a Python extension and
to WebAssembly (see below). `zarrs` already targets both.

**Why tensorstore for sources.** One API for zarr v2, zarr v3, n5 and precomputed across
file, S3, GCS and HTTP; async C++ I/O; a built-in byte cache; and future `virtual_chunked`
interop. It does not read HDF5; h5py covers that (serialised behind a lock, since HDF5 is
not thread-safe).

**Packaging: uv.** Everything the core needs is on PyPI. cellmap-flow can keep pixi for
CUDA/torch and depend on chunkmirage as a normal PyPI/git dependency.

## Extensibility

Three layers, from most to least structured:

1. **Built-in ops**: pointwise (threshold, cast, scale), filters (gaussian, uniform, dog),
   morphology, connected components with a size filter (`label`), spot detection (`spots`),
   and `contacts` over the channels of a `stack://` source, the first op over several images
   (the stack reads them on one grid; the op drops the channel axis). Planned: distance transform, on-the-fly
   downsampling, a `Combine` op taking another pipeline as input. Registration is a source,
   not an op (see below).
2. **Plugins**: subclass `Op`, declare `name`, `halo`, `cache`, register via the
   `chunkmirage.ops` entry point. cellmap-flow's models become one plugin package.
   Params are pydantic fields, so every op ships a JSON schema the UI/MCP can render.
3. **Live control**: the REST API takes a `PipelineSpec` (source + list of op specs). An
   MCP server is a thin wrapper over it (tools: `list_ops`, `set_pipeline`,
   `neuroglancer_link`, `cache_stats`), so an LLM agent can drive the viewer. Arbitrary
   code (a `PythonOp` taking source text) is possible but off by default; it is only
   acceptable on a single-user machine.

## Client-side / browser roadmap

Fully client-side is feasible and would make a compelling hosted demo:

* Host a static page containing Neuroglancer (or vizarr) **plus a Service Worker** on the
  same origin. The service worker intercepts `/virtual/…` requests, fetches real chunks
  from any CORS-enabled zarr/n5/precomputed URL (via `zarrita.js` or tensorstore-style
  key parsing), decodes with WASM codecs (`numcodecs.js`), runs the pipeline, and responds
  with **precomputed `raw`** chunks (literally the little-endian bytes; encoding is free).
* Pointwise and filter ops in **WebGPU compute shaders**; inference via **ONNX Runtime
  Web (WebGPU)**; both share the JSON `PipelineSpec` with the Python server so the same
  spec runs on either side.
* Same-origin is required because service workers only see requests from pages in their
  scope; a hosted `neuroglancer-demo.appspot.com` cannot be pointed at an in-browser
  server. Hosting our own Neuroglancer build is fine and commonly done.
* A shared Rust crate (chunk key parsing, zarr/n5/precomputed codecs) compiled to WASM and
  to a Python extension would keep the two implementations from drifting. Not needed to
  start.
* It lives in this repo as `web/`, a TypeScript project (Vite) next to the Python package,
  because the two share one definition: the Pydantic models (`PipelineSpec`, each op's
  parameters, `RegisterParams`) export a JSON Schema (`chunkmirage schema`), and `web/`
  generates its TypeScript types and form defaults from the committed copy. A model change
  and the page it breaks then land in one pull request: `tests/test_schema.py` fails until
  the copy is regenerated, and the `web` CI job until the types are. The Python package
  stays standalone all the same, since its wheel holds only `src/chunkmirage`. Demos that
  span several projects, and plugins with ops for both engines, would be repos of their
  own, built on the published packages.
* The first piece is registration: `web/register.html` reads two OME-Zarr
  images from their URLs with `zarrita.js` (range requests into sharded stores, zstd and
  blosc through numcodecs) and fits the same field as `register://` in WebGPU compute
  shaders, with the gradients written out by hand (WGSL has no autograd, and no float
  atomics, so each control point gathers its voxels' gradients). The local correlation's
  window sums are running sums along each axis in the shaders (one thread per line), so
  a wider window costs nothing extra and the solve is about twice as fast as with a
  per-voxel box filter. PyTorch keeps cuDNN's separable box filter: a prefix-sum version
  measured 4x slower at the default window (68 vs 17 ms per pooling on a 2^25-voxel
  level) and only wins past windows of about 60. Fed the same data, its
  field matches the PyTorch solver's to about 1% (a median 0.25 µm on a gut whose field
  moves tissue by 21 µm), and so do its scores. It then shows before and after in
  Neuroglancer hosted on the page's origin: the page's service worker hands the viewer's
  requests for the registered volume to the page, whose web workers resample the moving
  image through the field as `scene://` does, reading it from its URL through a cache of
  decoded blocks. The viewer caches what it shows and the page what it reads (every read of
  the stores goes through one cache in the service worker, shared by the page, its workers
  and the viewer; see [caching](concepts/caching.md#in-the-browser-engine)); the page only
  raises the viewer's memory limits to twice their defaults, since a chunk the viewer drops
  and asks for again costs a read of the moving image and a resample. Its 3D maximum projections
  are a checkbox, on unless the link says `3d=0`: volume rendering asks for chunks across
  the whole visible volume, which on a whole-organ image keeps the workers busy, so a slow
  computer is better off without. It asks for coarse levels only until the view is zoomed
  in, so it does not set off `refine`'s block fits across the organ (none in 150 s on the
  EASI-FISH pair). `refine` runs in the page too: a chunk worker that needs
  the field of a refined level asks the page for the window of that level's lattice its
  chunk touches, and the page fits the blocks it covers on its GPU (`blocks.ts`, one at a
  time, as `register.py` does), keeps them, and sends the window back; the after and field
  views share the blocks. The fits wait in one queue, which knows only the requests, so it
serves any client the same way, with the rules the Python server applies to all its work
([caching](concepts/caching.md#order-of-work-and-requests-given-up-on)). A client that stops waiting for a chunk aborts its request;
a service worker is not told of that (Chrome 148), but it answers each chunk request at
once with a head and a body it streams later, and a reply whose request was aborted has its
stream cancelled, which it does see. So every request holds a claim on the blocks it needs,
a cancelled request releases its claims, and a waiting block no request claims any more is
dropped without being fitted; one already fitting finishes and is kept. Zooming the
EASI-FISH pair to full resolution and then panning away, Neuroglancer gave up on 430 of the
973 requests it made for the refined views, and 29 waiting blocks were dropped. Among the
blocks still wanted, the finest level goes first: a viewer asks for coarser levels of the
same place to show while the fine chunks compute, and those placeholders are worth fitting
only once nothing finer waits (on the EASI-FISH zoom the full-resolution chunks on screen
were answered after a median 103 s instead of 166 s, the same blocks fitted in all). Within
a level the latest burst of requests goes first, in the order the client sent them, which is
its own priority; a block asked for again joins the current burst. Three run at once, so the
GPU fits one while the next ones read, and the rest cannot flood the network. The page reports the queue as it goes: blocks fitting and waiting per level, those dropped
  because nobody waits for them any more, fitted per level, and how much of what was read
  came from the cache. Fed the same affine and
  levels (the page's "same in Python" command carries them), its refined
  full-resolution chunks of the fly templates match Python's to a median 0.005 µm in the
  field (0.03 µm at the 95th percentile), as close as PyTorch on CUDA and on the CPU come
  to each other, while the refinement itself moves 60% of the tissue's voxels by more
  than a grey level. On emulated WebGPU (SwiftShader) a block takes about 13 s; a real GPU
  is many times faster. A coarsest level
  larger than the page's GPU budget (2^22 voxels, where a pyramid stops early or stays
  deep in z) is halved on the page before the solve, as a stored pyramid would be (the
  voxels' means, then normalized). So all of `register://` runs client side. WebGPU and service workers
  need a secure page, and a service worker will not run on a certificate that was only
  clicked through: `web/serve.py` serves the build over https with the
  self-signed certificate (trusted once in the system; it is made to the rules macOS and
  browsers apply even then, at most 398 days and for server authentication) or over plain
  http for `localhost`, and relays the standard Neuroglancer client under `/ng/`. The docs
  site needs neither: the docs workflow builds Neuroglancer from Google's source at a
  pinned tag (cached, so about 15 s the first time) and publishes it at `browser/ng/`
  next to the page, at
  [browser/register.html](https://yuriyzubov.github.io/chunkmirage/browser/register.html).
  Nothing of Neuroglancer is kept in this repo. With no images in its link the page opens
  with an example, two fly brain templates (JRC2018F and FCWB) as they are stored:
  `web/scripts/fetch_example.py` copies them at deploy time from the OME-NGFF
  transformation examples, whose bucket allows no CORS and uses a draft 0.6 layout,
  rewriting only the metadata as 0.5, and the site serves them next to the page. So the
  example needs no CORS, no VPN and no local network access. It can also start from the
  affine published with them (the affine part of the examples' JRC2018F-to-FCWB transform;
  `?start=published`).
* With the affine left empty the page finds one before the field (`affine.ts`, on the CPU:
  a few hundred thousand voxels are enough). It matches the two images' intensity
  moments, centre to centre and principal axis to principal axis, which leaves the axes'
  signs open; of the four orientations that do not mirror the image (with the page's
  "mirrored" box, `mirrored=true` in Python, of the four that do), the best correlated is
  kept. Handedness is left to the user because correlation cannot tell it on a nearly
  symmetric specimen: on the fly templates a mirror scores exactly as well (0.872) while
  226 µm from the published affine. Then it fits the 12 numbers by gradient ascent on the normalized
  cross-correlation, at two resolutions. On the fly templates, which start 58 µm apart
  (mean distance from where the published affine puts each voxel), that takes the
  correlation from 0.14 to 0.84 (moments) and 0.87 (fit), and ends 2 µm from the published
  affine (the voxels are 2.5 µm), in about 5 s. The moments assume both images show the
  same whole object; a crop of one would need an affine given.
* The page shows the same registration as a `chunkmirage serve 'register://…'` command,
  built through the generated `RegisterParams` type, with the found affine and the solved
  levels filled in after a run. Solved from that command, Python's field matches the
  page's to 0.01 µm (median; 0.04 µm at the 95th percentile) on the fly templates, whose
  field moves tissue by 7 µm (median). Python's `register://` finds the same affine with
  `affine=auto` (`registration.find_affine`, the page's search ported: PyTorch's autograd
  supplies the gradient the page writes out by hand).

## Deployment shapes

* **Laptop**: `chunkmirage serve … --port 8000`, open the printed link.
* **Cluster GPU node**: same, with `--public-url` pointing at an SSH tunnel or reverse
  proxy (cellmap-flow's setup).
* **Multi-user / shared cache**: put the cache in a real zarr store on local SSD (planned
  `DiskCache`), so workers share it and the cache doubles as a partially materialized
  output dataset. "Browse to materialize" falls out naturally.

## Open questions

* Should re-chunking default to the source chunk shape (current) or to a viewer-friendly
  64³? Neuroglancer performs best with ≤ 64³-ish chunks; sources often use 128³.
* Sharded zarr v3 output: worth emitting? Viewers read shards fine; only useful when the
  cache is on disk.
* Auth: currently none. A bearer token on `/api/*` is the minimum before exposing beyond
  a tunnel.
