# chunkmirage design

## What it is for

Browsing a computation as if it were a stored dataset: in any viewer, at any scale, before
anyone has computed it. Viewers such as Neuroglancer, BigDataViewer, vizarr and napari, and
libraries such as dask and tensorstore, read chunked array formats over plain HTTP, a
pyramid of levels, chunk by chunk, as the view moves. chunkmirage answers those requests
itself, so the "dataset" can be anything computable per chunk, and does three things for
a viewer that a lazily computed array does not. It serves a pyramid of the computed result:
each level is the ops run on that level's data, a preview of the fine result rather than
its exact downsample (a threshold at a coarse level is not the downsampled fine threshold,
and labels do not agree across levels), except that an op bound to one resolution runs
there and is downsampled from its cached output. It computes the chunks on screen first, at
the level shown, and drops the work nobody waits for any more, so an expensive op under a
panning viewer does the work the viewer still wants. And it speaks every format, so the
viewer is whoever's viewer. Editing the computation while looking follows from the
per-stage cache. It is meant to be the reusable, general form of a trick that had been
re-implemented per project, and it was inspired by the two that showed it:

* [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5): a Flask app that
  synthesises N5 metadata and chunks. It proved the idea: one format, one dataset.
* [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow), which grew out of
  example-virtual-n5: the same trick wrapped around live model inference, with a UI for
  swapping models and post-processing.

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

cellmap-flow is the first consumer, and a realistic test of the design: its server mode
serves a model's predictions through chunkmirage's server, and its browser mode is a page
on the browser engine ([another page on the engine](#another-page-on-the-engine)), the
last card of the [gallery](demos.md), the one not built here. Its own inference server is
gone: each model is an `Op` of its own package, with the model's context as its `halo`,
`slots` for the GPU and a `cache_token` for the weights, its output cached so that changing
the post-processing after it does not run the model again; model configs, warmup, weight
swaps, job launching and the dashboard stayed in cellmap-flow. What chunkmirage gained for
it is stated without models (ops that shave their halo or add axes, ops at one resolution,
`kind`, sources and routes from other packages, resolvers, `--ready-file`, `cache_token`,
`slots`). But chunkmirage is not built around it or any one project.

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

Levels are computed independently, the same ops on each, except from an op that declares
the voxel size it reads (a model trained at one resolution). That one runs on a single
level and the coarser levels are downsampled from its cached output, rather than running
it on data it was not trained for, or once per level. The browser engine keeps the simpler
rule: every op on every level, on its level's grid.

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
chunk it covers is first requested, coarse to fine within the block, and kept in the
server's chunk cache like any stage's chunks. Blocks are sized on their own, not as the output's chunks: a chunk's field reaches
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
same whatever order a client's requests fit blocks in. The same treatment is the engine's
for any op whose blocks disagree where they meet (`Op.overlap`: its blocks cached as a
stage of their own and blended into its chunks with the same weights; see
[pipelines](concepts/pipelines.md#overlapping-blocks-blended-where-they-meet)). Nothing of this is viewer-specific: any client reading chunks
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

**Why worldarray for placing them.** Where a stored array sits (its voxel size, the centre
of voxel 0, units, axes) is read by [worldarray](https://github.com/JaneliaSciComp/worldarray),
one tensorstore reader shared with cellmap-flow, cellmap-analyze and mesh-n-bone, which
had each drifted to their own half-voxel rules. Its specification and its fixture of
metadata layouts are the contract; chunkmirage keeps what only it needs on top: xarray
coordinate arrays and CF packing, the spec's overrides and `kind`
(`sources/tensorstore_source.py`). The browser engine's TypeScript readers place arrays
by the same rules.

**Packaging: uv.** Everything the core needs is on PyPI. cellmap-flow can keep pixi for
CUDA/torch and depend on chunkmirage as a normal PyPI/git dependency.

## Extensibility

Three layers, from most to least structured:

1. **Built-in ops**: pointwise (threshold, cast, scale), filters (gaussian, uniform, dog),
   morphology, connected components with a size filter (`label`), spot detection (`spots`),
   and `contacts` over the channels of a `stack://` source, the first op over several images
   (the stack reads them on one grid; the op drops the channel axis), and `downsample`.
   Planned: distance transform, a `Combine` op taking another pipeline as input. Registration is a source,
   not an op (see below).
2. **Plugins**: subclass `Op`, declare `name`, `halo`, `cache`, register via the
   `chunkmirage.ops` entry point. cellmap-flow's models are one plugin package.
   Params are pydantic fields, so every op ships a JSON schema the UI/MCP can render.
   Source schemes come through `chunkmirage.sources`, and endpoints of a plugin's own
   through `chunkmirage.routes` ([API](reference/api.md#routes-of-your-own)); the app can
   also be mounted inside another one.
3. **Live control**: the REST API takes a `PipelineSpec` (source + list of op specs), so
   a script, a notebook or an agent can drive the viewer; an MCP server as a thin wrapper
   over it (tools: `list_ops`, `set_pipeline`, `neuroglancer_link`) is on the
   [roadmap](roadmap.md).
4. **Code in the spec**: the [`python` op](reference/ops.md#your-own-python-the-python-op)
   carries its source, for what no op does yet, edited live as a Neuroglancer shader is
   (see below).

### Code in the spec: the `python` op

* **Python, not a language for the web.** The browser engine already runs the package's
  Python in Pyodide, so code written once runs in the page and on a server, with numpy,
  scipy and scikit-image in both. Code generated for WebGL or WebGPU would be one more
  language with no scipy of its own, that a server could not run without a GPU runtime;
  WGSL may come later for GPU pointwise ops and filters (the `wgpu` package runs it on a
  server too), as an option beside Python.
* **The code is a parameter.** It is a field of the op like any other, so it is part of
  the digest: an edit is a new URL, cached stages and refetching work as for a slider, and
  a share link or an agent's spec carries it whole. No step registers an op first.
* **Planned without running it.** A pipeline must know an op's halo and output dtype
  before any chunk is computed (to read padded blocks, to write the format's metadata),
  and a server that may not run the code must still read the spec to refuse it. So halo,
  output and sliders are comment lines read as text (`#halo`, `#output`, and `#uicontrol`
  in Neuroglancer's shader syntax, which its users know), and the code is checked by
  parsing it (`ast`), not by calling it.
* **Who may run it.** In the browser engine the code runs in the visitor's own workers, so
  it is always allowed; a link carrying someone else's code shows it and waits for the
  visitor to run it, since Python in a page can fetch and read the site's storage. On a
  server it runs as the server's user, with every file that user can write, so `/api`
  takes it only with `--allow-code`, which brings a generated token: the server listens on
  every interface by default, and on a shared cluster node its `localhost` is every user's.
  Without a token, `/api` refuses changes from other sites' pages (their `Origin`), since
  the CORS that viewers need would let any open tab send them.
* **A model writing it, in the page.** The demos' chat sends the visitor's request to the
  model the visitor picks, with the visitor's own key, from the page, and puts the code it
  writes in the editor. The demos are static pages, so there is no server of ours to hold a
  key, nor one to pay for every visitor's requests; the key stays in the page's memory,
  never in its storage, which a link's Python could read. The chat is
  [Deep Chat](https://deepchat.dev), a web component that needs no framework and connects
  to each provider itself (their APIs differ); the page only adds the layer's context to
  each message and takes the code from each reply. It is in `web/` only, loaded when the
  chat is opened. An agent with a server of its own drives `/api` instead (an MCP surface
  is on the [roadmap](roadmap.md)).

### What stays in the core

The core composes data and serves it; it does not compute. That is the rule for where code
goes, chosen so that nothing is left to judge case by case (is spot detection microscopy
or astronomy? is a ratio of two bands geo?): if it has an algorithm in it, it goes in
`chunkmirage_contrib`, a second package in the same wheel.

| package | holds |
| ------- | ----- |
| `chunkmirage` | the pipeline, cache, demand queue, server, REST API and CLI; the frontends, and `materialize`, which writes a dataset as they serve it (their own metadata and encoded chunks, so a store reads back as the dataset did); readers of stored formats (zarr, N5, precomputed, HDF5); `scene://` and the transforms (the pipeline's own resampler uses them), and that resampler on a GPU (the same code with CuPy for numpy, `gpu.py`, loaded only when a device is set: interpolation is what the core's own sources do, so a faster way to do it stays with them; CuPy rather than PyTorch because its calls are numpy's and scipy's, so one code path serves both and the CPU's tests cover it); the sources that compose other sources (`stack://`, `flip://`, `warp://`); meshes, a frontend's way of serving a surface; and two ops: `downsample`, which the pipeline runs to make the pyramid of an op bound to one resolution, and `python`, code carried in the spec |
| `chunkmirage_contrib` | every other op (filters, thresholds, morphology, labelling, detectors, terrain); the computed sources (`synthetic://`, `register://`, `stitch://`, GeoTIFF reading); the registration solver, stitching and tracking; and, in `web/src/contrib/`, the browser pages built on them |

Contrib reaches the core only through the entry points any other package would use:
`chunkmirage.ops`, `chunkmirage.sources`, `chunkmirage.source_matchers` (files told apart by
their path, a GeoTIFF by its extension) and `chunkmirage.models` (parameter models in
`chunkmirage schema`). No core module imports it, and `tests/test_core_boundary.py` fails if
one does. So the engine stands alone, and the plugin API is exercised by the code shipped
with it: a gap a third party would hit, contrib hits first. The browser engine's Pyodide
workers take contrib's Python by a glob of its files, so a module added there needs no
listing.

Two exceptions, each to avoid moving code twice: `warp://`'s procedural swirl fields stay in
the core until they are a field source of their own (`warp://` takes a field from any
source now, so that is the next step, and the demos' `field=swirl` with it); and
`meshes.py` keeps its `terrain` kind, a few lines of a frontend option.

It is one wheel: `pip install chunkmirage` installs both, and the heavy dependencies stay
extras (`torch` for `register://`, `tifffile` for GeoTIFFs). One bucket, not several: a split
by field (geo, microscopy) would bring back the judgment the rule removes. Should anyone want
the engine without contrib, contrib becomes a distribution of its own, with a
`pyproject.toml` and no change to the code.

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
a level the first asked for goes first. (Taking the latest burst of requests first was tried
and removed: nothing measured it, and a client that stops wanting a block cancels its claim
anyway.) Three run at once, so the GPU fits one while the next ones read, and the rest
cannot flood the network. The page reports the queue as it goes: blocks fitting and waiting
per level, those dropped because nobody waits for them any more, fitted per level, and how
much of what was read came from the cache. Fed the same affine and
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
  [browser/register.html](https://janeliascicomp.github.io/chunkmirage/browser/register.html).
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
  signs open; of the four orientations that do not mirror the image, the best correlated is
  kept. Mirrors are not searched for, since correlation cannot tell handedness on a nearly
  symmetric specimen: on the fly templates a mirror scores exactly as well (0.872) while
  226 µm from the published affine. (A switch to search mirrors instead was tried and
  removed: nothing tells a user when to turn it on.) A mirror image is registered by giving
  its affine. Then it fits the 12 numbers by gradient ascent on the normalized
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
* Pipelines of ops run in the page without a second implementation: `pipeline.html` loads
  Pyodide in a few web workers, writes the package's ops and `chunkmirage.fused` (the code
  a pipeline stage runs, which imports nothing beyond numpy and the ops) into its file
  system, and has them compute each chunk from a padded block the page's one reader
  worker reads in TypeScript. One reader rather than one per worker: each store chunk is
  decoded once for the page, and a piece it keeps is a whole store chunk (the inner chunk of
  a shard), since zarrita decodes all of a chunk to read any of it; a 65 MB sea-temperature
  tile in every worker, or decoded again for each part read, would not fit.
* The page's engine (`engine.ts`) is shared by the Neuroglancer page and a map page, and
  serves each view as OME-Zarr and as GeoZarr: the map page's OpenLayers reads the latter
  as it reads any GeoZarr store, the point being that chunkmirage's output is not tied to
  one viewer. A map client keeps no chunk cache of its own, so the engine keeps computed
  chunks (256 MB), and a slider that changes an op parameter serves the view under a new
  name, which the client fetches afresh while the old chunks stay valid. Porting the ops to TypeScript or WebGPU shaders, the plan before, would have
  meant one more implementation per op and parity tests to keep them equal; Pyodide made
  the Python itself the browser's, at the cost of a 6 s start and single-threaded
  WebAssembly per worker, which the ops' 0.05 to 0.13 s per block easily afford. The
  readers stay in TypeScript (tensorstore has no WebAssembly build), and the gallery
  (`index.html`) lists the demos; see [Demos](demos.md).

### Another page on the engine

A page of another project builds on the engine as the demos do: it clones `web/` at a
commit (nothing of `web/` is published to npm yet, and the Python package's wheel holds
none of it), imports `web/src/engine.ts` and what it needs beside it, and bundles them as
the demos are bundled (Vite; the Pyodide workers import the package's Python from `src/`
as text). What such a page gets, in the order it uses it:

* `Engine.start(views, status, {packages, python})` registers the service worker, starts
  the reader and opens `views`' sources; `Engine.read(view, level, lo, hi)` then reads a
  region of any of them, and `Engine.add(views)` opens more later. Python (Pyodide, about
  6 s and 15 MB) loads only when a view needs it, that is has ops to run, a `synthetic://`
  source or a mesh: a view with no ops is planned on the page from its source alone and
  served from the reader, each chunk the region as read placed whole in the chunk
  (`web/src/asread.ts`), and a view whose chunks the page computes itself needs none
  either. So a page that serves only such views never loads Python; one that will `call()`
  Python passes `python: "always"`, so it loads while the data opens. `ready` resolves once
  every view given to `start` is planned, Python loaded if it was started.
  `Engine.call(fn, args, arrays)` runs one step of a page's work in the next worker (JSON
  arguments, arrays as bytes): the package's own steps (`chunkmirage_contrib.stitching`,
  `tracking`), or the page's own Python, which `start`'s `modules` writes into the workers
  beside the package's (`"pkg/mod.py"` to its text) and `call("pkg.mod:fn", args, arrays)`
  reaches by module path: `fn(args, arrays)` answers JSON, one array (bytes or a numpy
  array) or JSON with arrays (`chunkmirage.calls`, which runs natively too, so the page's
  Python is tested with pytest). A module under `chunkmirage/` or `chunkmirage_contrib/`
  is refused: the package's own files are not a page's to replace. The workers take turns,
  so a call carries its own state; `workers` sets how many start, one at least (one does
  for a page that only calls). cellmap-flow's rank score is such Python, written once and
  run by its server and its page alike.
* `Engine.serve(view, info, chunk, produce)` serves a view whose chunks the page computes
  (a model run by ONNX Runtime Web, say): `produce(level, index)` gives each chunk as zarr
  v3 bytes, the engine queues, shares and drops the work as its own (finest level first,
  chunks no client waits for any more dropped unrun), keeps the chunks a while (256 MB)
  and serves them as OME-Zarr 0.5, zarr v2 and GeoZarr at `Engine.url(view)` and the rest.
* `Engine.serveStages(view, info, chunk, from, stages, padding)` serves a view the page
  computes in stages (`web/src/stages.ts`), the browser's side of the Python pipeline's
  per-stage cache: each stage has a `key` (the page's hash of what it computes: a model's
  weights and parameters), a `halo`, its output `dtype` and the leading axes it adds, and a
  `produce(input, at)` that makes its output block from its input, the stage before's
  output over the block grown by the halo. The first stage reads `from`, an opened view, at
  the level its `inputVoxelSize` names (the model-at-one-resolution case,
  `Op.input_voxel_size`: a model reads 8 nm voxels and writes 16 nm ones, the view declared
  at 16 nm), the box on that grid grown by its halo, padded past the array's edge as the
  pipeline pads (`padding`, edge or zero); a stage may return its block whole or shaved by
  its halo, as an op may. A stage's block is the typed array of the dtype it declares, any
  the engine's table knows (`TYPED` in `web/src/asread.ts`: numpy's names for the integers
  of 8 to 64 bits and the floats of 32 and 64), the 64-bit integers as `BigUint64Array` and
  `BigInt64Array`, whose elements are bigints, so a segmentation's labels past 2^53
  (`np.uint64` as the Python ops write it) reach the chunk whole and are served as zarr's
  `uint64` (`<u8` in v2); the stage cache and the chunk's layout go by bytes, so nothing
  converts them. Such a view has one level: the Python pipeline runs a pinned op at
  its level and downsamples its result for the coarser ones, the engine does not yet, so
  `serveStages` refuses more. Each stage's output is kept under the stage's key with what it
  reads and where (the source view and its padding, the box at its voxel size), not the
  view's name (512 MB), so a view served again with its postprocessing changed finds the
  model's output already made, two views sharing a stage share its outputs, and an output
  asked for twice while on its way is made once; the last stage's
  output is placed whole in the chunk, its added axes first, as the Pyodide workers lay a
  chunk out. `serve` is the one-stage case with the page doing the reading.
* An `onnx` stage (`web/src/onnx.ts`): a model's forward pass as a stage of such a view,
  ONNX Runtime Web on the GPU through its WebGPU execution provider, on WebAssembly (the
  CPU) when the browser has none. `new Onnx({wasmPaths, log})` holds the sessions, one per
  model (`OnnxModel`: a key that is its identity, its bytes when first needed, the tensors
  to feed and read, the session's first by default), shared by the views that use it
  (`retain`, `release`; `reset` drops them all, for a session in a bad state). One call
  into ORT runs at a time on the page, whatever its session, a forward pass or a session
  built or released (`exclusive`, which a page running sessions of its own on the same ORT
  calls too, from the engine's module): ORT Web runs every session in one WebAssembly
  instance, whose WebGPU build waits on the GPU through Asyncify, one suspended call at a
  time, so two calls overlapping never resolve, two models' as much as one's, and a page's
  second model built while its first computes is one such overlap. A pass that hangs holds
  the turn, and only a reload recovers from it; `limits()`
  gives the GPU's `maxBufferSize` and whether it has `shader-f16`, for a page choosing
  among a model's sizes. `onnxStage(onnx, model, {halo, inputVoxelSize, added})` feeds
  each input block as a float32 tensor `[1, C, z, y, x]` and returns the tensor the model
  wrote as the stage's block. Nothing of a model's meaning is in it: what its input is
  normalized to and what its output means are stages of the page's own, before and after,
  and whatever describes its models stays with the page. ORT's WebAssembly runtime (27 MB)
  is served beside the page by the Vite plugin in `web/vite-ort.ts` (`plugins:
  [ortRuntime()]` in the page's Vite config), from the installed package in development
  and copied into the build, so the page loads nothing from a CDN. The demos do not use it:
  `onnxruntime-web` is an optional peer dependency of `web/`, installed by a page that has
  the stage and by nothing else (the demos' builds carry none of it), and a dev dependency,
  so the stage is typed with the rest and run in `npm test` through the WebAssembly backend,
  which runs in Node as in a browser without WebGPU, on two models of a hundred bytes
  (`web/test/onnx.test.ts`, the models written by `web/test/fixtures/make_onnx.py`).
* `Engine.materialize(view, store, options)` writes a region of a served view into a store
  as an OME-Zarr of its own (`web/src/materialize.ts`), to open again later or copy where
  other tools read it: a model's output over a box, computed once in the page and kept.
  The store is anything with `put(path, bytes)`; `directoryStore(dir, root)` makes one of
  a `FileSystemDirectoryHandle` (a folder the visitor picks with
  `showDirectoryPicker({mode: "readwrite"})`, in Chrome and Edge, or the origin's private
  file system, `navigator.storage.getDirectory()`), its subdirectories made as needed and
  each file written whole through `createWritable()`, and `memoryStore()` one of a `Map`,
  for tests and for zipping a small result. The options: the view's `level` (the finest by
  default), a voxel box `lo`, `hi` on it (the whole level by default; clipped to the level),
  `extent` (`"box"`, the default: the array is exactly the box, translated to its first
  voxel, its chunks the served chunk shape counted from `lo`, each stitched from the served
  chunks it overlaps, since a page passing a box expects an array of that box; `"level"`:
  the array is the level's whole shape and the box's chunks are written, rounded out to
  whole chunks, for several boxes written into one store), `format` (`"zarr2"`, OME-Zarr
  0.4, the default, since more tools read it; or `"zarr3"`, OME-Zarr 0.5), `compression`
  (`"gzip"`, the default, or `null`), `skipEmpty` (a chunk of nothing but zeros left out, by
  default), `concurrency`, `signal` (an `AbortSignal`), `onProgress` (`done`, `total`,
  `written`, `bytes` for the array's chunks; `computed`, `needed` for the served chunks) and
  `attributes` (the consumer's provenance, beside the OME metadata). It resolves with
  `{chunks, written, skipped, bytes, aborted}`. The layout is the one the engine serves for
  the view, from one module (`web/src/layout.ts`) for both, with the one level written as
  the group's one dataset at `0` with its own scale and translation (moved to the box's
  first voxel, with the box's extent), so it lands where the view does
  ([formats](concepts/formats.md#a-region-written-to-a-store-browser-engine)). A box that
  does not start on the served chunks' edges is not computed again on the box's own grid:
  the engine keeps no chunk only materialize asked for, so asking a served chunk again for
  each of the box's chunks it overlaps would compute it up to eight times; instead the
  served chunks are each asked for once and copied into the box's chunks they overlap, a
  box chunk written once complete, so about a slab of the box's chunks is held at once. The
  walk puts the axes the box does not split outermost and the split axis with the most of
  the box's chunks next, so the slab is the one across the other split axes, the smallest
  any order of the axes gives (a wide, shallow box walked in C order would hold a slab
  across its two wide axes). `Engine.materializePlan(view, options)` gives that peak for a
  box before it starts (`peakChunks`, `peakBytes`), so a page can warn, or offer to put the
  box on the chunk grid along an axis; should a smaller peak ever matter, computing the
  plane of served chunks two slabs share twice, along the outermost split axis, is the way
  to it, not compressing chunks while they wait or refusing a box. A single level rather
  than the pyramid: the engine downsamples nothing itself, and every level would be the
  model run again on its own grid. The metadata is written first, so a store whose writing
  stopped is still an OME-Zarr, zero where no chunk was written. The
  chunks come from the same producer the routes use, through the same queue, but in the
  background: they wait behind the viewer's, in all its slots but one, and are never dropped
  as abandoned ([caching](concepts/caching.md#order-of-work-and-requests-given-up-on)), so
  browsing goes on while a region is written. An abort starts no more chunks, gives up
  those waiting for their turn, lets the writes under way end and resolves at once with
  `aborted`; a chunk that fails to compute or be written rejects it with the chunk's key
  and why, once the writes under way have ended. This is a first piece of
  [roadmap](roadmap.md) item 1: an explicit write of a box, asked for by the page, not yet
  the cache on disk that browsing fills. The parts that need no engine (the chunks of a
  box, the files, gzip, the stitching, the loop over any producer, the stores) run in
  `npm test`, the result read back by zarrita (`web/test/materialize.test.ts`).
* `Engine.unserve(view)` stops serving a view: its infos, producer and kept chunks are
  forgotten and chunks of it waiting to be computed dropped, so a page that serves a model
  again under a new name (viewers cache by URL, so an edit is a new name) does not keep the
  old one's chunks. `Engine.stop()` terminates the reader and the Pyodide workers, drops
  waiting chunks and forgets every view; a page that opens another source then starts a
  fresh `Engine`, whose URLs carry a new page id, so a viewer fetches afresh.
* The service worker. The engine registers `sw.js` beside the page, by name, so a page
  with routes of its own (an API its page answers, files picked on the machine) bundles its
  own service worker from `web/src/relay.ts`, as `web/src/sw.ts` is: `relay(request,
  {pick, timeout})` asks the open pages (all of them, or those `pick` chooses) for the
  request (an `Ask`, `web/src/types.ts`: its `path`, the URL's pathname alone; its `search`,
  the query string with its `?` or empty, where a page's API reads `?offset=12`; its range,
  method, headers and body), and streams the answer, a chunk's body once computed or an
  event stream's pieces as they come, telling the page when the client gives up;
  `fromStore(request)` is the store cache, with its `stats`, for the chunks only:
  `metadata(url)` says which requests are a store's metadata, which a page's worker leaves
  to the network so that Chrome may ask the visitor about a private-network host
  ([caching](concepts/caching.md#in-the-browser-engine)). On the page, `route(prefix,
  handler)` (`web/src/routes.ts`) answers the asks whose path holds a prefix (the path, not
  the query, so a route matches whatever a request's parameters):
  the engine routes its share of `/virtual/` there, a page adds its own beside it, and a
  page with no route for a path declines it at once, so the worker never waits on a page
  that has nothing for it. One listener for the page rather than one per route, since the
  worker takes a page's first message as its answer.

## Deployment shapes

* **Laptop**: `chunkmirage serve … --port 8000`, open the printed link.
* **Cluster GPU node**: same, with `--public-url` pointing at an SSH tunnel or reverse
  proxy (cellmap-flow's setup).
* **Multi-user / shared cache**: put the cache in a real zarr store on local SSD (planned
  `DiskCache`, opt-in and off by default), so workers share it and the cache doubles as a
  partially materialized output dataset. "Browse to materialize" falls out naturally.
* **Inside an application**: an application mounts the app in its own
  (`Mount("/prefix", create_app(...))`), or starts `chunkmirage serve` with
  `--ready-file` and its plugins installed; see the [API](reference/api.md#mounted-in-another-app).

## Open questions

* Should re-chunking default to the source chunk shape (current) or to a viewer-friendly
  64³? Neuroglancer performs best with ≤ 64³-ish chunks; sources often use 128³.
* Sharded zarr v3 output: worth emitting? Viewers read shards fine; only useful when the
  cache is on disk.
* Auth: `--token` guards `/api/*`; the datasets stay open, because viewers send no
  headers. Access per dataset (signed URLs, say) is still open.
