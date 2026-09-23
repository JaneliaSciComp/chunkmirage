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

1. **Built-in ops**: pointwise (threshold, cast, scale), filters (gaussian, uniform).
   Planned: label ops (connected components, size filter, relabel), morphology, distance
   transform, on-the-fly downsampling, `Resample` (affine / displacement field, the
   registration use case), `Combine` (multi-source: mask, difference, blend).
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
