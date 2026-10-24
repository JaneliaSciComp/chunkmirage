# Changelog

Notable changes per release. Versions follow the rules in
[contributing](docs/contributing.md#plugin-api-and-versioning): the plugin API is stable,
and a release that breaks it says so here.

## Unreleased

### Serving

- `materialize` and the GeoTIFF source open their stores on the context chunkmirage's
  other stores share (`wa.shared_context(0, threads=0)`), where they had tensorstore's
  default one: worldarray's watchdog now covers them, so a write or tile read stranded by
  tensorstore's lost wakeup runs within a second instead of never, and `materialize`
  writes without fsync or OS file locks, about 10 ms less per chunk on NFS. A chunk is
  still written whole, to a temporary file renamed into place.
- The lock moves to worldarray 0.9.0; the pin stays `>=0.8,<1`. The context chunkmirage's
  stores share is unchanged under 0.9.

## 0.1.0a5 (2026-10-23)

A dependency release, on `worldarray>=0.8,<1`: packages beside chunkmirage can now move to
a new worldarray without a chunkmirage release, and CI tests the newest worldarray every
Monday ([contributing](docs/contributing.md)). Nothing else changes.

### Serving

- worldarray 0.8 times out the open that creates an array, and the context chunkmirage's
  stores share (`threads=0`, pools still tensorstore's own) writes without fsync or OS file
  locks; chunkmirage only reads through it.

## 0.1.0a4 (2026-10-23)

A dependency release, on `worldarray>=0.7,<0.8`, so chunkmirage installs beside packages
that need worldarray 0.7 (cellmap-flow). Nothing else changes.

### Serving

- Opening a store and reading its metadata documents time out and retry under worldarray's
  `Retry` (5 s times the attempt, ten attempts), as chunk reads already did: a source that
  accepts a connection and never answers raises `TimeoutError` instead of hanging the
  server. Listings are not covered.

## 0.1.0a3 (2026-10-23)

The third alpha, on `worldarray>=0.6,<0.7`. The plugin API only grew
(`Frontend.chunk_request`). What moves: an N5 level with no `offset` has its corner at the
origin (under Sources), and the zarr frontends' OME multiscales are worldarray's (under
Serving).

### Serving

- A precomputed scale with a `voxel_offset` serves its chunks at the keys Neuroglancer and
  tensorstore ask for, counted from the offset (`3-19_2-18_1-9`); every chunk of one not at
  0 was a 404. Chunk keys of every format are worldarray's (`wa.chunk_key`,
  `wa.parse_chunk_key`).
- `worldarray>=0.6,<0.7`, one minor series at a time. A tensorstore read stranded by its
  lost wakeup, which never returned, now runs within a second: worldarray's watchdog kicks
  the pools of every context it opens a store on. chunkmirage's stores share worldarray's
  context at tensorstore's own pool sizes (`wa.shared_context(cache_bytes, threads=0)`), as
  wide as the machine; worldarray's default sizes them to the job's slots, one decoding
  thread in a plain `bsub`.
- Neuroglancer links and precomputed's `resolution` convert units by worldarray's table
  (`wa.units`) rather than chunkmirage's own: centimetres, picometres, microns,
  nanoseconds, British spellings, plurals and capitals are converted, where a spatial axis
  in one of them was taken as nanometres and a time axis as unitless.
- The zarr frontends' OME multiscales are written by worldarray (`wa.ome_multiscales`): a
  translation is written even when it is 0, OME 0.5 has its `version` only beside
  `multiscales`, an axis with a name that is neither spatial nor a channel nor time is left
  untyped, and the multiscale no longer carries `"type": "chunkmirage"` or a description.

### Sources

- An N5 level with no `offset` has its corner at the origin at every level, as Neuroglancer
  draws N5 (worldarray 0.1.1's rule): voxel 0's centre is `downsamplingFactors / 2 ×
  resolution`. A level with `pixelResolution` followed BigDataViewer, half a voxel of the
  full resolution lower ([formats](docs/concepts/formats.md#stored-sources)).

### Browser engine

- Stored arrays are placed by worldarray's rules (`web/src/placement.ts`), checked in `npm
  test` against worldarray's fixture of metadata layouts, and the engine's OME multiscales
  against its fixture of written ones (`web/test/metadata.test.ts`). What the browser now
  reads as the server does: an OME multiscale's own `coordinateTransformations`, OME 0.6's
  intrinsic coordinate system, N5's `offset`, funlib's `voxel_size` and a COSEM `transform`
  on an array, `pixelResolution` or `resolution` on an N5 group, N5 without an offset
  (corner at the origin), a zarr array outside an OME group (funlib, COSEM, or xarray of
  more than three axes), and a single N5 array. Its OME multiscales type a channel axis as
  `channel` and a `t` axis as time, as worldarray does; both were `space`.

- Any package Pyodide ships is loaded the first time the Pyodide workers' Python imports it
  (a page's own modules, the ops), found by its module in Pyodide's lockfile: Pillow for
  `from PIL import Image` inside a page's function, say. Only scipy and scikit-image were
  ([design](docs/design.md#another-page-on-the-engine)).

## 0.1.0a2 (2026-10-21)

The second alpha. The plugin API only grew (`overlap`, `computed_from`,
`demand.cancelled()`, source matchers, parameter models), but an `ArrayInfo` given no
translation now has voxel 0's corner at the origin, not its centre (under Sources): a
plugin's source that meant its centre there passes `translation=(0.0,) * ndim`.

### Core and contrib

- The core composes, it does not compute: everything with an algorithm in it moved to
  `chunkmirage_contrib`, a second package in the same wheel (every op but `downsample` and
  `python`; `synthetic://`, `register://`, `stitch://` and GeoTIFF reading; registration,
  stitching and tracking; the browser's register, stitch and track pages, under
  `web/src/contrib/`). It reaches the core only through entry points, two of them new:
  `chunkmirage.source_matchers` for files told apart by their path, and `chunkmirage.models`
  for parameter models in `chunkmirage schema`. Imports change with it:
  `from chunkmirage_contrib.ops import Threshold`, `chunkmirage_contrib.registration`
  ([design](docs/design.md#what-stays-in-the-core)).

### Serving

- A long op that asks `demand.cancelled()` between its steps gives up once no client
  waits for its work any more.
- BLAS kept to one thread per call in the server (`threadpoolctl`), and affines applied
  without it: its threads fought the server's, and a three-field `warp://` read took ten
  times as long.
- The chunks one read needs are computed at once on a pool of a thread per core
  (`demand.gather`, `demand.set_gather_threads`), for the read's request, so a single
  request for a deep chain of stages uses every core; it ran in one thread before.
- `serve --device cuda|cuda:N|auto` (`chunkmirage.gpu.set_device`, the `cupy` extra):
  `warp://` and `scene://` resample on a GPU, their own code run with CuPy and
  `cupyx.scipy.ndimage`, the points pushed through affines and sampled fields and the image
  interpolated there (0.5 s for a 184 M-voxel volume through three lattice fields on an
  A100, 13.5 s on 24 CPU threads); values within a grey level of the CPU's, cache keys the
  same. A sampled field is read in its own precision, not copied
  to float64.
- `chunkmirage materialize` and `chunkmirage.materialize`: a dataset, or a box of it,
  computed and written into a store (a directory, `gs://`, `s3://`) as a frontend serves it,
  zarr v3, zarr v2 or N5, so it reads back as a stored dataset of its own; the metadata
  first, chunks several at once, empty ones left out. A box is exactly the box by default
  (`--extent box`: arrays of its shape, placed where it is, each of the dataset's chunks it
  reaches computed once), or its chunks in arrays of the levels' whole shapes
  (`--extent level`).

### Sources

- `dataset://<name>`: another dataset of the server as a source, read in-process through
  its stage cache; editing a dataset rebuilds what reads it, cycles are refused, and
  `GET /api/datasets/{name}` lists a dataset's `upstream`.
- `warp://` takes its field from any source (`field=<url>`: a `(c, z, y, x)` array of
  displacements, another dataset's output among them), an `affine` after the displacement
  and the output grid from `target=<source>`; the swirls stay as they were. Several fields,
  `|` apart, are applied in turn and the image resampled once (the passes of a
  registration composed, the latest first), and an `affine` or `target` alone places the
  image with no field.
- `coverage://<name>`: one voxel per chunk of another dataset's last cached stage, saying
  whether it is computed and whether the full-resolution result under it is (a preview
  told from a result while browsing), read afresh at every request;
  `GET /api/datasets/{name}/coverage` gives the counts per level.
- Stored arrays are opened and placed by worldarray (`worldarray>=0.1,<1`), the reader
  chunkmirage shares with cellmap-flow, cellmap-analyze and mesh-n-bone; chunkmirage keeps
  xarray coordinates, CF packing, the overrides and `kind`
  ([design](docs/design.md#why-worldarray-for-placing-them)). What it adds or changes:
  a precomputed scale opened by its key (`vol/4_4_40`) or as `vol/s<N>`; a one-channel
  precomputed volume served as `z, y, x`, its channel dropped; N5 with a bare-list
  `pixelResolution`, `units` as one string, or an `axes` attribute; zarr v3
  `dimension_units` and a `scale` attribute as the voxel size; a dataset path of more than
  one part (`seg/s0`) found in an ancestor group; an OME `scale` that disagrees with an
  array's own `voxel_size` warns and OME wins, and one of the wrong length is refused; a
  bucket with listing off read through its refusals; `AWS_ENDPOINT_URL`.
- An array nothing places has its corner at the origin, voxel 0's centre half a voxel on
  along its spatial axes, as Neuroglancer draws a plain array and worldarray places one: a
  stored array without placement metadata, an xarray dimension without an even
  coordinate, and every array made from nothing (`synthetic://`, an `ArrayInfo` given no
  translation, the browser engine's alike). It was voxel 0's centre at the origin, so
  these move by half a voxel; `synthetic://` levels now cover the same extent, and served
  as precomputed their `voxel_offset` is 0 rather than a rounded half voxel. Explicit
  translations, and arrays that take their input's, are unchanged
  ([formats](docs/concepts/formats.md#stored-sources)).

### Ops and pipelines

- An op on a coarser grid (one vector per `stride` voxels) gets output chunks that follow
  the grid: the input's chunk divided by the factor, or what its `output_info` sets; an op
  at one resolution and the pyramid made from it keep their chunks, as a stored pyramid does.
  One that sets `computed_from` (its input's `ArrayInfo`) has `warp://` take a field's
  level by the voxels it was fitted on, so a zoomed-out view reads the lattice fitted at its
  level, not the finest, and sees its input's exact extent in `for_level`.
- `overlap` on an op: each block also computes that margin beyond its chunk, the blocks are
  cached as a stage of their own, and a served chunk is the weighted mean of every block
  reaching it, smooth where blocks computed on their own disagree and the same whatever
  order they are computed in; `register://`'s refined blocks blend with the same weights.
- `Pipeline.plan(level, box)`: what a read would compute, computing nothing: per stage, the
  chunks it is asked for, those held and those to compute, following `dataset://` into
  the datasets it reads and stopping at other sources.

### Browser engine

- The engine as a base for a page of another project
  ([design](docs/design.md#another-page-on-the-engine)): Python loads only when a view
  needs it, a view with no ops served from the reader alone; `unserve()` forgets a view and
  its chunks, `stop()` the engine's workers; the service worker's relay and store cache as
  a module (`web/src/relay.ts`, with `route()` on the page) a page's own service worker
  builds on, relaying any request and streaming an answer whole or in pieces; `serveStages()`
  for a view the page computes in stages, each stage's output kept under its own key as the
  Python pipeline keeps each stage's chunks, the first stage read at its own voxel size; an
  `onnx` stage (ONNX Runtime Web on WebGPU, WebAssembly without it) and the Vite plugin that
  serves its runtime beside the page.
- A page's own Python in the workers: `Engine.start`'s `modules` writes a page's modules
  beside the package's (never under `chunkmirage/` or `chunkmirage_contrib/`),
  `call("pkg.mod:fn", args, arrays)` runs one by module path (`chunkmirage.calls`:
  `fn(args, arrays)` answers JSON, one array as bytes or numpy, or JSON with arrays, and runs
  natively too), and `workers` sets how many Pyodide workers start, one at least
  ([design](docs/design.md#another-page-on-the-engine)).
- A staged view's stages may return 64-bit integer blocks (`BigUint64Array` and
  `BigInt64Array`; `uint64` and `int64` in the engine's dtype table), kept and served as
  zarr's `uint64` and `int64` with labels past 2^53 whole, as the Python ops' `np.uint64`
  are; a zarr source stored as 64-bit integers is read as stored, at its own width.
- A relayed request's query string reaches the page: `Ask.search` (the URL's, with its `?`,
  or empty), beside `path`, which stays the pathname alone that routes match.
- Neuroglancer precomputed volumes read in the browser (`precomputed://`, or found by their
  `info`): unsharded or sharded (`neuroglancer_uint64_sharded_v1`, identity hash), `raw`,
  `jpeg` or `compressed_segmentation`; `gs://` and `s3://` sources fetched as Neuroglancer
  fetches them, `gs://` through the Cloud Storage JSON API (a bucket's own URL serves a page
  no CORS headers unless its owner set them) and `s3://` at the bucket's https URL
  (`web/src/urls.ts`, which the service worker shares).
- The service worker leaves a store's metadata to the page (`metadata(url)` in
  `web/src/relay.ts`) and caches only chunks, so Chrome can ask the visitor about a volume
  on a private-network host instead of refusing the worker's fetch.
- ONNX calls take one turn on the page whatever their session (`exclusive` in
  `web/src/onnx.ts`): forward passes, and sessions built and released, since ORT Web runs
  every session in one WebAssembly instance that its WebGPU build suspends while it waits.
  Two models' passes, or a second model built while the first computed, could hang the page
  and crash the tab. A page's own passes on the same ORT take the turn through `exclusive`.
- `Engine.materialize(view, store, options)` writes a box of a served view (or a whole
  level) into a store as an OME-Zarr of its own, zarr v2 with OME 0.4 or zarr v3 with OME
  0.5, gzipped, its empty chunks left out, the metadata the engine serves for the view
  (`web/src/layout.ts`, shared with the routes) with the one level as its one dataset:
  exactly the box (`extent: "box"`, the default), `hi − lo` voxels with the level's
  translation moved by `lo` voxels, its chunks the served chunk shape counted from `lo`,
  each stitched from the served chunks it overlaps (each computed once, walked with the
  split axis of the most chunks outermost, so a slab across the other split axes is held
  at a time), those at its far edge stored whole with the fill value past the box; or the
  level's whole shape, the box's chunks in it rounded out to whole chunks (`extent:
  "level"`, for several boxes in one store); a box of the whole level is the same store
  either way; progress counts the array's chunks done and the served chunks computed
  (`computed`, `needed`), and `Engine.materializePlan(view, options)` says beforehand what
  it will write and the most bytes of the box's chunks it will hold (`peakBytes`); the
  Python side (`chunkmirage materialize --extent`) writes either too, the box by default;
  `directoryStore()` writes into a folder picked with the File System Access API (or the
  origin's private file system), `memoryStore()` into memory
  ([formats](docs/concepts/formats.md#a-region-written-to-a-store-browser-engine)). Its
  chunks share the queue and the kept chunks with the viewer's but run in the background:
  behind every chunk a client waits for, in all the queue's slots but one, and never dropped
  as abandoned (`submit`, `claim` and `release` take `background`;
  [caching](docs/concepts/caching.md#order-of-work-and-requests-given-up-on)).

## 0.1.0a1 (2026-10-05)

The first release, an alpha: the plugin API can still change before 0.1.0.

### Serving

- Pipelines served as N5, Zarr v2 (OME-NGFF 0.4), Zarr v3 (OME-NGFF 0.5) and Neuroglancer
  precomputed at once, and meshes made when fetched.
- A per-stage chunk cache keyed by the pipeline's hash, and cache-busting digests in links.
- Live edits through the REST API, Server-Sent Events, a control page and a
  python-neuroglancer viewer that follows them.
- Work bounded, shared between requests and ordered finest level first, then by when
  each request arrived; work no client waits for any more is dropped.
- `--token` for the control API, `--https` with a self-signed certificate.
- zarr v2 and v3 chunks compressed with blosc (zstd, byte shuffle) by default, 10 to 30
  times quicker to encode than gzip; `--compressor` chooses another. `--cache-gb` takes effect (an empty cache
  given to the registry used to be replaced by the default 2 GiB one), and `0` turns the
  cache off.
- `serve --port 0` and `--ready-file` for launchers; the port is bound before it is
  announced.
- Datasets resolved by name on first request (`resolver`), routes from other packages
  (`extra_routes`, the `chunkmirage.routes` entry point), and the app mounted under a prefix
  in another app.
- `chunkmirage.serve` and `Server`: an app served from Python on a port bound first, with a
  callback once it accepts connections, in a background thread or this one.

### Sources

- zarr v2/v3, N5, precomputed and HDF5 on file, S3, GCS and HTTP; xarray arrays with their
  coordinates and CF packing; cloud-optimized GeoTIFFs.
- Computed sources: `synthetic://`, `scene://` (OME-Zarr 0.6 transformations), `warp://`,
  `register://` (deformable registration on a GPU), `stitch://` (BigStitcher tiles),
  `stack://` and `flip://`.
- S3 read anonymously before credentials, GCS through its public URL when credentials
  fail, metadata reads that give up after seconds, zarr compressors with members
  tensorstore rejects.
- Voxels placed as Neuroglancer places them: precomputed `voxel_offset` and funlib `offset`
  are corners.
- Stored integers of 32 bits or more read as labels, booleans as masks, unless the spec's
  `kind` says otherwise.

### Ops and pipelines

- Pointwise ops, filters, morphology, connected components, spots, contacts, differences,
  gradients, downsampling, normalized differences, slope and hillshade, with halos handled
  by the pipeline.
- Ops that return their block shaved by the halo or add leading axes; ops at one
  resolution with the coarser levels downsampled from their output, or the nearest level
  read as it is (`input_level`, `level_rtol`).
- `cache_token()` for ops whose output depends on state outside their parameters, and
  `DatasetRegistry.refresh` to read it again.
- `slots` to bound how many calls of an op run at once.
- Zero padding past the volume's edge (`padding`).
- Channel axes read whole, so an op may change their length: select or combine channels.
- A level shrunk by a whole factor for an op at one voxel size is the mean of each block,
  not linear samples.

### Browser engine

- Registration on WebGPU and pipelines of the package's own ops in Pyodide, in a gallery of
  demos.
