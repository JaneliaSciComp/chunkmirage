# Changelog

Notable changes per release. Versions follow the rules in
[contributing](docs/contributing.md#plugin-api-and-versioning): the plugin API is stable,
and a release that breaks it says so here.

## 0.1.0a1 (unreleased)

The first release, an alpha: the plugin API can still change before 0.1.0.

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

- Pipelines served as N5, Zarr v2 (OME-NGFF 0.4), Zarr v3 (OME-NGFF 0.5) and Neuroglancer
  precomputed at once, and meshes made when fetched.
- A per-stage chunk cache keyed by the pipeline's hash, and cache-busting digests in links.
- Live edits through the REST API, Server-Sent Events, a control page and a
  python-neuroglancer viewer that follows them.
- Work bounded, shared between requests and ordered finest level first, then by when
  each request arrived; work no client waits for any more is dropped, and a long op that
  asks `demand.cancelled()` between its steps gives up once none does.
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
