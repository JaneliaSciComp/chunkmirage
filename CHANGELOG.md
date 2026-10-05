# Changelog

Notable changes per release. Versions follow the rules in
[contributing](docs/contributing.md#plugin-api-and-versioning): the plugin API is stable,
and a release that breaks it says so here.

## 0.1.0a1 (unreleased)

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
