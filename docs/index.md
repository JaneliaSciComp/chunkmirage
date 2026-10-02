# chunkmirage

**Spoof chunked array formats over HTTP with on-the-fly processing.**

[▶ Try the demos in your browser](https://yuriyzubov.github.io/chunkmirage/browser/){ .md-button .md-button--primary }
[All demos](demos.md){ .md-button }

<div class="grid" markdown>
[![Los Angeles fires: burn severity, read by Neuroglancer, a web map or GDAL](https://yuriyzubov.github.io/chunkmirage/browser/cards/fires.jpg){ width="32%" }](https://yuriyzubov.github.io/chunkmirage/browser/pipeline.html?card=fires)
[![A 3-D fractal 2^28 voxels across](https://yuriyzubov.github.io/chunkmirage/browser/cards/mandelbulb.jpg){ width="32%" }](https://yuriyzubov.github.io/chunkmirage/browser/pipeline.html?card=mandelbulb)
[![The Gulf Stream's fronts](https://yuriyzubov.github.io/chunkmirage/browser/cards/fronts.jpg){ width="32%" }](https://yuriyzubov.github.io/chunkmirage/browser/pipeline.html?card=fronts)
</div>

chunkmirage serves *virtual* datasets that look, to any HTTP-capable viewer or library
(Neuroglancer, BigDataViewer/Fiji, vizarr, napari, dask, tensorstore, ...), like ordinary
Zarr v2, Zarr v3, N5, or Neuroglancer Precomputed volumes. Nothing exists on disk. Every
chunk is computed when requested: read from a real source, pushed through a pipeline of
ops, encoded in whatever format the client asked for, and cached so that tweaking a
parameter downstream never re-reads or re-computes upstream stages.

```
viewer / dask  --HTTP-->  chunkmirage  --tensorstore/h5py-->  real data (zarr/n5/precomputed/hdf5; file/s3/gcs/http)
                             |
                             +-- pipeline: source -> [op, op, ...] -> encoded chunk
                             +-- per-stage chunk cache keyed by pipeline hash
                             +-- frontends: n5 | zarr | zarr3 | precomputed, all served at once
                             +-- REST API for live pipeline edits
```

chunkmirage was inspired by [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5),
which serves N5 chunks computed when a viewer asks for them, and by
[cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow), which grew out of example-virtual-n5
and serves live model inference the same way. chunkmirage makes that trick general: any
format, any source, any per-chunk computation, for any client.
It is a general-purpose tool: any viewer or library that reads chunked arrays over HTTP,
any source format, any per-chunk computation. **[See the demos](demos.md)**, which run in
your browser with nothing to install.

## When to use it

Use chunkmirage when you want to *look before you compute* on data too large to process
speculatively, and the operation is one a viewer's shader cannot do:

* **non-local**: filters, morphology, distance transforms;
* **learned**: model inference where GPU and weights live server-side;
* **geometric**: resampling under a registration transform (including OME-Zarr 0.6
  displacement fields, which viewers cannot apply: [`scene://`](concepts/formats.md#scene-sources-ome-zarr-06-transformations));
* **multi-source**: masking one volume by another, comparing two model versions;
* **format bridging**: exposing HDF5 or a custom reader as zarr to any viewer.

Nothing in it is particular to microscopy: a time series is an array too, and xarray-written
zarr (sea temperature, weather, images of the sun) reads with its own axes and coordinates,
so a day-to-day change is an op along time.

The demos also run entirely in the browser, with no server and nothing to install:
[the gallery](https://yuriyzubov.github.io/chunkmirage/browser/) has registration solved on
your GPU and pipelines of chunkmirage's own ops run in the page (see [Demos](demos.md)).

See [FAQ](faq.md) for when *not* to use it.

## License and authors

BSD 3-Clause, Howard Hughes Medical Institute. Authors: TBD (collaborative project).

## Where next

* [Getting started](getting-started.md): install, serve, open in Neuroglancer.
* [Interactivity](concepts/interactivity.md): control page, python viewer, how refetching works.
* [Caching](concepts/caching.md): what is cached, where, and why some stages are not.
* [Formats and URLs](concepts/formats.md): the exact URLs each viewer needs.
* [Roadmap](roadmap.md): what is planned and why.
