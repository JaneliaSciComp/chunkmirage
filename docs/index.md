# chunkmirage

**Spoof chunked array formats over HTTP with on-the-fly processing.**

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

It is a general-purpose tool: any viewer or library that reads chunked arrays over HTTP,
any source format, any per-chunk computation. Prior art that uses the same trick for one
format or one purpose includes [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5)
and the serving layer of [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow);
projects like those are expected consumers, not the reason it exists.

## When to use it

Use chunkmirage when you want to *look before you compute* on data too large to process
speculatively, and the operation is one a viewer's shader cannot do:

* **non-local**: filters, morphology, distance transforms;
* **learned**: model inference where GPU and weights live server-side;
* **geometric**: resampling under a registration transform (including OME-Zarr 0.6
  displacement fields, which viewers cannot apply: [`scene://`](concepts/formats.md#scene-sources-ome-zarr-06-transformations)),
  on-the-fly pyramids;
* **multi-source**: masking one volume by another, comparing two model versions;
* **format bridging**: exposing HDF5 or a custom reader as zarr to any viewer.

Registration also runs entirely in the browser, with no server and nothing to install:
[browser/register.html](https://yuriyzubov.github.io/chunkmirage/browser/register.html)
solves on your GPU and opens with an example.

See [FAQ](faq.md) for when *not* to use it.

## License and authors

BSD 3-Clause, Howard Hughes Medical Institute. Authors: TBD (collaborative project).

## Where next

* [Getting started](getting-started.md): install, serve, open in Neuroglancer.
* [Interactivity](concepts/interactivity.md): control page, python viewer, how refetching works.
* [Caching](concepts/caching.md): what is cached, where, and why some stages are not.
* [Formats and URLs](concepts/formats.md): the exact URLs each viewer needs.
* [Roadmap](roadmap.md): what is planned and why.
