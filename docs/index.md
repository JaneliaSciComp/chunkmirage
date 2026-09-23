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

It generalizes [example-virtual-n5](https://github.com/stuarteberg/example-virtual-n5) and
the serving layer of [cellmap-flow](https://github.com/janelia-cellmap/cellmap-flow), and is
intended to become the serving, caching and format backbone that cellmap-flow imports.

## When to use it

Use chunkmirage when you want to *look before you compute* on data too large to process
speculatively, and the operation is one a viewer's shader cannot do:

* **non-local**: filters, morphology, distance transforms;
* **learned**: model inference where GPU and weights live server-side;
* **geometric**: resampling under a registration transform, on-the-fly pyramids;
* **multi-source**: masking one volume by another, comparing two model versions;
* **format bridging**: exposing HDF5 or a custom reader as zarr to any viewer.

See [FAQ](faq.md) for when *not* to use it.

## License and authors

BSD 3-Clause, Howard Hughes Medical Institute. Authors: TBD (collaborative project).

## Where next

* [Getting started](getting-started.md): install, serve, open in Neuroglancer.
* [Caching](concepts/caching.md): what is cached, where, and why some stages are not.
* [Formats and URLs](concepts/formats.md): the exact URLs each viewer needs.
* [Roadmap](roadmap.md): what is planned and why.
