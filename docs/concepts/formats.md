# Formats and URLs

Every dataset is served through every frontend simultaneously. Given a dataset named
`em` on `http://localhost:8000`:

| frontend      | Neuroglancer source URL                                | notes                                            |
| ------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `n5`          | `n5://http://localhost:8000/em/n5`                     | multiscale group, `s0..sN`; gzip or raw chunks   |
| `zarr`        | `zarr2://http://localhost:8000/em/zarr`                | Zarr v2 + OME-NGFF 0.4 `multiscales`; consolidated `.zmetadata` too |
| `zarr3`       | `zarr3://http://localhost:8000/em/zarr3`               | Zarr v3 + OME-NGFF 0.5 in group attributes       |
| `precomputed` | `precomputed://http://localhost:8000/em/precomputed`   | `raw` encoding, 3-D or 4-D (c,z,y,x) only; HTTP gzip when accepted |

A cache-busting token may be inserted after the name: `/em/@{digest}/zarr3`. The API always
hands out this form; the plain form always serves the current pipeline.

## Chunk paths

| frontend      | metadata                              | chunk key                                       |
| ------------- | ------------------------------------- | ----------------------------------------------- |
| `n5`          | `attributes.json`, `s0/attributes.json` | `s0/x/y/z` (x first)                          |
| `zarr`        | `.zgroup`, `.zattrs`, `s0/.zarray`    | `s0/z/y/x` or `s0/z.y.x` (both accepted)        |
| `zarr3`       | `zarr.json`, `s0/zarr.json`           | `s0/c/z/y/x`                                    |
| `precomputed` | `info`                                | `s0/x0-x1_y0-y1_z0-z1`, must align to chunk grid |

Internally arrays are numpy C order `(z, y, x)`. N5 and precomputed list axes x-first, so
their metadata and keys are reversed relative to zarr.

## Compression

`Zarr2Frontend`, `Zarr3Frontend` accept `compressor="gzip" | "zstd" | "blosc" | "none"`;
`N5Frontend` accepts `gzip` or `raw`. Precomputed `raw` is uncompressed by definition, so
the server applies `Content-Encoding: gzip` when the client accepts it.

## Edge chunks

Zarr requires full-size chunks, so edge chunks are zero-padded. N5 and precomputed encode
the clipped size.

## Data types

Precomputed supports `uint8`, `uint16`, `uint32`, `uint64`, `float32` only; add a `cast` op
for anything else. Precomputed volumes with `uint32`/`uint64` are typed `segmentation`,
others `image`; override with `PrecomputedFrontend(volume_type=...)`.

## Sources

### Synthetic (procedural) sources

`synthetic://<kind>?shape=z,y,x&chunk=64,64,64&levels=N&seed=0&voxel_size=8&unit=nm`
generates data on the fly from voxel coordinates. Nothing is stored, so the volume can be
as large as you like, and each scale level is the same function sampled at a coarser
spacing, so the pyramid is exact (`s1[z,y,x] == s0[2z,2y,2x]`). Kinds: `blobs` (Gaussian
blobs), `shells` (hollow spheres, membrane-like), `noise` (fractal value noise), `julia`
(a 3-D slice of a quaternion Julia set); combine with `+`, e.g. `blobs+noise`. Useful for
demos and for stress-testing pipelines without I/O. Generation is vectorised numpy, so
the server's threadpool runs it on all cores.

### Stored sources

Sources are detected by content, not extension: `zarr.json` → zarr v3, `.zarray` → zarr v2,
`attributes.json` → N5, `info` → precomputed. Groups are walked for `s0, s1, ...`. Voxel
size and units come from OME-NGFF `multiscales`, N5 `transform`/`pixelResolution`, or
precomputed `resolution`, and can be overridden in the spec (`voxel_size`, `units`, `axes`,
`translation`). HDF5 uses `file.h5::/dataset` and needs the `hdf5` extra.
