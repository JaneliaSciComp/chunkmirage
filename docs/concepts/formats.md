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

### Scene sources (OME-Zarr 0.6 transformations)

`scene://<group>?image=<image path>&target=<image path or coordinate system>`

serves one image of an [OME-Zarr 0.6](https://ngff.openmicroscopy.org/specifications/0.6/)
scene resampled onto another grid through the scene's coordinate transformations,
including displacement fields. No viewer applies those yet: Neuroglancer refuses to load
an image whose transformation is a displacement field, and napari and vizarr skip them.
Served this way, every viewer and every zarr reader sees the registered volume, and
nothing is written. Each output chunk is computed on its own: its voxel centres go
through the transformation chain into the source image, the exact bounding box of the
results is read at the matching resolution level, and the values are interpolated. Ops
can follow as with any source.

```bash
chunkmirage serve "scene:///data/fly_brains.zarr?image=FCWB&target=JRC2018F"
```

`examples/fly_brain_registration.py` builds that scene from the public RFC-5 example (two
Drosophila brain templates, a displacement field plus affine between them) and serves the
registered template next to the fixed one.

| parameter | default | meaning |
| --------- | ------- | ------- |
| `image` | the group itself, if it is a multiscale image | image to resample, relative to the group |
| `target` | the scene's first coordinate system, else the image's own | an image path (its grid and levels are used) or a coordinate system name |
| `interpolation` | `linear`; `nearest` for label images | `nearest` keeps label ids exact |
| `inverse` | `exact` | `approx` also walks displacement fields backwards, estimating their inverse by fixed-point iteration |
| `shape`, `voxel_size`, `translation`, `levels`, `chunk` | the target image's grid, else the source image's | explicit output grid, e.g. for a scene coordinate system with no image: spatial axes, C order, target units; one value applies to every axis |
| `field_cache_gb` | `0.0625` | decoded-chunk cache per displacement field; raise it for fields stored in large chunks so each is decoded once (see below) |
| `large_field_chunks` | `0` | `1` accepts fields stored in chunks over 256 MiB, decoding them one at a time (see below) |

**Finding the transformation.** Coordinate systems (the scene's own and each image's) are
the nodes of a graph and transformations are its edges. The shortest path from the target
to the image's intrinsic system is used. An edge is walked backwards only if it has a
closed-form inverse (scale, translation, affine, rotation, mapAxis, and sequences or
byDimension of those) or is a `bijection`. Registration tools usually store fixed →
moving, which is the direction resampling needs, so this is rarely a limit; otherwise
`inverse=approx` estimates the inverse of displacement fields. Where a field folds there
is no unique inverse, and those voxels are left empty rather than filled wrongly.

**Details.**

* Transformation types: identity, scale, translation, affine, rotation, mapAxis,
  projectAxis, sequence, displacements, coordinates, bijection, byDimension; parameters
  inline or stored under `path`.
* Fields: 0.6 multiscale field groups (vector axis first), and the draft layout (a bare
  array with the vector axis last, as BigWarp exports). Outside its grid a field takes the
  nearest edge value. `cubic` field interpolation is read as linear.
* Time and channel axes of the image pass through unchanged; transformations act on the
  spatial axes. The output is the image's leading axes plus the target's spatial grid.
* Each output level reads the coarsest source level that is still at least as fine as one
  output voxel, measured through the transformation at the grid centre, so zoomed-out
  views read small levels.
* Voxels that map outside the image are 0.
* Field chunking decides speed and memory. Each output chunk reads the small window of the
  field it needs, but a store decodes whole chunks: a field saved as a few huge chunks (for
  example one full-z, full-x band per writer job) costs a full-chunk decode per output
  chunk, and a viewer's parallel requests start several at once. A 113 GB field in 6.6 GiB
  chunks exhausted a 93 GB workstation this way. So fields whose chunks exceed 256 MiB are
  refused, with the numbers, unless `large_field_chunks=1`; then they are decoded one at a
  time, and `field_cache_gb` above one chunk decodes each only once. Better: store fields in
  chunks of about 64³, or sharded with small inner chunks.
* One output region may read at most 2 GiB of the source image; a transformation that
  spreads a chunk over more than that is refused rather than allocated. Use smaller output
  chunks (`chunk=`) if you hit it.
* Also read: the 0.6 drafts' string `input`/`output` references, `multiscales` objects and
  snake_case `input_axes`.
* The [RFC-5 transformation conformance
  suite](https://github.com/clbarnes/ome_zarr_transformations_conformance) runs in the tests
  (`tests/data/rfc5_conformance`).

The cache key of each level folds in the whole transformation chain, so editing the
scene's metadata and re-`PUT`ting the dataset gives new URLs. A field rewritten in place
under the same path keeps its key; restart or clear the cache after doing that.

### Warp sources (procedural deformations)

```
warp://<image>?field=swirl&angle=90&radius=200&centre=z,y,x&plane=y,x
warp://<image>?field=swirls&count=8&seed=0&angle=90&radius=200
```

serves `<image>` (anything above or below, including `synthetic://` URLs) twisted on its
own grid by swirls computed from coordinates, and with `&show=field` the displacement
itself: a `(c, z, y, x)` float32 volume whose three components are the z, y and x
displacement in the image's units. Nothing is stored, not even the field, so any
parameter can change at any time: `PUT` a new URL and the viewer refetches what is on
screen. The resampling is the same as for scene sources, and each chunk reads only the
compact region its rotated voxels come from, however large the displacement.

`swirl` is one twist about an axis along a spatial axis (z for the default `plane=y,x`),
fading with distance from that axis, so it is a column through the volume. `swirls`
(3-D images) are `count` balls of twist at random places, each about its own axis in a
random direction and fading with distance from its centre, so they move points along z
too. Their displacements add; where they overlap the sum is smooth but no longer a pure
rotation. The same `seed` gives the same swirls.

With `&frames=N` the swirl grows along a new leading `t` axis, from none at frame 0 to
`angle` at the last, so a viewer animates it by playing `t` (Neuroglancer's playback)
instead of loading new URLs: each frame is its own chunks, computed when first shown and
cached from then on. `examples/swirl_demo.py` shows the original, the swirled image and
the field side by side and plays the frames.

| parameter | default | meaning |
| --------- | ------- | ------- |
| `field` | (required) | `swirl` or `swirls` |
| `angle` | `90` | twist on the axis, degrees; it fades as `exp(-(r/radius)²)` with distance `r` from the axis (`swirl`) or the centre (`swirls`). Each of the `swirls` turns by ½ to 1 × `angle`, either way |
| `radius` | `swirl`: a quarter of the smaller in-plane extent; `swirls`: a fifth of the smallest extent | fall-off distance, image units; each of the `swirls` gets ½ to 1 × `radius` |
| `centre` | the volume centre | image units, C order: a point on the axis (`swirl`), the middle of the box the centres fall in (`swirls`) |
| `plane` | the last two axes, e.g. `y,x` | `swirl`: the two axes that turn |
| `count` | `8` | `swirls`: how many |
| `seed` | `0` | `swirls`: random seed for their centres, axes, radii and angles |
| `spread` | half the volume | `swirls`: half-size of the box around `centre` their centres fall in; one value or `z,y,x` |
| `show` | `image` | `field` serves the displacement instead |
| `frames` | none | frames on a leading `t` axis; frame `i` twists by `angle·i/(frames−1)`. An image's own `t` axis must hold one time point, which the frames replace |
| `chunk` | the image's | output chunk shape of the spatial axes, C order; thin chunks such as `8,128,128` compute less for a view of one plane |
| `interpolation` | `linear`; `nearest` for uint32/uint64 | as for scene sources |

The parameters follow the last `?`, so the image URL may carry its own query. To show the
field as RGB, its components must be a Neuroglancer shader channel dimension (`c^`, read
with `getDataValue(0..2)`). The precomputed frontend makes them one, but holds only
`(c, z, y, x)`; with frames, serve zarr and rename the channel dimension `c'` to `c^`
(`Viewer.rename_dimensions`, as the demo does).

### Stored sources

Sources are detected by content, not extension: `zarr.json` → zarr v3, `.zarray` → zarr v2,
`attributes.json` → N5, `info` → precomputed. A group's levels are the paths in its
`multiscales[0].datasets` (OME-NGFF, COSEM N5), in that order; without that, `s0, s1, ...`
are probed. N5 and precomputed data are transposed to C order `(z, y, x)` on read.

Voxel size, translation, units and axes are read per level, first match wins:

| Format      | Metadata, in order of precedence                                                   |
|-------------|------------------------------------------------------------------------------------|
| zarr v2/v3  | parent OME-NGFF `multiscales` entry whose `path` is this array, composed with the multiscale-level `coordinateTransformations` if present (0.4/0.5; in 0.6 those lead to other coordinate systems and are applied only through a [`scene://`](#scene-sources-ome-zarr-06-transformations) source, and axes come from the intrinsic coordinate system); else the array's own `resolution`/`voxel_size`, `offset`, `units`, `axis_names` (funlib) or `transform` (COSEM), C order |
| N5          | the array's `transform` (COSEM, C order); the parent's `multiscales[].datasets[].transform` for this path; `pixelResolution`/`resolution` (array, else group) × `downsamplingFactors`, plus `offset`, x-first |
| precomputed | `resolution` (nm) and `voxel_offset` × `resolution` as the translation             |
| HDF5        | `resolution`/`voxel_size` and `offset` attributes, C order                         |

`offset`/`translate` are in world units. Anything in the spec (`voxel_size`, `units`,
`axes`, `translation`) overrides what was read. HDF5 uses `file.h5::/dataset` and needs
the `hdf5` extra.
