# Formats and URLs

Every dataset is served through every frontend simultaneously. Given a dataset named
`em` on `http://localhost:8000`:

| frontend      | Neuroglancer source URL                                | notes                                            |
| ------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `n5`          | `n5://http://localhost:8000/em/n5`                     | multiscale group, `s0..sN`; gzip or raw chunks   |
| `zarr`        | `zarr2://http://localhost:8000/em/zarr`                | Zarr v2 + OME-NGFF 0.4 `multiscales`; consolidated `.zmetadata` too |
| `zarr3`       | `zarr3://http://localhost:8000/em/zarr3`               | Zarr v3 + OME-NGFF 0.5 in group attributes       |
| `precomputed` | `precomputed://http://localhost:8000/em/precomputed`   | `raw` encoding, 3-D or 4-D (c,z,y,x) only; HTTP gzip when accepted |
| `mesh`        | `precomputed://http://localhost:8000/em/mesh`          | a surface of the dataset as Neuroglancer meshes, segment `1`, each fragment meshed when fetched, at one resolution or several (finer where you zoom); see [Meshes](#meshes-computed-when-fetched) |

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

The group and each level also answer as directories, for clients that browse a store
rather than name its keys: a path ending in `/`, or a group or level path asked for with
HTML first in `Accept` (as browsers and Java's HTTP client send), gets a listing in the
form Python's `http.server` writes, with the metadata keys and one `sN/` per level, never
the chunks. That is how Fiji's N5 viewer finds the levels (n5's HTTP access lists them;
checked with n5 4.0.1 and n5-zarr 2.0.1 for N5, zarr v2 and v3). Clients that ask for
`*/*`, as Neuroglancer, tensorstore and zarr-python do, get the metadata at those paths as
before. Precomputed has no directories.

| client | reads | checked |
| ------ | ----- | ------- |
| Neuroglancer | all four formats | the docs' demos |
| Fiji / BigDataViewer (n5-universe) | N5, zarr v2, zarr v3, and browses the levels | n5's HTTP access reading every format and listing the levels; the Fiji application itself not run |
| zarr-python, dask, napari's zarr reader | zarr v2, v3 | zarr-python 3.4 and dask reading a pipeline over HTTP |
| tensorstore | all four formats | `tests/test_clients.py`, against a running server |
| webKnossos | zarr v2, v3, N5, precomputed | not run; it asks for byte ranges only of sharded data, which the server does not produce |

## Compression

`Zarr2Frontend`, `Zarr3Frontend` accept `compressor="blosc" | "zstd" | "gzip" | "none"`,
blosc (zstd with byte shuffle) by default: on a 64×256×256 chunk it encodes float32 30
times faster than gzip and uint8 about 30 times faster, at the same ratio, and every
zarr reader listed above reads it except Fiji and BigDataViewer without the native Blosc
library (`--compressor gzip` for them). `N5Frontend` accepts `gzip` or `raw`. Precomputed `raw` is uncompressed by definition, so
the server applies `Content-Encoding: gzip` when the client accepts it.

## Edge chunks

Zarr requires full-size chunks, so edge chunks are zero-padded (by the browser engine too).
N5 and precomputed encode the clipped size.

## GeoZarr, for map clients (browser engine)

The browser engine serves each view twice: as OME-Zarr 0.5 for Neuroglancer
(`virtual/<page>/<view>/`) and, for a view read from a georeferenced source (a GeoTIFF), as
GeoZarr for map clients such as OpenLayers (`virtual/<page>/geo/<view>/`). The GeoZarr
group carries the zarr-conventions `multiscales` (one `layout` entry per level, its
`spatial:shape`), `proj:` (`proj:code`, the projection the page names: a map client must
know it, as the Moon page registers the lunar south polar stereographic) and `spatial:`
(`spatial:dimensions` `[y, x]`, `spatial:bbox` the pixels' corners) attributes; level `<i>`
is a group holding the view as its one band, `<i>/<view>`, a 2-D `y, x` array (the view's
one z plane; a map reads 2-D bands). Chunks are the same chunks, computed once whichever
layout asks. The Python server has no GeoZarr frontend yet. The engine also serves each
view as zarr v2 with OME-Zarr 0.4 (`virtual/<page>/zarr2/<view>/`), the same chunks, for
readers that predate zarr v3's final spec: GDAL 3.8, as the fires demo runs it in the page
(gdal3.js), opens a level as a zarr array and translates it to a GeoTIFF.

## A dataset written to a store (Python and the command line)

```bash
chunkmirage materialize SOURCE OUT [--op ...] [--format zarr3|zarr|n5] [--level N ...] [--box lo:hi]
```

computes SOURCE through its ops and writes it to OUT (a directory, or a `file://`, `gs://`
or `s3://` URL, through tensorstore) as a frontend serves it: zarr v3 with OME-Zarr 0.5
(the default), zarr v2 with OME-Zarr 0.4 and consolidated metadata, or N5. The metadata
and every chunk are byte for byte what a client fetches, so the store opens wherever the
served dataset did and reads back as a stored source of its own: a stage computed once (a
registration's pass, a model's output over a region) is then read from disk instead of
computed again. Every level's metadata is written first, so a store whose writing stopped
part way is still the dataset, zero where no chunk was written yet; then the chunks of the
levels asked for (all by default), several at once, each of nothing but zeros left out
(`--keep-empty` writes them). `--box` takes level 0's voxels along the last axes, `lo:hi`
each comma separated in C order (`0,0,0:64,512,512`), and every level writes the same
region on its own voxels, rounded out to whole chunks. From Python,
`chunkmirage.materialize(pipeline, where, format=, levels=, box=, compressor=,
skip_empty=, progress=)` does the same for any pipeline, another dataset's chain read
through `dataset://` among them (`registry.get(name)`), and returns the counts
(`{chunks, written, skipped, bytes}`). Inside a running server it competes with clients'
requests as any work asked for from Python does (never dropped, in arrival order), and
its chunks pass through the server's cache.

## A region written to a store (browser engine)

The browser engine also writes a view, or a box of it, into a store as files
(`Engine.materialize`; see [the design notes](../design.md#another-page-on-the-engine)): a
folder the visitor picks, the origin's private file system or memory. What it writes is
the layout it serves for the view, one level of it, so the result opens wherever the view
did and lands in the same place: the group lists the level written as its one dataset,
`0`, with that level's scale; the array has the chunk shape and dtype served, and chunks
compressed with gzip unless asked not to be. What the array covers is the `extent`:

* `"level"` (the default): the level's whole shape, with the level's translation, so a box
  keeps its place in it; the chunks of the box are written, rounded out to whole chunks,
  each the chunk the engine serves.
* `"box"`: exactly the box, `hi − lo` voxels (clipped to the level, rounded out to whole
  voxels, not to chunks), with the level's translation moved by `lo` voxels (the
  translation is where voxel 0's centre is, so it lands where the view's voxel `lo` does).
  Its chunks have the served chunk shape, counted from `lo`. Where `lo` is not on the
  served chunks' edges, each of the box's chunks is stitched from the served chunks it
  overlaps, up to two along each axis, each served chunk computed once: they are walked in
  C order and each is copied into the box's chunks it overlaps, a box chunk written once
  the last of them is in, so only the box's chunks begun and not yet complete are held, at
  most a slab of them (one chunk thick along the first axis on which `lo` is off the
  chunk edges), a row and one more; none where `lo` is on chunk edges, the box's chunks
  then the served chunks as they are. The chunks at the box's far edge overhang it and are
  stored whole, zero (the fill value) past the box, as zarr stores a chunk at an array's
  edge (zarr v3's regular grid: border chunks have the full chunk size, the fill value
  recommended outside the array; zarr v2: every chunk has the same shape). A box of the
  whole level is the same store as the level's extent, byte for byte.

A chunk of nothing but zeros is left out (zarr reads a missing chunk as the fill value,
0), unless asked to be written. A channel axis the view's ops add is the first axis and
one chunk, as served, whole whatever the extent. Progress counts the array's chunks done
(written or skipped) of its total, and the served chunks computed of those the box needs;
each only grows. As zarr v2 with OME-Zarr 0.4 (the default, which more tools read):

```text
.zgroup        {"zarr_format": 2}
.zattrs        {...extra attributes, "multiscales": [{"name": <view>, "axes": [...],
                "datasets": [{"path": "0", "coordinateTransformations": [scale, translation]}],
                "version": "0.4"}]}
0/.zarray      {"zarr_format": 2, "shape": <the level's, or the box's>, "chunks": <as served>, "dtype": "<u2",
                "compressor": {"id": "gzip", "level": 6}, "fill_value": 0, "order": "C",
                "filters": null, "dimension_separator": "/"}
0/.zattrs      {"_ARRAY_DIMENSIONS": ["z", "y", "x"]}
0/<i>/<j>/<k>  each chunk, gzipped
```

As zarr v3 with OME-Zarr 0.5:

```text
zarr.json        {"zarr_format": 3, "node_type": "group", "attributes": {...extra attributes,
                  "ome": {"version": "0.5", "multiscales": [{... "datasets": [{"path": "0", ...}]}]}}}
0/zarr.json      the level's array as served (the box's shape, with the box's extent),
                  its codecs [{"name": "bytes", ...}, {"name": "gzip", "configuration": {"level": 6}}]
0/c/<i>/<j>/<k>  each chunk, gzipped
```

The metadata is written first, so a store whose writing was stopped part way is still a
valid OME-Zarr, zero where no chunk was written. Extra attributes (a consumer's
provenance) sit beside the OME ones in the group's attributes, which they cannot replace.
gzip is what a browser compresses with on its own (`CompressionStream`; level 6 is the
level it writes at, which no reader needs); the Python server's default, blosc, would need
an encoder of its own in the page.

## Meshes, computed when fetched

The `mesh` frontend serves a dataset's surface in Neuroglancer's precomputed mesh formats,
for a segmentation layer whose source is `precomputed://.../<name>/mesh`, segment `1`.

**One resolution** (the default): the legacy format. `info`
(`{"@type": "neuroglancer_legacy_mesh"}`), the manifest `1:0` listing one fragment per chunk
of one level, and each fragment `1:0:i_j_k`, meshed when it is fetched (`uint32` vertex
count, `float32` x, y, z per vertex in nanometres, `uint32` triangle corners). A fragment is
its chunk plus one voxel on its high sides, so neighbouring fragments meet exactly (the
surface is watertight across them), and closed at the array's edges. Neuroglancer fetches
every fragment of the manifest, so the level bounds the work, and zooming in shows nothing
finer.

**Several resolutions** (`lods` above 1): the multi-resolution format, whose meshes get finer
where the viewer zooms in. Level of detail `i` is the pyramid level `level - lods + 1 + i`,
its octree nodes that level's chunks, so each finer level splits a node in eight. `info`
(`neuroglancer_multilod_draco`, 16-bit positions), the index `1.index` (per level of detail
its nodes and their sizes; each level's voxel size is its scale, the offset of its origin
its vertex offset) and the fragments' file `1`, which Neuroglancer reads by HTTP Range
requests, one node at a time, for the nodes in view at the detail it wants (a segmentation
layer's `meshRenderScale`: larger is coarser). The format wants every fragment's size before
any is fetched, which on-demand meshing cannot know, so every fragment is padded to one size
(128 KB; Draco decoders read what they need and ignore the rest, and the server gzips them,
so a padded fragment costs a few hundred bytes to a few tens of kilobytes on the wire). A
node's mesh is marching cubes on each of its octants apart (no triangle may cross one, so
Neuroglancer can show part of a coarse node while its children load), its vertices integers
across the node, coarsened past 100,000 triangles, and encoded with Draco (DracoPy, in the
`mesh` extra; the browser engine uses Draco's own WebAssembly build). Only nodes within two
voxels of the coarsest level's surface are listed, which the index is made from when it is
first asked for (the whole coarsest level is read once): a finer level's surface lies near
the coarser one's, and a listed node that turns out empty costs only its fetch. Four levels
of detail of the Mandelbulb from its 256³ level list about 47,000 nodes, a 750 KB index;
more than two million are refused.

The spec's `mesh` (CLI `--mesh`) says what is meshed: `kind: surface` (default) is the
boundary of the voxels at or above `threshold` (128), by marching cubes (scikit-image, the
`mesh` extra); `kind: terrain` is an elevation model (`y, x`, or `z, y, x` with one z) as
two triangles per cell, its elevation times `exaggeration` as height, cells with a NaN
corner left out (one resolution only). `level` picks the level, the coarsest with `lods`
(default: the finest whose longest side is at most 512 voxels), and `lods` the levels of
detail. The browser engine serves the same at `virtual/<page>/<view>/mesh`, its workers
running the same module (`chunkmirage.meshes`); for the multi-resolution index it masks the
coarsest level's eighths in its workers at once (each with a border, for its surface band),
and cuts the coarsest nodes' fragments from that mask (the Python server does the same), so
the coarsest level is computed once. The index has to be made from the whole coarsest level
before Neuroglancer can ask for any fragment: the Mandelbulb's appears after about 16 s in a
browser, against 8 s for its single-resolution mesh.

```bash
chunkmirage serve 'synthetic://mandelbulb?shape=268435456,268435456,268435456&voxel_size=1&unit=nm' \
  --mesh threshold=255,level=20,lods=4 --chunk 32,32,32 --python-viewer
```

## Data types

Precomputed supports `uint8`, `uint16`, `uint32`, `uint64`, `float32` only; add a `cast` op
for anything else. Precomputed volumes are typed by what their values are (`ArrayInfo.kind`,
which ops declare): labels and masks `segmentation`, images `image`; where nothing says,
`uint32`/`uint64` are `segmentation` and others `image`. Override with
`PrecomputedFrontend(volume_type=...)`.

Stored arrays (zarr, N5, precomputed, HDF5) say nothing of what their values are, so their
kind is guessed from the dtype: integers of 32 bits or more are labels, booleans masks, the
rest unknown. Labels and masks are resampled by nearest voxel and downsampled by their most
common value, never averaged, so no ids are invented between two segments. A spec's `kind`
(`open_source(..., kind=...)` in Python) overrides the guess: `"kind": "image"` for a uint32
image, `"label"` for ids stored as uint16.

## Sources

### Synthetic (procedural) sources

`synthetic://<kind>?shape=z,y,x&chunk=64,64,64&levels=N&seed=0&voxel_size=8&unit=nm`
generates data on the fly from voxel coordinates. Nothing is stored, so the volume can be
as large as you like, and each scale level is the same function sampled at a coarser
spacing, so the pyramid is exact (`s1[z,y,x] == s0[2z,2y,2x]`). Kinds: `blobs` (Gaussian
blobs), `shells` (hollow spheres, membrane-like), `noise` (fractal value noise), `julia`
(a 3-D slice of a quaternion Julia set), `mandelbulb` (the power-8 Mandelbulb, to zoom
into); combine with `+`, e.g. `blobs+noise`. Useful for demos and for stress-testing
pipelines without I/O. Generation is vectorised numpy, so the server's threadpool runs it
on all cores, and the browser engine's Pyodide workers run the same module for a view
whose source is `synthetic://` (nothing is read).

`mandelbulb` spans `[-1.25, 1.25]` on every axis in float64 from integer indices, so an
array 2^28 voxels across (21 levels, 10^25 voxels) still resolves its finest voxels. Its
value is 4 × the smooth escape iteration (255 inside), the same at every level, so a
colour stays a colour as the viewer changes level; and it is the one kind whose levels are
not exactly the same function: a level iterates `10 + 3 log2(256 / voxels across)` times,
more the finer it is, as fractal zoomers do, so zooming in resolves new detail.

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

### Warp sources (an image through a displacement field)

```
warp://<image>?field=<source>&affine=<12 numbers>&target=<source>
warp://<image>?field=<newest source>|...|<oldest source>&affine=<12 numbers>
warp://<image>?affine=<12 numbers>&target=<source>
warp://<image>?field=swirl&angle=90&radius=200&centre=z,y,x&plane=y,x
warp://<image>?field=swirls&count=8&seed=0&angle=90&radius=200
```

serves `<image>` (anything above or below, including `synthetic://` URLs) resampled
through a displacement field and an affine. The field is **any source**: a `(c, z, y, x)`
array whose three components are the z, y and x displacement, in the image's units, at
each voxel of the field's own grid, in the output's space. Another dataset of this server
([`dataset://<name>`](#dataset-sources-another-dataset-of-the-server-in-process): a block
matcher's lattice, a model's predicted deformation), a stored zarr, or a warp's own
`show=field`. A point `x` of the output is read from the image at `A(x + d(x))`, `A` the
`affine` (identity by default), the convention `register://` uses. The output is on the
image's own grid, or on `target`'s (any source: its levels are the output's, each reading
the image level that suits it through the mapping), so a pass fitted on a fixed image's
grid is applied there: `warp://moving?field=dataset://pass1&target=dataset://fixed`.

**Several fields**, `|` apart, are applied in turn, the first listed first, each sampled
where the ones before it have moved the point, and the image is read once. With
`T_i(x) = x + d_i(x)`, `field=d_n|...|d_1` reads the image at `A(T_1(...T_n(x)...))`; for
two fields that is `A(x + d_2(x) + d_1(x + d_2(x)))`, not a sum of the two vectors at `x`.
That is how the passes of a registration compose: the latest pass's field goes first,
since it was fitted on the image the earlier passes had already moved, so after two passes
`warp://moving?field=dataset://pass2|dataset://pass1&target=dataset://fixed` is the image
both passes correct. Warping the warped image instead
(`warp://warp://moving?field=...pass1...?field=...pass2...`) is the same mapping but
interpolates the image once per pass, which blurs it and the images later passes match on;
the one-resample form is sharper, measurably. **No field** with an `affine` or a `target`
places the image through the affine alone, on the target's grid: the first step of a
chain, before any pass has made a field, and the way to put two images on one grid for a
two-input op. Every field's keys are part of the output's, so an edit upstream of any of
them reaches it. A field or target URL with a query of its own is percent-encoded
(`urllib.parse.quote(url, safe="")`); `dataset://` names need none.

Or the field is `swirl` or `swirls`, deformations computed from coordinates: the image
twisted on its own grid, nothing stored, not even the field, so any parameter can change
at any time: `PUT` a new URL and the viewer refetches what is on screen. With
`&show=field` either kind serves the displacement itself, a `(c, z, y, x)` float32 volume;
of several fields, the net displacement `T_1(...T_n(x)...) - x`, the affine kept apart.
The resampling is the same as for scene sources, and each chunk reads only the compact
region its displaced voxels come from, however large the displacement.

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
| `field` | required unless `affine` or `target` is given | a source URL, or several `\|` apart applied in that order (displacements sampled from each, at its level nearest each output level's voxel size; a lattice's levels by the voxels they were fitted on, `ArrayInfo.computed_from`, so a zoomed-out view reads the lattice fitted at its level), `swirl` or `swirls` |
| `affine` | identity | applied after the displacement: 12 numbers, 3 rows of 4 in C order (z, y, x), or 16 with the last row `0 0 0 1`; 6 or 9 for a 2-D image |
| `target` | the image's grid | the source whose grid (levels, shape, voxel size, position) the output takes |
| `angle` | `90` | twist on the axis, degrees; it fades as `exp(-(r/radius)²)` with distance `r` from the axis (`swirl`) or the centre (`swirls`). Each of the `swirls` turns by ½ to 1 × `angle`, either way |
| `radius` | `swirl`: a quarter of the smaller in-plane extent; `swirls`: a fifth of the smallest extent | fall-off distance, image units; each of the `swirls` gets ½ to 1 × `radius` |
| `centre` | the volume centre | image units, C order: a point on the axis (`swirl`), the middle of the box the centres fall in (`swirls`) |
| `plane` | the last two axes, e.g. `y,x` | `swirl`: the two axes that turn |
| `count` | `8` | `swirls`: how many |
| `seed` | `0` | `swirls`: random seed for their centres, axes, radii and angles |
| `spread` | half the volume | `swirls`: half-size of the box around `centre` their centres fall in; one value or `z,y,x` |
| `show` | `image` | `field` serves the displacement instead |
| `frames` | none | the swirls only: frames on a leading `t` axis; frame `i` twists by `angle·i/(frames−1)`. An image's own `t` axis must hold one time point, which the frames replace |
| `chunk` | the grid's | output chunk shape of the spatial axes, C order; thin chunks such as `8,128,128` compute less for a view of one plane |
| `interpolation` | `linear`; `nearest` for uint32/uint64 | as for scene sources |

The parameters follow the last `?`, so the image URL may carry its own query. With
`serve --device cuda` (or `auto`; `chunkmirage.gpu.set_device` in Python) the resampler runs
on a GPU, here and for `scene://`: every output voxel through the affines and sampled fields
and the image interpolated, the resampler's own code run with CuPy for numpy and
`cupyx.scipy.ndimage` for `scipy.ndimage` (the `cupy` extra), in single precision, so values
agree with the CPU's to within a grey level and the cache keys are the same. Measured on a 184 M-voxel volume
through three lattice fields, CuPy took 0.5 s on an A100 where one CPU thread took 154 s
and 24 took 13.5 s, most of it sampling the fields. A chain with any other step (a swirl,
an inverse estimated numerically) is resampled on the CPU. To show the field as RGB, its components must be a Neuroglancer shader channel dimension (`c^`, read
with `getDataValue(0..2)`). The precomputed frontend makes them one, but holds only
`(c, z, y, x)`; with frames, serve zarr and rename the channel dimension `c'` to `c^`
(`Viewer.rename_dimensions`, as the demo does).

### Dataset sources (another dataset of the server, in-process)

```
dataset://<name>
```

reads another dataset this server serves, by name, as a source: its levels are that
dataset's own output stages, so a chunk comes through its stage cache, never over HTTP and
never computed twice, and every stage key downstream holds its keys. A chain of such
datasets is how passes of a computation are stacked without writing anything to disk
between them (each stage still cached): a field fitted by one pass, the moving image
warped by it, the next pass fitted on that
(`stack://dataset://fixed|dataset://corrected`), its field, and the original image warped
through both fields at once (`warp://dataset://moving?field=dataset://pass2|dataset://pass1`,
one resample; see [warp sources](#warp-sources-an-image-through-a-displacement-field)). Each
stage of the chain is cached once, and a change to one pass's parameters recomputes only
that pass and what follows it on screen.

Editing a dataset (`PUT`) rebuilds the datasets that read it, transitively, so their
digests, links and `change` events follow the edit; `GET /api/datasets/{name}` lists what a
dataset reads under `upstream`. A dataset may not read itself by any chain: a cycle is
refused when it is added (400). Removing a dataset leaves those reading it serving what
they last read of it, until they are edited. A `dataset://` URL resolves against the
registry building the dataset that holds it, so it is opened through the server (the REST
API, `create_app`'s datasets, a `DatasetRegistry`), not by `open_source` on its own.

### Coverage sources (what the cache holds of a dataset)

```
coverage://<name>
```

one voxel per chunk of another dataset of this server, saying what has been computed
there, to tell a real result from a preview while browsing: zoomed out, a dataset shows its
ops run on the coarse level, not a downsample of the full-resolution result, and nothing on
screen says whether that result exists. The levels are the dataset's. A voxel covers one
chunk of that level's last cached stage (the last op with `cache`; the blocks of an op with
`overlap`, where it ran; stage 0 when nothing after it is cached) and holds the sum of two
bits: 1 when that chunk is in the cache, 2 when every chunk of the finest level under it
is. So at full resolution a chunk is 3 or 0; zoomed out, 1 marks a preview (computed on the
coarse level, the fine chunks under it not all computed), 3 a region whose full-resolution
result exists. Served as labels (`kind` label), so a viewer colours the values apart. It is
read afresh at every request and never cached; a viewer that keeps what it fetched, as
Neuroglancer does, shows the state at the time, so reload the layer to see more.
`GET /api/datasets/{name}/coverage` gives the counts per level. A dataset none of whose
stages is cached (a `dataset://` source with uncached ops, or `cache_source: false`) has
no coverage, since its chunks are computed at each request: look at the datasets it reads.
As `dataset://` does, it resolves against the registry building the dataset that holds it,
and counts the dataset it describes as one it reads, so editing that rebuilds it.

### Register sources (deformable registration on a GPU)

```
register://<moving>?fixed=<fixed>&affine=<matrix.npy>
register://<moving>?fixed=<fixed>&affine=<matrix.npy>&show=pair
```

serves `<moving>` registered onto `<fixed>`'s grid, at every level of `<fixed>` and with
every channel of `<moving>`. Opening the source solves for a smooth displacement field `u`
in the fixed image's space such that the moving image sampled at `affine(p + u(p))` looks
like the fixed image at `p`. The solve reads a few coarse levels of both images, fits
their local normalized cross-correlation with a penalty on the field's gradient, and runs
Adam from coarse levels to fine ones, on a GPU when there is one (PyTorch:
`pip install chunkmirage[gpu]`; otherwise the CPU, slowly). Left out of the fit are
windows with no contrast in either image, and fixed voxels whose window the moving image
does not cover under the affine (a moving image with a smaller field of view): there is
nothing to match, and fitting them would drag the moving image's edge over whatever lies
beyond it, so the field there follows from its smoothness. The field lives on a control grid a few voxels
apart, so it is small (megabytes for an organ) and stays in memory. Every output chunk is
then the moving image resampled through it by the scene sources' resampler, so full
resolution is served without anything being written.

`u` has the convention registration pipelines write to disk (a displacement in fixed
space, applied before the fixed-to-moving affine, e.g. bigstream's), so it can be
compared with theirs directly (`show=field`), and the moving image's other channels come
out registered too. On a 1231×14124×5452 EASI-FISH gut round (an RTX 2080 Ti, the affine
from a keypoint fit, scored at level 3, finer than the solve): levels 6 and 5 solve in
2.4 s and beat a bigstream registration on local correlation and on the overlap of bright
voxels; levels 6 to 4, the default, take 15 s and match a cluster block-matching
registration's local correlation, 0.02 below it on overlap, its field within a median
13 µm of theirs.

Solved fields are remembered per process (the last 8), so opening the same URL again (a
downstream op edit, or another `show`) does not solve again, while changing a solver
parameter does. The solve runs in whatever opens the source (server start-up, or the
`PUT` that sets it), plus about 10 s the first time to import PyTorch from a network
file system.

With `refine=n`, the `n` levels below the solved ones are fitted where they are looked
at, not up front. Each gets its own control lattice, `grid` voxels apart, in blocks of
`block` voxels (64×128×128 by default, whatever the output's chunks), and a block is
fitted when a chunk it covers is first requested: from the
solved field over the block plus `halo` voxels of context, against the fixed and moving
voxels those reach, coarse to fine over halved copies of them (down to about the solved
level's resolution) for that level's `iterations` per copy; then it is kept in the
server's chunk cache with the computed chunks, so `--cache-gb` bounds it and
`DELETE /api/cache` clears it (1 GiB of its own when opened from Python without a cache). Every block starts from the solved field, so no level waits on
another's blocks. The finest fitted level's field serves every level below it. So the fit's
detail grows with the zoom and its cost with what is viewed rather than with the volume,
and whatever asks for chunks drives it: Neuroglancer, Fiji, webKnossos, a dask array.
Blocks are fitted separately, each over its context too, so neighbouring blocks overlap,
and they are blended there, as bigstream blends its blocks: a lattice point's value is the
average of the blocks whose fits cover it, each weighted 1 over its own block and less the
further into its context, so the field turns from one block's fit to the next's across the
overlap. Without it, neighbouring fits that disagree made the field step at their shared
edge: on the EASI-FISH pair below, by up to 0.9 µm per voxel, three times the steepest 1%
inside a block, enough to shift structures by a few voxels at a seam. Blended, the field
changes no faster across block edges than inside a block (0.17 to 0.21 µm per voxel at the
99th percentile, against 0.18 to 0.24 inside). A block's fit still depends only on the
solved field and the images, so the result does not depend on which blocks were fitted
first. The price is a ring of blocks: a chunk at a block's edge needs the block on the
other side too (zooming the browser page into the EASI-FISH pair, 66 blocks instead of 45,
and the view complete in 115 s instead of 98 s). On the tests' swirled volume the block
fit and a whole-image solve of the same levels agree to 2×10⁻³ in correlation with the
fixed image.

The window matters most at the fine levels. Two rounds of one EASI-FISH fly brain sit in a
public bucket that allows any origin (3.4 G voxels per channel, 0.23 µm), so this runs from
anywhere:

```bash
E=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119
chunkmirage serve "register://$E/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0?fixed=$E/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0&affine=auto&refine=3&iterations=100,40,40,40&window=15,31,31,31&show=pair" --https
```

`affine=auto` finds in 2 s how round 2 was remounted: turned over (z and x reversed, a half
turn about y) and tilted about 22° in the z-y plane. Level 3 (the default, 54 M voxels)
solves in 20 s on an RTX 2080 Ti, and levels 2 to 0 are fitted as you zoom. Whether round 2
is instead a mirror image (z alone reversed, as a stack taken the other way round would be)
the images cannot say: the brain is nearly symmetric, and a tilted mirror, given as the
affine, correlates as well (0.826 against 0.822), full-resolution chunks splitting between
the two. The half turn needs the smaller field afterwards (a median 0.2 µm against 0.5 µm),
so the search's rotation is kept; whoever knows the round was a mirror gives its affine.
Three full-resolution chunks in the tissue correlate with the fixed image at 0.75, 0.64 and
0.86 through the level-3 field, and at 0.85, 0.77 and 0.87 through blocks fitted with the
command above. Measured with blocks of 128³ voxels, the window matters most: 7 gives 0.79,
0.69 and 0.86, 15 gives 0.83, 0.75 and 0.87, 31 gives 0.86, 0.79 and 0.87 (a wider window
helps the level-3 solve too: 15 there alone gives 0.82, 0.73 and 0.86), while a larger halo
or more iterations changed nothing. Deeper blocks fit better, having room to go coarse to
fine: with the command's windows, blocks of 16×128×128 voxels reach 0.83, 0.75 and 0.87,
64×128×128 (the default) 0.85, 0.77 and 0.87, and 128³ 0.86, 0.79 and 0.87. The first
chunk of a region takes about 20 s, mostly its blocks (with the neighbours its
interpolation touches), then about 10 s a chunk, and blocks are kept, so a region fills in
faster the longer it is looked at.

`affine=auto` finds the starting affine from the images (the coarsest solved level of
each, halved to a few hundred thousand voxels): their intensity moments are matched, centre
to centre and principal axis to principal axis, the best-correlated of the orientations
that do not mirror the image is kept, and the 12
numbers are then fitted by gradient ascent
on the normalized cross-correlation at two resolutions, as the browser page does. The
moments assume both images show the same whole object; a crop of one needs an affine given.
A mirror image is not searched for: on a nearly symmetric specimen a mirror correlates as
well as the right rotation while swapping left and right (on the fly templates exactly as
well, 0.872, and 226 µm from the published affine), and a fit cannot reach one from a
rotation (it would pass through a matrix that flattens the volume). An affine with a
negative determinant, such as z reversed, is used as given. A found affine is
remembered per image pair, like the solves.

`show=pair` serves a `(c, z, y, x)` volume whose two channels are the fixed image's
matched channel and the registered moving image's, so one Neuroglancer shader can compare
them on the viewer's GPU (overlay, checkerboard, fade, difference) with no refetch. The
source carries that shader, so the link `chunkmirage serve` prints, the REST API's
`neuroglancer` routes and the python viewer show the pair in colour without being asked:
fixed magenta, registered green, white where they agree, with contrast from the coarsest
solved level and a `mode` control for the other three comparisons. `show=field` likewise
comes as a heat map of how far each point moved (a `direction` box colours by direction).
`examples/register_demo.py` shows that next to the affine alone (`iterations=0`), and
re-solves when you type new settings, the viewer keeping its camera.

| parameter | default | meaning |
| --------- | ------- | ------- |
| `fixed` | (required) | the fixed image: anything `open_source` reads, with the same spatial units as `<moving>`; percent-encode it if it has a query |
| `affine` | identity | fixed-to-moving affine in physical units, C order: a `.npy` or text file with a 4×4 or 3×4 matrix, or its 12 or 16 values inline, row by row; `auto` finds one from the images |
| `fixed_channel`, `moving_channel` | `0` | the channel each image is matched on, for images with a `c` axis |
| `levels` | from the coarsest with ≥ 16 voxels on every axis to the finest with ≤ 2²⁵ | fixed-image levels to solve on, coarse to fine, e.g. `6,5,4`; each is matched with the moving level nearest its voxel size |
| `iterations` | `100` | Adam steps per level: one value, one per level, or one per level with the refined ones; `0` leaves the affine alone (no GPU needed) |
| `refine` | `0` | levels below the solved ones fitted block by block where they are viewed: `refine=3` with `levels=6,5,4` fits levels 3, 2 and 1 |
| `halo` | `8` | voxels of context on every side of a block being fitted: how far beyond it the fit looks |
| `block` | `64,128,128` | voxels per refined block, C order, independent of the output's chunks; deeper blocks fit better (coarse to fine within them), thinner ones answer a single slice sooner |
| `smooth` | `1` | weight of the penalty on the field's gradient |
| `grid` | `4` | control-point spacing, voxels of each level |
| `window` | `7` | correlation window, voxels (odd): one value, one per level, or one per level with the refined ones; fine levels gain from a wider one (see below) |
| `show` | `image` | `pair` (fixed and registered as two channels) or `field` (`u`, components first, physical units) |
| `frames` | none | a leading `t` axis of that many snapshots of the solve, from the affine alone to the final field: play `t` to watch it converge. The moving image's own `t` axis must hold one time point, which the frames replace |
| `chunk` | the fixed image's | output chunk shape of the spatial axes, C order |
| `interpolation` | `linear`; `nearest` for uint32/uint64 | as for scene sources |
| `device` | `auto` | `auto` is the GPU with the most free memory, else the CPU; or `cpu`, `cuda:1`, ... |

The parameters follow the last `?`, so the moving image's URL may carry its own query (a
`warp://` URL, say, which is how the tests check that a known swirl is undone). They are
one Pydantic model, `RegisterParams`, whose JSON Schema (`chunkmirage schema`) the browser
engine's types and form defaults are generated from: the
[browser page](https://janeliascicomp.github.io/chunkmirage/browser/register.html) solves the
same spec on the viewer's GPU and shows the `chunkmirage serve 'register://…'` command for
its settings, and fed the same affine and levels the two fields agree to 0.01 µm (median)
on the fly templates, whose field moves tissue by 7 µm (median). It fits blocks on demand
(`refine`) the same way, on the viewer's GPU, so the EASI-FISH pair above runs from a link
with nothing installed:
[register.html with the EASI-FISH rounds](https://janeliascicomp.github.io/chunkmirage/browser/register.html?fixed=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0&moving=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0&refine=3&iterations=100,40,40,40&window=15,31,31,31)
(it shows the two rounds; Register solves).

### Stack sources (several images as one array's channels)

```
stack://<image>|<image>[|<image>...]
```

serves images that share a grid as the channels of one `(c, z, y, x)` array (`(c, y, x)` for
images in y and x, such as GeoTIFF bands): channel 0 is
the first image's first channel, channel 1 the second's, and so on. The images must agree
level by level on shape, voxel size, translation and units (the stack has as many levels as
the image with the fewest), and each is anything `open_source` reads, so a stack can hold a
stored volume next to a `synthetic://` or `register://` one. Nothing is copied: a stack chunk
reads the same box of each image, through that image's own cache.

A stack exists for ops that need two images at once. Such an op takes the channel axis and
returns an array without it; the pipeline reads every channel and pads only the spatial axes
by the op's halo, and `--chunk` may then give just the three spatial values. The first such
op is [`contacts`](../reference/ops.md): the voxels within a distance of both structures.
On OpenOrganelle's published organelle predictions, mirrored back into the EM's frame (see
below), followed by `label` to colour and size-filter the sites:

```
P=https://janelia-cosem-datasets.s3.amazonaws.com/jrc_hela-2/jrc_hela-2.n5/labels
chunkmirage serve "stack://flip://$P/mito_pred?axes=y|flip://$P/er_pred?axes=y" \
    --op contacts:distance=12 --op label:min_size=50 --chunk 16,128,128 --python-viewer
```

`examples/contact_sites.py` serves the same with the EM underneath and the two predictions
tinted, opens where the two organelles touch most, and takes new settings at a prompt.
Only the chunks on screen are computed, out of 122 gigavoxels of cell, and a change of
distance recomputes just those, from predictions the first pass left in the cache. The
`distance` is in nanometres and each level counts it in its own voxels, so a contact means
the same at every zoom (`radius`, in voxels, would reach proportionally further on a
zoomed-out level, as op parameters in voxels do).

### Flip sources (images stored the other way round)

```
flip://<image>?axes=y[,x,...]
```

serves `<image>` mirrored along the named axes, every level in place: voxel `i` of a
flipped axis of length `n` is voxel `n − 1 − i` of the image, and shape, voxel size,
translation and chunks stay the image's. It repairs data whose orientation its metadata
gets wrong: OpenOrganelle's N5 organelle predictions of jrc_hela-2 (`labels/*_pred`) are
upside down in y relative to its EM, zarr and N5 alike (with the flip, every predicted
mitochondrion voxel of a coarse slice lies on cell; without it a fifth do), though nothing
in their attributes says so. The levels must share their centre, as OME-Zarr and COSEM
pyramids whose extents halve evenly do, since a level mirrored about its own centre would
otherwise drift from the others; `flip://` refuses a pyramid whose levels share their
corner instead (as `synthetic://` levels do).

### Stitch sources (tiles stitched by interest points and RANSAC)

```
stitch://<BigStitcher project.xml>?channel=0&model=translation&epsilon=5&threshold=0.005
```

stitches the tiles of one channel of a BigStitcher project when opened, as BigStitcher's
interest-point registration does, then serves them fused, region by region: no fused copy
is written. Each overlap of two tiles' stage positions, grown by `margin`, is read at the
tiles' level `level`, and its bright blobs found in both tiles: local maxima of a
difference of Gaussians (`sigma`, and 2^(1/4) times it, scaled to the same physical size on
every axis) at least `threshold` of the region's intensity range, located to a fraction of
a voxel. A point's descriptor is the offsets to `neighbors` of its `neighbors + redundancy`
nearest neighbours (every such subset), which a shift leaves unchanged; two points match
when their descriptors are the closest pair both ways and `significance` times closer than
the next candidate. RANSAC then draws `iterations` minimal samples of the matches, keeps the
`model` (`translation`, `rigid` or `affine`) that most matches agree with to within
`epsilon`, refits it to them until they settle, and keeps the pair if it has `min_inliers`
inliers and `min_inlier_ratio` of its matches. Finally every tile's correction is fitted to
all the kept matches at once (each round refits each tile to where its neighbours put their
ends of its matches), one tile of each linked group held still.

The fused volume has a level for each level every tile has, the first tile's voxel size
there, covering every placed tile. A region of it is each tile that reaches it, sampled
trilinearly where its placement puts it, averaged with weights that fall off as a half
cosine over `blend` at the tiles' edges in y and x, so the seams do not show. The project's
images must be OME-Zarr (BigStitcher-Spark's `bdv.multimg.zarr` loader); a tile's stage
position is its view transforms but the outermost ones named "Stitching Transform" (a
stitching done before), and the log says how far the result lies from that one. The steps
are functions of arrays in `chunkmirage_contrib.stitching` (numpy and scipy, so the `ops` extra), the
same code the browser's stitch page runs in Pyodide.

```bash
chunkmirage serve 'stitch://https://janelia-bigstitcher-spark.s3.amazonaws.com/Stitching/dataset.xml' --python-viewer
```

On BigStitcher-Spark's example (a larval fly CNS in 2 × 3 tiles), the defaults keep 8 of
its 11 overlaps and put the tiles 0.4 µm (RMS) from BigStitcher's own phase-correlation
stitching; opening takes about 9 s, most of it reading the overlaps.

### GeoTIFF sources (cloud-optimized GeoTIFFs)

```text
https://.../site.tif        (s3://, gs:// or a local path too; .tif or .tiff)
```

The tiled GeoTIFFs most geospatial and planetary rasters are published as, cloud-optimized
ones (COGs) above all, read where they are looked at: the header once (a few range
requests), then each tile a read covers, fetched by range, decoded by tifffile and kept in a
256 MB cache. The pages of decreasing size inside a COG (its overviews) are the levels, so
a zoomed-out view reads the small ones; masks are skipped. One sample per pixel; the axes
are `y, x` in the projection's units (metres, unitless for a geographic raster in degrees),
`y` counting down the image as rows do (minus the northing), voxel `(0, 0)` at the tiepoint
(`ModelTiepoint`, `ModelPixelScale`). A float raster's `GDAL_NODATA` reads as NaN. Strip
TIFFs (untiled) are refused: every read would fetch whole rows. Needs the `tiff` extra
(tensorstore's own `tiff` driver reads only uint8 images). The browser engine reads them
with geotiff.js, the same way, as `z, y, x` with one z.

```bash
chunkmirage serve 'https://astrogeo-ard.s3.us-west-2.amazonaws.com/moon/lro/lola/barker_south_pole_dems/Site01/Site01.tif' \
  --op hillshade:azimuth=135,altitude=10 --chunk 256,256 --python-viewer
```

### Stored sources

Sources are detected by content, not extension: `zarr.json` → zarr v3, `.zarray` → zarr v2,
`attributes.json` → N5, `info` → precomputed. A group's levels are the paths in its
`multiscales[0].datasets` (OME-NGFF, COSEM N5), in that order; without that, `s0, s1, ...`
are probed. N5 and precomputed data are transposed to C order `(z, y, x)` on read.

Remote stores are read the way a public dataset most likely lets one in, with credentials
only when a public read is refused. `s3://` is read anonymously first, then with AWS's
default credentials (environment, `~/.aws`, instance metadata), since S3 refuses even a
public read signed with a stale or foreign key. `gs://` is read with Google's default
credentials (anonymously without any), then through the bucket's public
`https://storage.googleapis.com` URL, for credentials that are there but broken. Which way
worked is remembered per bucket. Metadata reads give up after about five seconds of
retries rather than tensorstore's default of many minutes, so a mistyped host fails fast.
A zarr v2 compressor with members tensorstore rejects, such as the `checksum` newer
numcodecs writes for zstd, is read without them.

Voxel size, translation, units and axes are read per level, first match wins:

| Format      | Metadata, in order of precedence                                                   |
|-------------|------------------------------------------------------------------------------------|
| zarr v2/v3  | parent OME-NGFF `multiscales` entry whose `path` is this array, composed with the multiscale-level `coordinateTransformations` if present (0.4/0.5; in 0.6 those lead to other coordinate systems and are applied only through a [`scene://`](#scene-sources-ome-zarr-06-transformations) source, and axes come from the intrinsic coordinate system); else, for an array xarray wrote (geo, climate and solar data), its dimension names (`_ARRAY_DIMENSIONS`, or zarr v3 `dimension_names`) as the axes and the 1-D coordinate array named after each dimension as its spacing and origin, where evenly spaced and increasing (`days since …` and other CF time units become seconds, degrees unitless); else the array's own `resolution`/`voxel_size`, `offset`, `units`, `axis_names` (funlib) or `transform` (COSEM), C order |
| N5          | the array's `transform` (COSEM, C order); the parent's `multiscales[].datasets[].transform` for this path; `pixelResolution`/`resolution` (array, else group) × `downsamplingFactors`, plus `offset`, x-first |
| precomputed | `resolution` (nm) and `voxel_offset` (the first voxel's corner, in voxels)         |
| HDF5        | `resolution`/`voxel_size` and `offset` attributes, C order                         |

A zarr array with CF packing attributes (`scale_factor`, `add_offset`) is read as float32
in its units, `stored × scale_factor + add_offset`, and its `_FillValue` (else the array's
fill value) as NaN: NASA's MUR sea temperature, stored as int16 hundredths of a degree
offset from 298.15 K, reads as kelvin with land NaN. Arrays without those attributes are
read as stored. Data whose axes are not named `z, y, x` (`time, lat, lon`) keeps its own
names: the OME frontends type a time axis as `time` (by name or a time unit), and the
viewer helpers show its last three axes as x, y and z.

Unitless `z, y, x` axes are served as nanometres by every frontend (left without a unit,
Neuroglancer would read OME axes as metres). Neuroglancer counts zoom
(`crossSectionScale`) in the smallest scale among the viewer's dimensions, whatever their
units, so beside a time axis in seconds a zoom of 1 is not one pixel per voxel:
`chunkmirage.neuroglancer.cross_section_scale(dims, axis, voxels_per_pixel)` converts.

`offset`/`translate` are in world units. A translation is where voxel 0's centre is, as in
OME-Zarr, COSEM's `transform` and CF coordinates. Formats that give voxel 0's corner are
moved half a voxel along their spatial axes on reading: precomputed's `voxel_offset`, and
funlib's `offset` (0 when only a `resolution` is given), which is how Neuroglancer and
funlib place them. N5 levels with `pixelResolution` and `downsamplingFactors` and no
offset follow BigDataViewer: a downsampled voxel's centre is the centre of the block it
covers. Served as precomputed, `voxel_offset` is the translation less half a voxel,
rounded to whole voxels, so a precomputed source served again keeps its offset. Served as
OME-Zarr, the translation is written as is, and Neuroglancer moves it back half a voxel to
the corner. So a volume shows in the same place whichever format serves it, except that
precomputed cannot hold half-voxel offsets. Anything in the spec (`voxel_size`, `units`,
`axes`, `translation`) overrides what was read. The spec's `select` pins non-spatial axes
to one index each, `{"c": 1, "t": 0}` (`--select c=1,t=0`), so the ops see one channel of
one time point as a `z, y, x` volume; only that channel is read. HDF5 uses `file.h5::/dataset` and needs
the `hdf5` extra.

The browser engine reads the same stored formats in TypeScript (`web/src/sources.ts`), all
but HDF5, and places them as above: OME-Zarr v2/v3, N5 (gzip or raw), xarray-style zarr
arrays, and Neuroglancer precomputed volumes, each scale a level, unsharded or sharded
(`neuroglancer_uint64_sharded_v1` with the `identity` hash), encoded `raw`, `jpeg` (a
chunk's z planes one tall image, decoded by the browser) or `compressed_segmentation`. A
precomputed URL may keep Neuroglancer's `precomputed://`. `gs://` and `s3://` URLs are
fetched as Neuroglancer fetches them: `s3://` at the bucket's `https` URL, which S3 serves
with CORS for a public bucket; `gs://` through the Cloud Storage JSON API
(`storage.googleapis.com/storage/v1/b/<bucket>/o/<object>?alt=media`, the object's name
encoded as one piece), because the bucket's own URL answers a page with no CORS headers
unless its owner set them, and most have not. The service worker's store cache sees the
API's URLs, and tells metadata from chunks by the decoded name.

### Your own schemes (sources from other packages)

A package adds a URL scheme of its own the way it adds an op: an opener that takes the URL
and returns a `MultiscaleSource` (a list of levels, each a `Source` with an `info` and a
`read(box)`), listed under the `chunkmirage.sources` entry point with the scheme as its
name. `open_source` and so `chunkmirage serve` then open `myscheme://...` like the built-in
schemes, and pipelines, caching and every frontend work on it unchanged. An opener is passed
`cache_bytes` (tensorstore's pool) and `cache` (the pipeline's chunk cache) if it takes
them. `chunkmirage.sources.register_source(scheme, opener)` does the same in a running
process, and `schemes()` lists what is known. The built-in schemes cannot be replaced. A
format told apart by its path rather than a scheme (a GeoTIFF by its `.tif`) lists a
*matcher* instead, `path -> opener or None`, under `chunkmirage.source_matchers` (or
`register_matcher`); `open_source` asks each in turn for a path with no known scheme before
trying tensorstore. The sources shipped with chunkmirage that compute (`synthetic://`,
`register://`, `stitch://`, GeoTIFFs) are registered this way too, from `chunkmirage_contrib`
([what stays in the core](../design.md#what-stays-in-the-core)).

```toml
[project.entry-points."chunkmirage.sources"]
myscheme = "mypackage.sources:open_myscheme"

[project.entry-points."chunkmirage.source_matchers"]
myformat = "mypackage.sources:match"   # def match(path) -> opener | None
```

A source suits work that reads the data itself rather than a padded block: a model wrapper
that opens its own input, a format with its own reader, data computed from coordinates.
