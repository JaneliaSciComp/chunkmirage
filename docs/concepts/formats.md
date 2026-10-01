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
[browser page](https://yuriyzubov.github.io/chunkmirage/browser/register.html) solves the
same spec on the viewer's GPU and shows the `chunkmirage serve 'register://…'` command for
its settings, and fed the same affine and levels the two fields agree to 0.01 µm (median)
on the fly templates, whose field moves tissue by 7 µm (median). It fits blocks on demand
(`refine`) the same way, on the viewer's GPU, so the EASI-FISH pair above runs from a link
with nothing installed:
[register.html with the EASI-FISH rounds](https://yuriyzubov.github.io/chunkmirage/browser/register.html?fixed=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_1_1_SS00090_Spab_546_Nplp1_647_1x_Central.zarr/0&moving=https://janelia-data-examples.s3.amazonaws.com/fly-efish/NP31_R2_20240119/NP31_R2_2_1_SS00090_FMRFa_546_Proc_647_1x_Central.zarr/0&refine=3&iterations=100,40,40,40&window=15,31,31,31)
(it shows the two rounds; Register solves).

### Stack sources (several images as one array's channels)

```
stack://<image>|<image>[|<image>...]
```

serves images that share a grid as the channels of one `(c, z, y, x)` array: channel 0 is
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
    --op contacts:radius=3 --op label:min_size=50 --chunk 16,128,128 --python-viewer
```

`examples/contact_sites.py` serves the same with the EM underneath and the two predictions
tinted, opens where the two organelles touch most, and takes new settings at a prompt.
Only the chunks on screen are computed, out of 122 gigavoxels of cell, and a change of
radius recomputes just those, from predictions the first pass left in the cache. Like every
op's parameters, `radius` counts voxels of the level being served, so a zoomed-out view
reaches proportionally further.

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

### Stored sources

Sources are detected by content, not extension: `zarr.json` → zarr v3, `.zarray` → zarr v2,
`attributes.json` → N5, `info` → precomputed. A group's levels are the paths in its
`multiscales[0].datasets` (OME-NGFF, COSEM N5), in that order; without that, `s0, s1, ...`
are probed. N5 and precomputed data are transposed to C order `(z, y, x)` on read.

Voxel size, translation, units and axes are read per level, first match wins:

| Format      | Metadata, in order of precedence                                                   |
|-------------|------------------------------------------------------------------------------------|
| zarr v2/v3  | parent OME-NGFF `multiscales` entry whose `path` is this array, composed with the multiscale-level `coordinateTransformations` if present (0.4/0.5; in 0.6 those lead to other coordinate systems and are applied only through a [`scene://`](#scene-sources-ome-zarr-06-transformations) source, and axes come from the intrinsic coordinate system); else, for an array xarray wrote (geo, climate and solar data), its dimension names (`_ARRAY_DIMENSIONS`, or zarr v3 `dimension_names`) as the axes and the 1-D coordinate array named after each dimension as its spacing and origin, where evenly spaced and increasing (`days since …` and other CF time units become seconds, degrees unitless); else the array's own `resolution`/`voxel_size`, `offset`, `units`, `axis_names` (funlib) or `transform` (COSEM), C order |
| N5          | the array's `transform` (COSEM, C order); the parent's `multiscales[].datasets[].transform` for this path; `pixelResolution`/`resolution` (array, else group) × `downsamplingFactors`, plus `offset`, x-first |
| precomputed | `resolution` (nm) and `voxel_offset` × `resolution` as the translation             |
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

`offset`/`translate` are in world units. Anything in the spec (`voxel_size`, `units`,
`axes`, `translation`) overrides what was read. The spec's `select` pins non-spatial axes
to one index each, `{"c": 1, "t": 0}` (`--select c=1,t=0`), so the ops see one channel of
one time point as a `z, y, x` volume; only that channel is read. HDF5 uses `file.h5::/dataset` and needs
the `hdf5` extra.
