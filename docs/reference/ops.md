# Ops reference

Run `chunkmirage ops` or `GET /api/ops` for the live list with JSON schemas. This page must
list every registered op; `tests/test_docs.py` enforces it.

Ops that make masks (`threshold`, `morphology`, `contacts`) and labels (`label`, `spots`)
say so (`output_kind`), and viewers show what they make as segmentations.

Every parameter carries a description (pydantic `Field(description=...)`) that the control
page shows under the control and `GET /api/ops` returns in the JSON schema. Add one to any
new op; `tests/test_docs.py` requires it.

| op          | parameter   | default | meaning |
| ----------- | ----------- | ------- | ------- |
| `threshold` | `low`       | 0       | lower bound (inclusive), in the source's intensity units |
|             | `high`      | none    | upper bound (exclusive); leave empty for no upper bound |
|             | `value`     | 1       | label written for passing voxels; output is uint8 |
| `cast`      | `dtype`     | uint8   | target numpy dtype name |
|             | `clip`      | true    | clip to the integer range first to avoid wrap-around |
| `scale`     | `factor`    | 1.0     | multiply (contrast); output float32 |
|             | `offset`    | 0.0     | then add (brightness) |
| `gaussian`  | `sigma`     | 1.0     | blur width in voxels; halo = `ceil(sigma × truncate)` |
|             | `truncate`  | 3.0     | kernel radius in sigmas; rarely changed |
| `uniform`   | `size`      | 3       | edge of the averaging cube in voxels; halo = `size // 2 + 1` |
| `dog`       | `sigma`     | 2.0     | difference of Gaussians: enhances blobs of about this size; halo = `ceil(3 × sigma × ratio)` |
|             | `ratio`     | 1.6     | larger blur = `sigma × ratio` |
|             | `gain`      | 4.0     | scales the difference into 0..255 (output uint8, 128 = zero) |
| `diff`      | `axis`      | 0       | change along one axis: each voxel minus the one `lag` steps before it on `axis` (0: time in a `t, y, x` series); output float32; halo = `lag` on that axis only; the first `lag` steps compare against the first |
|             | `lag`       | 1       | how many steps back to compare with |
| `gradient`  | `axes`      | last three | rate of change along each axis listed (counted from the first: `[1, 2]` for latitude and longitude of a `time, lat, lon` series), per unit of the axes (the level's voxel size): central differences, one channel each on a new leading `c` axis; output float32; halo = 1 on those axes, and only the interior is returned (a valid convolution) |
| `downsample`| `factor`    | 2, 2, 2 | voxels per output voxel along each of the last axes; the grid changes with it (voxels `factor` times bigger, the shape divided and rounded up, each voxel's position the centre of its block); how a pipeline makes the coarser levels of an op with an input voxel size |
|             | `mode`      | auto    | `mean` of each block (integers rounded), `mode` (its most common value), or `auto`: mode for labels and masks, mean otherwise |
| `morphology`| `operation` | open    | `open`, `close`, `erode`, `dilate` on a mask (input > 0); output uint8 |
|             | `radius`    | 2       | spherical structuring element radius in voxels; halo = `2 × radius + 1` |
| `label`     | `min_size`  | 0       | connected components of a mask, output uint32 segment ids; drop components smaller than this |
|             | `connectivity` | 1    | 1 = 6-connected, 2 = 18, 3 = 26 |
| `spots`     | `sigma`     | 1.0     | bright diffraction-limited spots (single mRNAs in smFISH, EASI-FISH): difference of Gaussians, local maxima above `threshold`, each drawn as a small ball whose uint32 id comes from its position, so it is the same whichever chunk finds it; spot size in y-x voxels; needs a `z, y, x` volume (`select` a channel) |
|             | `sigma_z`   | 0.6     | spot size in z voxels |
|             | `threshold` | 10      | least difference-of-Gaussians response, image intensity units |
|             | `separation`| 2       | spots closer than this (y-x voxels) are one |
|             | `radius`    | 1       | ball drawn per spot, y-x voxels (0 marks one voxel) |
| `slope`     | `z_factor`  | 1.0     | slope of an elevation model in degrees (0 flat, 90 a cliff), on the last two axes, from each level's pixel spacing (`for_level`); output float32; halo 1; elevation units per unit of the spacing (1 when both are metres) |
| `hillshade` | `azimuth`   | 315     | shaded relief, the terrain lit by a distant sun from this direction (degrees clockwise from the top of the image), local illumination only (no cast shadows); output uint8, 1 unlit to 255 facing the sun, 0 where the elevation is NaN; halo 1 |
|             | `altitude`  | 45      | the sun's height above the horizon, degrees |
|             | `z_factor`  | 1.0     | as for `slope`; above 1 exaggerates relief |
| `contacts`  | `radius`    | 3.0     | contact sites between the first two channels of a `stack://` source: voxels within this many voxels (Euclidean) of both structures; output uint8 mask; halo = `radius + 1` |
|             | `distance`  | none    | the reach in the data's units (nm) instead: each level counts it in its own voxels, so a contact means the same at every zoom; halo planned on the finest level |
|             | `a_low`     | 128     | values at or above this in the first channel are the first structure (128 for a uint8 probability map, 1 for a segmentation) |
|             | `b_low`     | 128     | the same for the second channel |

`label` numbers components **per chunk** (salted by chunk position so ids never collide).
An object spanning chunks therefore gets one colour per chunk. That is the honest per-chunk
preview of a global operation; see [FAQ](../faq.md#where-does-it-fall-short).

Ops that need to know *where* a block sits override `apply_at(block, box)` instead of
`apply(block)`; `label` uses it for the salt, `spots` for its ids.

`examples/fish_spots.py` runs `spots` on both FISH channels of a public EASI-FISH round of
a whole fly central brain (3.4 gigavoxels per channel), next to the raw channels, with the
threshold editable at a prompt. Only the chunks on screen are searched, and a chunk's spots
are identical to those found in one pass over a larger block.

Ops over several images take the channels of a
[`stack://` source](../concepts/formats.md#stack-sources-several-images-as-one-arrays-channels)
and return an array without the channel axis, as `contacts` does (its `output_info` drops the
axis). The pipeline reads every channel and pads only the spatial axes by the halo; ops after
it in the same stage see the plain spatial block.

None of the built-ins cache their output by default (`cache=False`); a pipeline turns it on
for one op with `"cache": true` in its spec ([caching](../concepts/caching.md#why-not-cache-every-stage)).
Everything except `threshold`,
`cast`, `scale`, `diff`, `slope` and `hillshade` needs the `ops` extra (scipy).

`slope` on NASA's 5 m south-pole elevation of the Moon (the ridge between Shackleton and de
Gerlache craters, LOLA, Barker et al.) equals the slope USGS publishes with it, to 0.0°
(median and 95th percentile over a 512 × 512 region at full resolution); the gallery's Moon
card draws ground under a chosen slope over it, and the relief lit by a sun you move.

`diff` along time is the view a hurricane's cold wake or a solar flare shows up in: each
day's sea temperature minus the day before's (`examples/hurricane_wakes.py`, and the
gallery's hurricanes card), each 6-minute image of the sun minus the one before
(`examples/solar_flares.py`, the running difference solar physicists use).

## CLI syntax

```
--op threshold:low=120,high=200
--op '{"op": "gaussian", "sigma": 2}'
--op gaussian:sigma=2,cache=true
```

Values are parsed as JSON where possible, else strings. Repeat `--op` to chain.
