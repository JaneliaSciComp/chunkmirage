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
| `gradient`  | `axes`      | last three | rate of change along each axis listed (counted from the first: `[1, 2]` for latitude and longitude of a `time, lat, lon` series), per unit of the axes (the level's voxel size): central differences, one channel each on a new leading `c` axis; output float32; halo = 1 + `ceil(3 × sigma)` on those axes, and only the interior is returned (a valid convolution) |
|             | `sigma`     | 0       | smoothing first, in voxels along those axes (a Gaussian derivative), so noise a voxel or two across does not point every voxel its own way; 0 = none |
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
| `normalized_difference` | `pair` | 0, 1 | `(a - b) / (a + b)` of two channels of a `stack://` source, float32: the indices remote sensing reads plants, water and burn scars from (NDVI, NDWI, NBR); NaN where a band is zero or less |
|             | `minus`     | none    | two more channels whose index is subtracted from the first pair's: a change between dates (burn severity, dNBR, is the NBR before minus the NBR after) |
|             | `offset`    | 0       | added to every channel first, to make reflectances of stored numbers |
|             | `nodata`    | none    | a stored value meaning no data: the index there is NaN |
|             | `floor`     | 0       | where a pair sums to less than this, its index is noise (water) and NaN |
| `python`    | `code`      | a template | Python of your own defining `apply(block, ...)`, with `#halo`, `#output` and `#uicontrol` lines; see [below](#your-own-python-the-python-op). A server runs it only when started with `--allow-code` |
|             | `params`    | none    | values of the code's `#uicontrol` sliders, over their defaults |

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

## Your own Python: the `python` op

The `python` op carries its code in the pipeline spec, so a pipeline can do something no op
does without a plugin package: edit the code and the view is computed again, as
Neuroglancer's shaders are edited live on its pixels. The control page and the browser demos
show it as a code box.

```python
#halo 1
#output uint32 label
#uicontrol float low slider(min=0, max=255, default=128)
def apply(block, low):
    from scipy import ndimage
    return ndimage.label(block > low)[0]
```

`apply` gets the block, padded by the halo, and the values of the sliders by name, and
returns the block computed (or its interior, shaved by the halo on every side). numpy is
`np`. A parameter named `box` gets where the block sits, as `apply_at` does. The result is
cast to the output dtype. What it prints shows under the code in the browser demos, by
level and chunk, and in a server's log. Three kinds of line, read as text so a pipeline is
planned without running the code:

| line | says | default |
| ---- | ---- | ------- |
| `#halo 2`, `#halo 0 4 4` | voxels of context needed on each side: one for every axis, or one per axis for the last axes | 0 |
| `#output uint32 label` | the dtype returned (an integer or float numpy dtype), and/or whether it is an `image`, `label` or `mask` | the input's |
| `#uicontrol float low slider(min=0, max=255, step=1, default=128)` | a slider, in Neuroglancer's shader syntax: `float` or `int` `slider` (min and max required), or `bool` `checkbox(default=true)`; `apply` takes a parameter of the same name | `default`: `min`, or false |

The spec gives the sliders' values as `params`, so moving one changes the parameters, not
the code: `{"op": "python", "code": "...", "params": {"low": 140}}`. A value for a slider
the code no longer has is dropped.

**How a slider reaches the code.** The `#` lines are comments to Python itself, and `apply`
is an ordinary function: outside chunkmirage, `apply(volume, sigma=1, low=300, min_size=500)`
runs it on any numpy array. chunkmirage reads the `#uicontrol` lines to draw the sliders
(the browser demos and the control page draw the same ones), keeps each slider's value in
the op's `params`, and calls `apply(block, **params)` on every chunk; a moved slider is a
new spec, so the view is computed again. Nothing in the code belongs to a viewer or a page:
from Python, `Pipeline.from_spec({"source": ..., "ops": [{"op": "python", "code": code,
"params": {"low": 400}}]})` runs it the same way. Both are part of the
op's digest, so an edit is a new URL and viewers fetch the result afresh. Mistakes (a
syntax error, an unknown dtype, a slider `apply` does not take) are refused when the
pipeline is built, naming the line.

**Where it runs.** In the browser engine, always: the code runs in the visitor's own
Pyodide workers, which load what it imports (numpy, scipy, scikit-image and the rest of
Pyodide's packages; not torch, cupy or tensorstore). On a server, the code runs as the
server's user, with its files, so `chunkmirage serve` refuses specs with `python` ops sent to
`/api` unless started with `--allow-code`, which then needs a token
([CLI](cli.md#serve)). Pipelines built in your own process (`Pipeline`, `--op`) run it
regardless.

**As a plugin.** `chunkmirage.ops.python_op(name, code)` makes an op class of the same code
whose sliders are its parameters (pydantic fields, with their bounds), for the
`chunkmirage.ops` entry point; the code boxes' "copy as plugin" gives such a module.

## CLI syntax

```
--op threshold:low=120,high=200
--op '{"op": "gaussian", "sigma": 2}'
--op gaussian:sigma=2,cache=true
--op python:code=@slice.py
--op '{"op": "python", "code": "@slice.py", "params": {"low": 30}}'
```

Values are parsed as JSON where possible, else strings. Repeat `--op` to chain. A `python`
op's `code=@file` is read from that file, on the command line only (a spec sent to a server
reads no file). The demos that start with Python show their command this way, and save the
file it reads (*Save slice.py*): the code as you have it, its sliders' defaults set to where
you left them. That is how a result found in a demo reaches Fiji or napari: run the command,
then open the server's `/processed/n5` in Fiji or `/processed/zarr` in napari (what a demo
computes exists only in its tab).
