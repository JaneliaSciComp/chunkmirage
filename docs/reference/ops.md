# Ops reference

Run `chunkmirage ops` or `GET /api/ops` for the live list with JSON schemas. This page must
list every registered op; `tests/test_docs.py` enforces it.

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
| `morphology`| `operation` | open    | `open`, `close`, `erode`, `dilate` on a mask (input > 0); output uint8 |
|             | `radius`    | 2       | spherical structuring element radius in voxels; halo = `2 × radius + 1` |
| `label`     | `min_size`  | 0       | connected components of a mask, output uint32 segment ids; drop components smaller than this |
|             | `connectivity` | 1    | 1 = 6-connected, 2 = 18, 3 = 26 |
| `spots`     | `sigma`     | 1.0     | bright diffraction-limited spots (single mRNAs in smFISH, EASI-FISH): difference of Gaussians, local maxima above `threshold`, each drawn as a small ball whose uint32 id comes from its position, so it is the same whichever chunk finds it; spot size in y-x voxels; needs a `z, y, x` volume (`select` a channel) |
|             | `sigma_z`   | 0.6     | spot size in z voxels |
|             | `threshold` | 10      | least difference-of-Gaussians response, image intensity units |
|             | `separation`| 2       | spots closer than this (y-x voxels) are one |
|             | `radius`    | 1       | ball drawn per spot, y-x voxels (0 marks one voxel) |
| `contacts`  | `radius`    | 3.0     | contact sites between the first two channels of a `stack://` source: voxels within this many voxels (Euclidean) of both structures; output uint8 mask; halo = `radius + 1` |
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
`cast`, `scale` and `diff` needs the `ops` extra (scipy).

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
