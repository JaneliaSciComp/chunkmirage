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
| `morphology`| `operation` | open    | `open`, `close`, `erode`, `dilate` on a mask (input > 0); output uint8 |
|             | `radius`    | 2       | spherical structuring element radius in voxels; halo = `2 × radius + 1` |
| `label`     | `min_size`  | 0       | connected components of a mask, output uint32 segment ids; drop components smaller than this |
|             | `connectivity` | 1    | 1 = 6-connected, 2 = 18, 3 = 26 |

`label` numbers components **per chunk** (salted by chunk position so ids never collide).
An object spanning chunks therefore gets one colour per chunk. That is the honest per-chunk
preview of a global operation; see [FAQ](../faq.md#where-does-it-fall-short).

Ops that need to know *where* a block sits override `apply_at(block, box)` instead of
`apply(block)`; `label` uses it for the salt.

None of the built-ins cache their output (`cache=False`); everything except `threshold`,
`cast` and `scale` needs the `ops` extra (scipy).

## CLI syntax

```
--op threshold:low=120,high=200
--op '{"op": "gaussian", "sigma": 2}'
```

Values are parsed as JSON where possible, else strings. Repeat `--op` to chain.
