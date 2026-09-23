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

None of the built-ins cache their output (`cache=False`); `gaussian` and `uniform` need the
`ops` extra (scipy).

## CLI syntax

```
--op threshold:low=120,high=200
--op '{"op": "gaussian", "sigma": 2}'
```

Values are parsed as JSON where possible, else strings. Repeat `--op` to chain.
