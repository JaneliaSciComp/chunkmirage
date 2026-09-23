# Ops reference

Run `chunkmirage ops` or `GET /api/ops` for the live list with JSON schemas. This page must
list every registered op; `tests/test_docs.py` enforces it.

| op          | parameters (defaults)                 | halo                  | cache | output  | description |
| ----------- | ------------------------------------- | --------------------- | ----- | ------- | ----------- |
| `threshold` | `low=0.0`, `high=None`, `value=1`     | 0                     | no    | uint8   | `low <= x < high` → `value`, else 0 |
| `cast`      | `dtype="uint8"`, `clip=True`          | 0                     | no    | `dtype` | cast, clipping to integer range first |
| `scale`     | `factor=1.0`, `offset=0.0`            | 0                     | no    | float32 | `x * factor + offset` |
| `gaussian`  | `sigma=1.0`, `truncate=3.0`           | `ceil(sigma*truncate)`| no    | float32 | isotropic Gaussian blur (scipy) |
| `uniform`   | `size=3`                              | `size // 2 + 1`       | no    | float32 | mean filter over a `size`³ cube (scipy) |

`gaussian` and `uniform` need the `ops` extra (scipy).

## CLI syntax

```
--op threshold:low=120,high=200
--op '{"op": "gaussian", "sigma": 2}'
```

Values are parsed as JSON where possible, else strings. Repeat `--op` to chain.
