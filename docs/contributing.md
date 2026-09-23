# Contributing and documentation policy

## Development

```bash
uv sync --all-extras --group dev --group docs
uv run pytest -q
uv run ruff check src tests examples
uv run mkdocs serve          # live docs at http://127.0.0.1:8000
```

CI runs ruff, pytest and `mkdocs build --strict` on every push and pull request. Docs
deploy to GitHub Pages from `main`.

## Documentation must move with the code

The docs are part of the definition of done. Any change that adds, removes or renames a
user-visible thing updates the relevant page in the same commit:

| change                          | update                                                  |
| ------------------------------- | ------------------------------------------------------- |
| new / removed / renamed op      | `docs/reference/ops.md`                                 |
| new / removed frontend or URL   | `docs/concepts/formats.md`                              |
| REST endpoint                   | `docs/reference/api.md`                                 |
| CLI command or flag             | `docs/reference/cli.md`                                 |
| caching behaviour               | `docs/concepts/caching.md`                              |
| architectural decision          | `docs/design.md`                                        |
| roadmap item shipped            | remove from `docs/roadmap.md`, document where it landed |

Two mechanisms enforce this:

* `tests/test_docs.py` fails if a registered op is missing from the ops reference, a
  frontend is missing from the formats page, or a REST route is missing from the API
  reference.
* `mkdocs build --strict` fails on broken links and missing pages.

`CLAUDE.md` at the repo root restates this policy for AI-assisted sessions.

## Adding an op

```python
from chunkmirage.ops import Op, register
import numpy as np

@register
class MyOp(Op):
    """One-line description shown by `chunkmirage ops`."""
    name = "my_op"
    halo = 4            # voxels of upstream context needed per side (int or per-axis tuple)
    cache = False       # True for expensive stages (inference)

    strength: float = 1.0          # pydantic fields become parameters and JSON schema

    def output_dtype(self, in_dtype): return np.dtype("float32")
    def apply(self, block): return block.astype(np.float32) * self.strength
```

Register it in `pyproject.toml` under `[project.entry-points."chunkmirage.ops"]` (or in your
own package's entry points), add a test, and add a row to `docs/reference/ops.md`.
