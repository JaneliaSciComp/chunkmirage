# Contributing and documentation policy

## Development

```bash
uv sync --extra all --extra gpu --group dev --group docs   # or --extra cpu without an NVIDIA GPU
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
| control page / viewer / events  | `docs/concepts/interactivity.md`                        |
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
from pydantic import Field
import numpy as np

@register
class MyOp(Op):
    """One-line description shown by `chunkmirage ops`."""
    name = "my_op"
    halo = 4            # voxels of upstream context needed per side (int or per-axis tuple)
    cache = False       # True for expensive stages (inference)

    strength: float = Field(1.0, description="What it does, in words a user understands")  # shown in the UI

    def output_dtype(self, in_dtype): return np.dtype("float32")
    def apply(self, block): return block.astype(np.float32) * self.strength
```

An op shipped with chunkmirage goes in `src/chunkmirage_contrib/ops/`: the core keeps only
`downsample` and `python` ([what stays in the core](design.md#what-stays-in-the-core)).
Register it in `pyproject.toml` under `[project.entry-points."chunkmirage.ops"]` (or in your
own package's entry points), add a test, and add a row to `docs/reference/ops.md` (a
shipped op; a plugin's op is documented by its own package). An op that is a few lines of
numpy with sliders needs no class: `python_op(name, code)` makes one from code written for
the [`python` op](reference/ops.md#your-own-python-the-python-op), as the demos' *Copy as
plugin* does. Users' choice between the three is set out under
[your own op](reference/ops.md#your-own-op).

## Adding a source

```python
import numpy as np
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources import MultiscaleSource, Source

class Ramp(Source):
    def __init__(self, n):
        self._info = ArrayInfo((n, n, n), np.float32, (64, 64, 64), (8, 8, 8), ("nm",) * 3, ("z", "y", "x"))
    @property
    def info(self): return self._info
    def read(self, box: Box):  # exactly box.shape, within info.shape
        z, y, x = np.mgrid[box.slices()]
        return (x + y + z).astype(np.float32)

def open_ramp(url: str) -> MultiscaleSource:  # ramp://512
    return MultiscaleSource([Ramp(int(url.split("://")[1]))], name="ramp")
```

List it under `[project.entry-points."chunkmirage.sources"]` as `ramp = "mypackage:open_ramp"`
(see [formats](concepts/formats.md#your-own-schemes-sources-from-other-packages)). The core's
own schemes, the ones that compose other sources (`scene://`, `warp://`, `stack://`,
`flip://`), are listed in `chunkmirage.sources.registry`; a computed source shipped with
chunkmirage (`synthetic://`, `register://`) goes in `src/chunkmirage_contrib/sources/` and
under the entry point like any other package's. A format told apart by its path rather than
a scheme (a GeoTIFF by its extension) lists a matcher, `path -> opener or None`, under
`[project.entry-points."chunkmirage.source_matchers"]` (`register_matcher` in a running
process). Either is documented in `docs/concepts/formats.md`.

## Plugin API and versioning

Packages build on chunkmirage through its plugin API, which is stable:

* **Ops.** `Op` and its contract: the class attributes `name`, `halo`, `overlap`, `cache`,
  `packages`, `output_kind` and `slots`; the methods `apply`, `apply_at`, `output_dtype`,
  `output_info`, `input_voxel_size`, `for_level` and `cache_token`; parameters as pydantic
  fields. `register` and the `chunkmirage.ops` entry point. `demand.cancelled()` and
  `demand.Cancelled`, for a long op to give up once no request waits for it.
* **Sources.** `Source` (`info`, `read`, `read_padded`, `cache_key`), `ChunkedSource`,
  `MultiscaleSource`, `ArrayInfo` (its `computed_from` among it) and `Box`; an opener `opener(url, *, cache_bytes, cache)`
  returning a `MultiscaleSource`, registered with `register_source` or the
  `chunkmirage.sources` entry point, or for files told apart by their path a matcher
  registered with `register_matcher` or `chunkmirage.source_matchers`; `open_source`.
* **Schema.** A parameter model listed under `chunkmirage.models` appears in
  `chunkmirage schema` under the entry point's name, for the browser engine's types.
* **Serving.** `PipelineSpec`, `Pipeline`, `create_app` and its parameters, `serve` and
  `Server`, `materialize`, `gpu.set_device`, `demand.set_gather_threads`,
  `DatasetRegistry` (`add`, `get`, `resolve`, `refresh`, `remove`, `subscribe`) and its
  resolver, the `chunkmirage.routes` entry point, the REST routes in the
  [API reference](reference/api.md), the served URL layout, and the CLI's flags.

Everything else may change in any release: names starting with `_`, modules not named
above (`chunkmirage.fused`, the frontends' internals), log messages, the control page and
the browser engine.

Versions follow [semantic versioning](https://semver.org). Before 1.0, a minor release
(0.x.0) may change the stable API, but only with a changelog entry under its own heading
saying what to change in a plugin, and, where it can, after one minor release in which the
old way still works and warns. A patch release (0.x.y) never breaks it. From 1.0, a change
to the stable API waits for a major release.

## Releasing

1. Move the changelog's unreleased entries under the new version, and set that version in
   `pyproject.toml` and `chunkmirage/__init__.py` (`tests/test_docs.py` checks the three
   agree).
2. Merge to `main` with CI green.
3. Publish to PyPI only when the owners say so: `uv build && uv publish`.
