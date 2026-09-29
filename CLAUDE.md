# chunkmirage: notes for AI-assisted sessions

Library + CLI that spoofs zarr v2/v3, N5 and Neuroglancer precomputed over HTTP with a
pipeline of block ops, per-stage caching, and a REST API. General-purpose: not built for
any one downstream project. example-virtual-n5 and cellmap-flow are prior art / example
consumers, not the purpose. Architecture and rationale: `docs/design.md`.

## Workflow

- Env: `uv sync --all-extras --group dev --group docs`. In the Claude Code sandbox the
  default uv cache is read-only; use `UV_CACHE_DIR=$PWD/.uv-cache` (gitignored).
- Check: `uv run pytest -q && uv run ruff check src tests examples web && uv run mkdocs build --strict`.
- Browser engine (`web/`, TypeScript + Vite): `cd web && npm ci && npm run check`. After
  changing a Pydantic model it mirrors, regenerate its types:
  `uv run chunkmirage schema --out web/src/generated/chunkmirage.schema.json && (cd web && npm run gen)`.
- Do not push or create the GitHub repo unless asked.

## Documentation policy (non-negotiable)

Docs are part of every change. When you add, remove or rename anything user-visible,
update the matching page in the same commit:

- op → `docs/reference/ops.md`; frontend/URL → `docs/concepts/formats.md`;
  REST route → `docs/reference/api.md`; CLI flag → `docs/reference/cli.md`;
  caching behaviour → `docs/concepts/caching.md`; control page / viewer / events →
  `docs/concepts/interactivity.md`; design decision → `docs/design.md`;
  shipped roadmap item → remove from `docs/roadmap.md` and document where it landed.
- `tests/test_docs.py` enforces ops/frontends/routes coverage; `mkdocs build --strict`
  catches broken links. Keep both green.
- Prefer updating an existing page over adding a new one. Keep `README.md` short and
  pointing at the docs.

## Conventions

- Arrays are numpy C order `(z, y, x)`; N5 and precomputed reverse axis lists at the edge.
- Every pipeline stage is a `ChunkedSource`; caching and halos live there, not in ops.
- Ops are pydantic models with class-level `name`, `halo`, `cache`; register via
  `@register` and the `chunkmirage.ops` entry point.
