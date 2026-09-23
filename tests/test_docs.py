"""Docs must move with the code. These tests fail when a user-visible thing is undocumented."""

import re
from pathlib import Path

from chunkmirage.frontends import FRONTENDS
from chunkmirage.ops.base import list_ops
from chunkmirage.server import create_app

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_every_op_is_in_ops_reference():
    text = (DOCS / "reference" / "ops.md").read_text()
    missing = [name for name in list_ops() if f"`{name}`" not in text]
    assert not missing, f"undocumented ops: {missing} (add rows to docs/reference/ops.md)"


def test_every_frontend_is_in_formats_page():
    text = (DOCS / "concepts" / "formats.md").read_text()
    missing = [name for name in FRONTENDS if f"`{name}`" not in text]
    assert not missing, f"undocumented frontends: {missing} (update docs/concepts/formats.md)"


def test_every_route_is_in_api_reference():
    text = (DOCS / "reference" / "api.md").read_text()
    app = create_app({})
    # Starlette reports converters like `{path:path}`; docs write plain `{path}`.
    routes = {re.sub(r"\{(\w+):\w+\}", r"{\1}", r.path) for r in app.routes if hasattr(r, "path")}
    # A route without a trailing segment is covered by its documented `{path}` variant.
    missing = [p for p in routes if f"`{p}`" not in text and f"`{p}/{{path}}`" not in text]
    assert not missing, f"undocumented routes: {missing} (update docs/reference/api.md)"


def test_roadmap_does_not_claim_shipped_features_are_future():
    """Cheap guard: anything named here exists, so it must not appear as a roadmap item."""
    text = (DOCS / "roadmap.md").read_text().lower()
    shipped = ["zarr3 frontend", "threshold op", "rest api for live edits"]
    for item in shipped:
        assert not re.search(rf"^\d+\.\s+\*\*.*{re.escape(item)}", text, re.M), f"{item} is shipped"
