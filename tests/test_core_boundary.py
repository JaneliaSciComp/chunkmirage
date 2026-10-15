"""The core composes, it does not compute: nothing under ``chunkmirage`` imports
``chunkmirage_contrib``, by text or at import time, so the engine stands without it and
contrib reaches it only through the entry points any other package would use."""

import importlib
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import chunkmirage

SRC = Path(chunkmirage.__file__).parent


def test_no_core_module_imports_contrib():
    # by text, so a lazy import inside a function counts too (docstrings may name it)
    imports = re.compile(r"^\s*(from|import)\s+chunkmirage_contrib\b", re.M)
    offenders = [str(p.relative_to(SRC.parent)) for p in SRC.rglob("*.py") if imports.search(p.read_text())]
    assert not offenders, f"core modules that import contrib: {offenders}"


def test_importing_every_core_module_loads_no_contrib():
    # in a fresh interpreter, so what this process has already imported does not count
    modules = [m.name for m in pkgutil.walk_packages([str(SRC)], "chunkmirage.")]
    assert "chunkmirage.pipeline" in modules
    code = (
        "import importlib, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        "print(sorted(m for m in sys.modules if m.startswith('chunkmirage_contrib')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"importing the core loaded contrib: {out.stdout}"


def test_contrib_reaches_the_core_through_entry_points():
    from chunkmirage.ops.base import list_ops
    from chunkmirage.schema import spec_schema
    from chunkmirage.sources import open_source, schemes

    assert {"threshold", "gaussian", "spots", "slope"} <= set(list_ops())
    assert {"synthetic", "register", "stitch"} <= set(schemes())
    assert {"register", "stitch"} <= set(spec_schema()["properties"])
    assert open_source("synthetic://blobs?shape=8,8,8").levels[0].info.shape == (8, 8, 8)
    assert importlib.import_module("chunkmirage_contrib.sources.geotiff").match("a.tif") is not None
