"""URL schemes other packages add: registered in process or through the
``chunkmirage.sources`` entry point, opened by ``open_source`` like the built-in ones and
served like any source."""

from importlib.metadata import EntryPoint

import numpy as np
import pytest

from chunkmirage import Pipeline, open_source
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources import MultiscaleSource, Source, register_source, registry, schemes


class Ramp(Source):
    """x + 10 y + 100 z, of a size the URL gives."""

    def __init__(self, n: int):
        self._info = ArrayInfo((n, n, n), np.float32, (4, 4, 4), (1, 1, 1), ("nm",) * 3, ("z", "y", "x"))

    @property
    def info(self):
        return self._info

    def read(self, box: Box):
        z, y, x = np.mgrid[box.slices()]
        return (x + 10 * y + 100 * z).astype(np.float32)


def open_ramp(url: str) -> MultiscaleSource:
    return MultiscaleSource([Ramp(int(url.split("://")[1]))], name="ramp")


seen: dict = {}


def open_kept(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    seen.update(cache_bytes=cache_bytes, cache=cache)
    return open_ramp(url.replace("kept", "ramp"))


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(registry, "_SCHEMES", dict(registry._BUILTIN))
    monkeypatch.setattr(registry, "_ENTRYPOINTS_LOADED", False)


def test_a_registered_scheme_opens_and_serves(clean):
    register_source("ramp", open_ramp)  # an opener that wants only the URL
    assert "ramp" in schemes() and "synthetic" in schemes()
    p = Pipeline(open_source("ramp://8"), [{"op": "threshold", "low": 400}])
    np.testing.assert_array_equal(p.read(0, Box((0, 0, 0), (8, 8, 8))), (open_ramp("ramp://8")[0].read(Box((0, 0, 0), (8, 8, 8))) >= 400))


def test_an_entry_point_scheme_is_found_and_given_what_it_takes(clean, monkeypatch):
    ep = EntryPoint("kept", f"{__name__}:open_kept", "chunkmirage.sources")
    monkeypatch.setattr(registry, "entry_points", lambda group: [ep] if group == "chunkmirage.sources" else [])
    ms = open_source("kept://4", cache_bytes=123)
    assert ms.name == "ramp" and seen == {"cache_bytes": 123, "cache": None}


def test_built_in_schemes_cannot_be_replaced(clean, monkeypatch):
    with pytest.raises(ValueError, match="built in"):
        register_source("synthetic", open_ramp)
    ep = EntryPoint("stack", f"{__name__}:open_ramp", "chunkmirage.sources")
    monkeypatch.setattr(registry, "entry_points", lambda group: [ep])
    assert open_source("synthetic://blobs?shape=8,8,8&chunk=8,8,8&levels=1").levels[0].info.shape == (8, 8, 8)
    assert registry._opener("stack").__module__ == "chunkmirage.sources.stack"
