"""``dataset://`` sources: another dataset of the same server, read in-process.

    dataset://<name>

The levels are the named dataset's own output stages, so a chunk is read through that
dataset's stage cache, never over HTTP and never computed twice, and every stage key
downstream holds its keys: an edit to the named dataset is an edit to everything built on
it, which the registry then rebuilds (``DatasetRegistry.add``). A chain of such datasets
(a field, the image warped by it, the next pass fitted on that, its field, ...) costs one
read per stage. A dataset may not read itself by any chain: a cycle is refused when the
dataset is added.

A ``dataset://`` URL resolves against the registry that is building the dataset holding it
(``building``), so it is opened through a ``DatasetRegistry`` or the REST API, not by
``open_source`` on its own.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from chunkmirage.sources.base import MultiscaleSource

if TYPE_CHECKING:
    from chunkmirage.server import DatasetRegistry

# while a registry builds a dataset: the registry, the names being built (the outermost
# first: a dataset a resolver builds while another is being built sees the whole chain),
# and the set the names it reads are collected into
_BUILDING: contextvars.ContextVar[tuple[DatasetRegistry, tuple[str, ...], set[str]] | None] = (
    contextvars.ContextVar("chunkmirage.dataset.building", default=None)
)


@contextmanager
def building(registry: DatasetRegistry, name: str) -> Iterator[set[str]]:
    """Within it, ``dataset://`` URLs resolve against ``registry`` and name the datasets
    ``name`` reads into the set yielded (its upstreams, for the registry to rebuild it when
    one of them changes)."""
    outer = _BUILDING.get()
    chain = (outer[1] if outer else ()) + (name,)
    upstreams: set[str] = set()
    token = _BUILDING.set((registry, chain, upstreams))
    try:
        yield upstreams
    finally:
        _BUILDING.reset(token)


def open_dataset(url: str) -> MultiscaleSource:
    name = url[len("dataset://") :].strip("/")
    if not name:
        raise ValueError("dataset:// needs a name: dataset://<name>")
    ctx = _BUILDING.get()
    if ctx is None:
        raise ValueError(
            f"{url}: dataset:// reads another dataset of the server building this one, so it "
            "is opened through a DatasetRegistry (or the REST API), not on its own"
        )
    registry, chain, upstreams = ctx
    # a cycle: the dataset being built, or one being built around it, is this one or
    # reads it (through datasets already registered)
    for n in chain:
        if n == name or n in registry.upstreams(name):
            raise ValueError(f"{url}: a cycle: {' -> '.join((*chain, name))}")
    pipeline = registry.resolve(name)
    if pipeline is None:
        raise ValueError(f"{url}: no dataset named {name!r}")
    upstreams.add(name)
    return MultiscaleSource(list(pipeline.levels), name=name, cached=True)
