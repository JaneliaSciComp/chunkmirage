"""python-neuroglancer integration: a viewer whose layers track a ``DatasetRegistry``.

Why this exists
---------------
Neuroglancer caches chunks by URL and has no "refetch" signal. chunkmirage embeds the
pipeline digest in every source URL, so a change of pipeline is a change of URL. With the
``neuroglancer`` Python package we control the viewer state from Python: when a dataset is
edited we swap that layer's source URL inside a state transaction. Neuroglancer refetches
that layer and nothing else, and the camera, other layers and shader settings are preserved.

The client code can be the bundled build or the one hosted at
``https://neuroglancer-demo.appspot.com`` (``client="appspot"``); the Python server proxies
it so the page still runs on localhost and state sync works.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from chunkmirage.neuroglancer import (
    DEFAULT_VIEWER,
    dimensions,
    global_dimensions,
    is_segmentation,
    source_url,
)
from chunkmirage.pipeline import Pipeline, PipelineSpec
from chunkmirage.server import DatasetRegistry

_SCHEMES = {"n5": "n5", "zarr": "zarr2", "zarr3": "zarr3", "precomputed": "precomputed"}


class Viewer:
    """A live Neuroglancer viewer bound to a registry.

    Args:
        registry: the registry the ASGI app serves (edits through REST or Python propagate).
        public_url: base URL clients use for chunk requests, e.g. ``http://localhost:8000``.
        format: which frontend the layers read from.
        client: ``"bundled"`` (default), ``"appspot"``, or a URL of a Neuroglancer build.
        bind_address / port: where the python-neuroglancer server listens. Use
            ``"0.0.0.0"`` to let other machines on the network open the viewer.
        public_host: host to put in the viewer URL (e.g. the machine's LAN IP).
    """

    def __init__(
        self,
        registry: DatasetRegistry,
        public_url: str,
        *,
        format: str = "zarr3",
        client: str = "bundled",
        bind_address: str = "127.0.0.1",
        port: int = 0,
        public_host: str | None = None,
    ):
        import neuroglancer

        if client == "appspot":
            neuroglancer.set_static_content_source(url=DEFAULT_VIEWER)
        elif client not in (None, "bundled"):
            neuroglancer.set_static_content_source(url=client)
        neuroglancer.set_server_bind_address(bind_address, port)
        self._ng = neuroglancer
        self.registry = registry
        self.public_url = public_url.rstrip("/")
        self.format = format
        self._renames: dict[str, dict[str, str]] = {}
        self.viewer = neuroglancer.Viewer()
        self.public_host = public_host
        self._lock = threading.Lock()
        self._unsubscribe = registry.subscribe(self._on_change)
        registry.viewer_url = self.url
        self.sync()

    # --- public -----------------------------------------------------------------------
    @property
    def url(self) -> str:
        """Viewer URL; the host is replaced by ``public_host`` when one was given, so the
        link works from other machines even though python-neuroglancer reports the bind
        address or FQDN."""
        url = str(self.viewer)
        if self.public_host:
            from urllib.parse import urlsplit, urlunsplit

            parts = urlsplit(url)
            netloc = f"{self.public_host}:{parts.port}" if parts.port else self.public_host
            url = urlunsplit(parts._replace(netloc=netloc))
        return url

    def hosted_link(self, viewer: str = DEFAULT_VIEWER) -> str:
        """The current state as a link to a hosted Neuroglancer (appspot by default). It
        does not follow later edits, but opens in any browser that can reach the chunk
        URLs, with no python server needed."""
        return self._ng.to_url(self.viewer.state, prefix=viewer.rstrip("/") + "/")

    def rename_dimensions(self, name: str, rename: Mapping[str, str]) -> None:
        """Show dataset ``name`` with some of its Neuroglancer dimensions renamed, e.g.
        ``{"c'": "c^"}`` to make the channel axis a shader channel (``getDataValue(i)``)
        or ``{"t": "t'"}`` to make time local to the layer. Kept across edits, and
        rebuilt from the dataset's current axes each time. zarr formats only."""
        if self.format not in ("zarr", "zarr3"):
            raise ValueError(f"renaming dimensions needs a zarr format, not {self.format!r}")
        self._renames[name] = dict(rename)
        self.sync({name: self.registry.get(name)})

    def set_ops(self, name: str, ops: list[dict]) -> Pipeline:
        """Convenience: replace the ops of dataset ``name`` keeping its source and settings."""
        p = self.registry.get(name)
        if p is None or p.spec is None:
            raise KeyError(f"{name!r} is not a spec-defined dataset")
        spec = p.spec.model_copy(update={"ops": ops})
        return self.registry.add(name, spec)

    def set_spec(self, name: str, spec: PipelineSpec | dict) -> Pipeline:
        return self.registry.add(name, spec)

    def set_dimensions(self, name: str) -> None:
        """Take the viewer's dimensions, position and display axes from dataset ``name``
        (by default they come from the first dataset by name), e.g. the one with a ``t``
        axis of frames."""
        with self._lock, self.viewer.txn() as s:
            self._set_dimensions(s, self.registry.get(name))

    def sync(self, names: Mapping[str, Pipeline | None] | None = None) -> None:
        """Make layers match the registry (all datasets, or just ``names``)."""
        items = dict(self.registry.items()) if names is None else dict(names)
        with self._lock, self.viewer.txn() as s:
            if names is None and len(s.dimensions.names) == 0 and items:
                first = next(p for p in items.values() if p is not None)
                self._set_dimensions(s, first)
            for name, p in items.items():
                if p is None:
                    if name in s.layers:
                        del s.layers[name]
                    continue
                url = self._source(name, p)
                if name in s.layers:
                    s.layers[name].source = url
                elif is_segmentation(p):
                    layer = self._ng.SegmentationLayer(source=url)
                    if p.ops and p.ops[-1].name == "threshold":
                        layer.segments = [int(getattr(p.ops[-1], "value", 1))]
                        layer.selected_alpha = 0.4
                        layer.not_selected_alpha = 0
                    s.layers[name] = layer
                else:
                    s.layers[name] = self._ng.ImageLayer(source=url)

    def close(self) -> None:
        self._unsubscribe()
        if self.registry.viewer_url == self.url:
            self.registry.viewer_url = None

    # --- internals ----------------------------------------------------------------------
    def _on_change(self, name: str, pipeline: Pipeline | None) -> None:
        self.sync({name: pipeline})

    def _source(self, name: str, p: Pipeline):
        url = source_url(self.public_url, name, self.format, _SCHEMES[self.format], p.digest())
        rename = self._renames.get(name)
        if not rename:
            return url
        dims = {rename.get(n, n): v for n, v in dimensions(p.info(0)).items()}
        space = self._ng.CoordinateSpace(
            names=list(dims),
            units=[u for _, u in dims.values()],
            scales=[v for v, _ in dims.values()],
        )
        transform = self._ng.CoordinateSpaceTransform(output_dimensions=space)
        return self._ng.LayerDataSource(url=url, transform=transform)

    def _set_dimensions(self, s, p: Pipeline) -> None:
        dims, position, display = global_dimensions(p.info(0))
        s.dimensions = self._ng.CoordinateSpace(
            names=list(dims),
            units=[u for _, u in dims.values()],
            scales=[v for v, _ in dims.values()],
        )
        s.position = position
        s.display_dimensions = display
