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

from chunkmirage.neuroglancer import _UNIT_TO_M, DEFAULT_VIEWER, is_segmentation, source_url
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
        bind_address / port: where the python-neuroglancer server listens.
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
        self.viewer = neuroglancer.Viewer()
        self._lock = threading.Lock()
        self._unsubscribe = registry.subscribe(self._on_change)
        self.sync()

    # --- public -----------------------------------------------------------------------
    @property
    def url(self) -> str:
        return str(self.viewer)

    def set_ops(self, name: str, ops: list[dict]) -> Pipeline:
        """Convenience: replace the ops of dataset ``name`` keeping its source and settings."""
        p = self.registry.get(name)
        if p is None or p.spec is None:
            raise KeyError(f"{name!r} is not a spec-defined dataset")
        spec = p.spec.model_copy(update={"ops": ops})
        return self.registry.add(name, spec)

    def set_spec(self, name: str, spec: PipelineSpec | dict) -> Pipeline:
        return self.registry.add(name, spec)

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
                url = source_url(
                    self.public_url, name, self.format, _SCHEMES[self.format], p.digest()
                )
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

    # --- internals ----------------------------------------------------------------------
    def _on_change(self, name: str, pipeline: Pipeline | None) -> None:
        self.sync({name: pipeline})

    def _set_dimensions(self, s, p: Pipeline) -> None:
        info = p.info(0)
        names = [a for a in info.axes if a != "c"]
        scales = [
            vs * _UNIT_TO_M.get(u, 1e-9)
            for a, vs, u in zip(info.axes, info.voxel_size, info.units)
            if a != "c"
        ]
        s.dimensions = self._ng.CoordinateSpace(
            names=names, units=["m"] * len(names), scales=scales
        )
        s.position = [sh / 2 for a, sh in zip(info.axes, info.shape) if a != "c"]
