"""Mesh frontend: a dataset's surface as Neuroglancer precomputed meshes, each fragment meshed
when it is fetched (``chunkmirage.meshes``): the legacy single-resolution format, or with
``lods`` above 1 the multi-resolution one, whose fragments are read by HTTP Range requests.
Neuroglancer shows it as a segmentation layer of segment 1:
``precomputed://http://host/<name>/mesh``."""

from __future__ import annotations

import re
import threading
import weakref

import numpy as np

from chunkmirage import meshes
from chunkmirage.core import ArrayInfo, Box
from chunkmirage.frontends.base import ChunkRequest, Frontend, Metadata, register_frontend
from chunkmirage.pipeline import Pipeline


def _spec(pipeline: Pipeline) -> meshes.MeshSpec:
    return (pipeline.spec.mesh if pipeline.spec else None) or meshes.MeshSpec()


@register_frontend
class MeshFrontend(Frontend):
    name = "mesh"
    neuroglancer_scheme = "precomputed"

    def _level(self, pipeline: Pipeline) -> int:
        return meshes.mesh_level(
            [pipeline.info(i) for i in range(pipeline.num_levels)], _spec(pipeline)
        )

    def __init__(self):
        self._octrees: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _octree(self, pipeline: Pipeline):
        """The multi-resolution levels, nodes and index of a pipeline's mesh, made once from
        its whole coarsest level."""
        spec = _spec(pipeline)
        with self._lock:
            got = self._octrees.get(pipeline)
            if got is None or got[0] != spec:
                infos = [pipeline.info(i) for i in range(pipeline.num_levels)]
                levels = meshes.lod_levels(infos, spec)
                top = infos[levels[-1]]
                chunk = top.chunk_shape[-3:]
                coarse = pipeline.read(levels[-1], Box((0,) * top.ndim, top.shape))
                nodes = meshes.multires_nodes(spec, coarse, infos, chunk)
                flat = [(lv, tuple(int(v) for v in n)) for lv, ns in zip(levels, nodes) for n in ns]
                got = (spec, levels, chunk, flat, meshes.multires_index(nodes, infos, levels, chunk))
                self._octrees[pipeline] = got
            return got

    def resolve_range(self, pipeline: Pipeline, path: str, start: int, stop: int) -> ChunkRequest | None:
        if path.strip("/") != "1" or _spec(pipeline).lods == 1:
            return None
        _, _, _, flat, _ = self._octree(pipeline)
        total = len(flat) * meshes.FRAGMENT_BYTES
        if not 0 <= start < stop <= total:
            raise ValueError(f"bytes {start}-{stop} of a {total}-byte mesh")
        k = start // meshes.FRAGMENT_BYTES
        return ChunkRequest(flat[k][0], flat[k][1], part=(start, stop), total=total)

    def resolve(self, pipeline: Pipeline, path: str) -> Metadata | ChunkRequest | None:
        path = path.strip("/")
        if _spec(pipeline).lods > 1:
            if path == "info":
                return Metadata.json(meshes.multires_info())
            if path == "1.index":
                return Metadata(self._octree(pipeline)[4], "application/octet-stream")
            return None
        if path == "info":
            return Metadata.json({"@type": "neuroglancer_legacy_mesh"})
        if path == "1:0":
            info = pipeline.info(self._level(pipeline))
            return Metadata.json({"fragments": meshes.fragment_names(info)})
        m = re.fullmatch(r"1:0:(\d+(?:_\d+)*)", path)
        if m:
            level = self._level(pipeline)
            index = tuple(int(i) for i in m.group(1).split("_"))
            if len(index) != pipeline.info(level).ndim or any(
                i >= n for i, n in zip(index, pipeline.info(level).chunk_grid)
            ):
                return None
            return ChunkRequest(level, index)
        return None

    def compute(self, pipeline: Pipeline, req: ChunkRequest) -> bytes:
        if req.part is not None:  # bytes of the multi-resolution fragments, each padded
            spec, _, chunk, flat, _ = self._octree(pipeline)
            size = meshes.FRAGMENT_BYTES
            first, last = req.part[0] // size, (req.part[1] - 1) // size
            body = b"".join(self._fragment(pipeline, spec, chunk, *flat[k]) for k in range(first, last + 1))
            return body[req.part[0] - first * size : req.part[1] - first * size]
        info = pipeline.info(req.level)
        box = meshes.fragment_box(info, req.index)
        return meshes.fragment(_spec(pipeline), pipeline.read(req.level, box), box, info)

    def _fragment(self, pipeline: Pipeline, spec, chunk, level: int, node) -> bytes:
        info = pipeline.info(level)
        box = meshes.node_box(info, node, chunk)
        verts, faces = meshes.multires_fragment(spec, pipeline.read(level, box), box, info, chunk)
        return meshes.encode_draco(verts, faces)

    def encode(self, info: ArrayInfo, index: tuple[int, ...], block: np.ndarray) -> bytes:
        raise NotImplementedError("meshes are computed from their fragment's box, not a chunk")
