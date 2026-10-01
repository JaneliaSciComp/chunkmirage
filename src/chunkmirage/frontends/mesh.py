"""Mesh frontend: a dataset's surface as Neuroglancer precomputed (legacy) meshes, each
fragment meshed when it is fetched (``chunkmirage.meshes``). Neuroglancer shows it as a
segmentation layer of segment 1: ``precomputed://http://host/<name>/mesh``."""

from __future__ import annotations

import re

import numpy as np

from chunkmirage import meshes
from chunkmirage.core import ArrayInfo
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

    def resolve(self, pipeline: Pipeline, path: str) -> Metadata | ChunkRequest | None:
        path = path.strip("/")
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
        info = pipeline.info(req.level)
        box = meshes.fragment_box(info, req.index)
        return meshes.fragment(_spec(pipeline), pipeline.read(req.level, box), box, info)

    def encode(self, info: ArrayInfo, index: tuple[int, ...], block: np.ndarray) -> bytes:
        raise NotImplementedError("meshes are computed from their fragment's box, not a chunk")
