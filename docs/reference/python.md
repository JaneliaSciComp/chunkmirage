# Python API

## Top level

::: chunkmirage.pipeline.Pipeline
::: chunkmirage.pipeline.PipelineSpec
::: chunkmirage.sources.registry.open_source
::: chunkmirage.server.create_app
::: chunkmirage.server.DatasetRegistry

## Ops

::: chunkmirage.ops.base.Op
::: chunkmirage.ops.base.register

## Sources and core types

::: chunkmirage.sources.base.Source
::: chunkmirage.sources.base.ChunkedSource
::: chunkmirage.sources.base.MultiscaleSource
::: chunkmirage.core.ArrayInfo
::: chunkmirage.core.Box
::: chunkmirage.cache.LRUCache

## Coordinate transformations

The model every registration format is read into, the OME-Zarr 0.6 reader, and the
`scene://` source built on them.

::: chunkmirage.transforms.Transform
::: chunkmirage.transforms.Affine
::: chunkmirage.transforms.VectorField
::: chunkmirage.transforms.Displacements
::: chunkmirage.transforms.InverseDisplacements
::: chunkmirage.transforms.SwirlField
::: chunkmirage.transforms.Swirls
::: chunkmirage.transforms.simplify
::: chunkmirage.ngff.Scene
::: chunkmirage.ngff.parse_transform
::: chunkmirage.sources.scene.open_scene
::: chunkmirage.sources.warp.open_warp

## Frontends

::: chunkmirage.frontends.base.Frontend
