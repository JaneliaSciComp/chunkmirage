# Python API

## Top level

::: chunkmirage.pipeline.Pipeline
::: chunkmirage.pipeline.PipelineSpec
::: chunkmirage.sources.register.RegisterParams
::: chunkmirage.schema.spec_schema
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
::: chunkmirage.demand.Claim
::: chunkmirage.demand.Cancelled
::: chunkmirage.demand.claimed
::: chunkmirage.demand.Slots
::: chunkmirage.demand.Queue

## Coordinate transformations

The model every registration format is read into, the OME-Zarr 0.6 reader, the
deformable solver, and the `scene://`, `warp://` and `register://` sources built on them.

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
::: chunkmirage.registration.solve
::: chunkmirage.registration.Settings
::: chunkmirage.registration.find_affine
::: chunkmirage.sources.register.open_register

## Stitching

Tiles stitched by interest points and RANSAC, and fused, behind `stitch://` and the
browser's stitch page.

::: chunkmirage.stitching.StitchParams
::: chunkmirage.stitching.detect
::: chunkmirage.stitching.match
::: chunkmirage.stitching.ransac
::: chunkmirage.stitching.optimize
::: chunkmirage.stitching.register
::: chunkmirage.stitching.fuse
::: chunkmirage.stitching.tiles_from_bdv
::: chunkmirage.sources.stitch.open_stitch

## Tracking

An object followed through a time series of labels by its overlap, frame by frame, and
through its divisions into its likely daughters (new objects appearing near where it was: a
guess, not something the labels record), behind the browser's track page and
`examples/track_nucleus.py`.

::: chunkmirage.tracking.lineage
::: chunkmirage.tracking.follow
::: chunkmirage.tracking.step
::: chunkmirage.tracking.newborns
::: chunkmirage.tracking.daughters
::: chunkmirage.tracking.mother_of
::: chunkmirage.tracking.measure

## Frontends

::: chunkmirage.frontends.base.Frontend
