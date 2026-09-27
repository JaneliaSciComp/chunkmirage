# RFC-5 transformation conformance cases

Copied unmodified from
[clbarnes/ome_zarr_transformations_conformance](https://github.com/clbarnes/ome_zarr_transformations_conformance)
at commit `6f93379` (MIT, see `LICENSE`). Each `cases/*/` directory is an OME-Zarr 0.6
scene plus a `conformance.toml` giving source points and the expected target points.
`tests/test_scene.py` runs every case through `chunkmirage.ngff.Scene`.
