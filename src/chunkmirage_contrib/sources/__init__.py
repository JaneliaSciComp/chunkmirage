"""The computed sources shipped with chunkmirage: ``synthetic://``, ``register://``,
``stitch://`` and GeoTIFF files. Each is found through the ``chunkmirage.sources`` (or
``chunkmirage.source_matchers``) entry point and imported only when a URL asks for it, so
this module imports none of them: ``register://`` needs torch, GeoTIFFs tifffile."""
