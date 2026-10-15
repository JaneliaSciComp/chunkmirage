"""chunkmirage_contrib: what computes, shipped beside the engine.

The ``chunkmirage`` core composes data and serves it; it computes nothing but the
downsampling a pyramid needs. Everything with an algorithm in it lives here: the ops
(``chunkmirage_contrib.ops``), the computed sources (``synthetic://``, ``register://``,
``stitch://``, GeoTIFF reading), and the registration, stitching and tracking code behind
them. The core finds all of it through the same entry points any other package would use
(``chunkmirage.ops``, ``chunkmirage.sources``, ``chunkmirage.source_matchers``,
``chunkmirage.models``) and imports none of it; ``tests/test_core_boundary.py`` keeps it so.
"""
