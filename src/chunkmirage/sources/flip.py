"""``flip://`` sources: an image mirrored along some of its axes, for data stored the other
way round from the images it belongs with (OpenOrganelle's N5 organelle predictions of
jrc_hela-2 are upside down in y relative to its EM, though no metadata says so).

    flip://<image>?axes=y[,x,...]

Each level is mirrored in its own index range: voxel ``i`` of a flipped axis of length ``n``
is voxel ``n - 1 - i`` of the image. Shape, voxel size, translation and chunks stay the
image's. Every level then mirrors about the same physical plane when its voxel centres sit
as a pyramid's do (each level's extent half the one below, centred on it), as OME-Zarr and
COSEM pyramids whose extents divide evenly are; ``open_flip`` refuses a pyramid that does
not, since its levels would disagree on where things are.
"""

from __future__ import annotations

from urllib.parse import parse_qs

import numpy as np

from chunkmirage.core import ArrayInfo, Box
from chunkmirage.sources.base import MultiscaleSource, Source


class FlipSource(Source):
    """One level of ``inner`` mirrored along ``axes`` (indices into its axes)."""

    def __init__(self, inner: Source, axes: tuple[int, ...], key: str):
        self.inner = inner
        self.axes = axes
        self._key = key

    @property
    def info(self) -> ArrayInfo:
        return self.inner.info

    def cache_key(self) -> str:
        return self._key

    def read(self, box: Box) -> np.ndarray:
        shape = self.inner.info.shape
        start, stop = list(box.start), list(box.stop)
        for a in self.axes:
            start[a], stop[a] = shape[a] - box.stop[a], shape[a] - box.start[a]
        data = self.inner.read(Box(tuple(start), tuple(stop)))
        return np.ascontiguousarray(np.flip(data, axis=self.axes))


def _centre(info: ArrayInfo, a: int) -> float:
    return info.translation[a] + info.voxel_size[a] * (info.shape[a] - 1) / 2


def open_flip(url: str, *, cache_bytes: int = 0, cache=None) -> MultiscaleSource:
    from chunkmirage.sources.registry import open_source

    location, _, query = url[len("flip://") :].rpartition("?")
    q = {k: v[-1] for k, v in parse_qs(query).items()}
    if not location or "axes" not in q:
        raise ValueError("flip:// needs an image and the axes to mirror: flip://<image>?axes=y")
    if extra := set(q) - {"axes"}:
        raise ValueError(f"flip:// takes only axes=..., not {sorted(extra)}")
    image = open_source(location, cache_bytes=cache_bytes, cache=cache)
    names = image.levels[0].info.axes
    wanted = [a.strip() for a in q["axes"].split(",") if a.strip()]
    if not wanted or any(a not in names for a in wanted):
        raise ValueError(f"flip:// axes={q['axes']}: {location} has axes {names}")
    axes = tuple(sorted({names.index(a) for a in wanted}))
    base = image.levels[0].info
    for i, lvl in enumerate(image.levels[1:], start=1):
        for a in axes:
            c0, c = _centre(base, a), _centre(lvl.info, a)
            if abs(c - c0) > 1e-3 * base.voxel_size[a]:
                raise ValueError(
                    f"flip:// {location}: level {i} is centred at {c:g} on {names[a]}, level 0 at "
                    f"{c0:g}, so mirroring each in place would move them apart"
                )
    tag = ",".join(names[a] for a in axes)
    levels = [FlipSource(lvl, axes, f"flip:{tag}:{lvl.cache_key()}") for lvl in image.levels]
    return MultiscaleSource(levels, name=image.name, shader=image.shader)
