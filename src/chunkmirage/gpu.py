"""The resampler on a GPU, through CuPy: ``warp://`` and ``scene://`` push every output voxel
through a chain of affines and sampled displacement fields and interpolate the image there,
which on the CPU is most of a registration's live view (sampling the fields above all).
With a device set (``set_device``, ``serve --device``) that code runs as it is with CuPy for
numpy and ``cupyx.scipy.ndimage`` for ``scipy.ndimage``, whose calls are the same: the
points, each field's window and the image's window on the device, in single precision, so
values agree with the CPU's to within a grey level and the cache keys are the same. Only
chains of affines and sampled fields move; a slab whose chain has anything else (a swirl,
an inverse estimated numerically, coordinates read from a field) is resampled on the CPU.

The functions here let one piece of code take either kind of array: ``array_module`` and
``ndimage`` give the libraries for an array, ``host`` brings one back as numpy.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_device: int | None = None  # the CUDA device the resampler runs on; None: numpy and scipy


def set_device(device: str) -> None:
    """Where the resampler runs: ``cpu`` (numpy and scipy, the default), ``cuda`` or
    ``cuda:N`` (CuPy on that GPU), or ``auto`` (the first GPU if CuPy sees one, else the
    CPU)."""
    global _device
    if device == "cpu":
        _device = None
        return
    try:
        import cupy
    except ImportError as e:
        if device == "auto":
            _device = None
            return
        raise ValueError(f"device {device!r} needs CuPy (the cupy extra)") from e
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        count = 0
    if device == "auto":
        _device = 0 if count else None
        return
    n = 0 if device == "cuda" else int(device[5:]) if device.startswith("cuda:") and device[5:].isdigit() else None
    if n is None:
        raise ValueError(f"device {device!r}: expected cpu, auto, cuda or cuda:N")
    if n >= count:
        raise ValueError(f"device {device!r}: CuPy sees {count} GPU(s)")
    _device = n


def device() -> int | None:
    """The CUDA device the resampler runs on, or None for the CPU."""
    return _device


def _on_device(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "cupy"


def array_module(a: Any) -> ModuleType:
    """numpy for a numpy array (or anything else), CuPy for a CuPy one."""
    if _on_device(a):
        import cupy

        return cupy
    return np


def ndimage(xp: ModuleType) -> ModuleType:
    """``scipy.ndimage``, or ``cupyx.scipy.ndimage`` for CuPy: the same calls."""
    if xp is np:
        import scipy.ndimage

        return scipy.ndimage
    import cupyx.scipy.ndimage

    return cupyx.scipy.ndimage


def host(a: Any) -> np.ndarray:
    """``a`` as a numpy array, copied from the device if it is there."""
    return a.get() if _on_device(a) else np.asarray(a)
