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

One step has a kernel of its own: ``interpolate``, the linear interpolation of a field's or
an image's window at the points. Through ``cupyx.scipy.ndimage`` it took a launch per
component plus the clamping and offsetting around them, each of which holds the GIL and so
paces the server's other threads, and that library compiles a kernel for each number of
points it is given. The kernel is ``map_coordinates``'s own arithmetic in its own order, so
its values are the same bits; ``KERNELS = False`` goes back to the library, the reference
``tests/test_gpu.py`` checks it against on a GPU.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

import numpy as np

KERNELS = True  # False: interpolate through cupyx.scipy.ndimage instead (the reference)
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


def interpolate(coords, window, lo, hi=None, *, components_first=False, dtype=None):
    """Every component of ``window`` (``(c, *spatial)``, on the device) interpolated linearly
    at ``coords`` (``(n, d)``, on the device), which are in the grid ``window`` was cut from
    at corner ``lo``: each coordinate clamped to ``[0, hi]`` first when ``hi`` is given, then
    made relative to ``lo``, then interpolated as ``cupyx.scipy.ndimage.map_coordinates``
    does with ``order=1, mode="nearest", prefilter=False``, bit for bit. Returns ``(n, c)``,
    or ``(c, n)`` with ``components_first``, in ``dtype`` (by default the window's)."""
    import cupy

    n, d = coords.shape
    nc = window.shape[0]
    out = cupy.empty((nc, n) if components_first else (n, nc), dtype=dtype or window.dtype)
    clamp = [int(v) for v in hi] if hi is not None else [0] * d
    _interpolate_kernel(d)(
        cupy.ascontiguousarray(coords),
        cupy.ascontiguousarray(window),
        *(int(v) for v in lo),
        *clamp,
        int(hi is not None),
        nc,
        1 if components_first else nc,
        n if components_first else 1,
        out,
        size=n,
    )
    return out


@functools.cache
def _interpolate_kernel(d: int):
    """``interpolate``'s kernel for ``d`` spatial axes: one thread a point, every component.
    The lines mirror cupyx's ``_generate_interp_custom`` for order 1 and mode ``nearest``
    (weights in the coordinates' type ``W``, the sum in the output's ``Y``, the corners in
    the same order), so the compiler fuses the same multiply-adds and the bits agree."""
    import cupy

    ops = [f"const int xs_{j} = (int)x.shape()[{j + 1}];" for j in range(d)]
    ops.append(f"const ptrdiff_t sx_{d - 1} = 1;")
    ops += [f"const ptrdiff_t sx_{j - 1} = sx_{j} * xs_{j};" for j in range(d - 1, 0, -1)]
    ops.append("const ptrdiff_t plane = sx_0 * xs_0;")
    for j in range(d):
        ops.append(f"""
        W c_{j} = g[i * {d} + {j}];
        if (use_hi) c_{j} = c_{j} < (W)0 ? (W)0 : (c_{j} > (W)hi_{j} ? (W)hi_{j} : c_{j});
        c_{j} = c_{j} - (W)lo_{j};
        const int cf_{j} = (int)floor((double)c_{j});
        const int cc_{j} = cf_{j} + 1;
        const int n_{j} = (c_{j} == cf_{j}) ? 1 : 2;
        const int cfb_{j} = min(max(cf_{j}, 0), xs_{j} - 1);
        const int ccb_{j} = min(max(cc_{j}, 0), xs_{j} - 1);""")
    ops.append("for (int q = 0; q < nc; q++) {\nY out = 0.0;")
    for j in range(d):
        ops.append(f"""
        for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++) {{
            W w_{j};
            ptrdiff_t ic_{j};
            if (s_{j} == 0) {{
                w_{j} = (W)cc_{j} - c_{j};
                ic_{j} = cfb_{j} * sx_{j};
            }} else {{
                w_{j} = c_{j} - (W)cf_{j};
                ic_{j} = ccb_{j} * sx_{j};
            }}""")
    corner = " + ".join(f"ic_{j}" for j in range(d))
    weight = " * ".join(f"w_{j}" for j in range(d))
    ops.append(f"Y val = (Y)x[q * plane + {corner}];\nout += val * (Y)({weight});")
    ops.append("}" * d)
    ops.append("o[i * sp + q * sq] = out;\n}")
    params = ["raw W g", "raw X x"]
    params += [f"int32 lo_{j}" for j in range(d)] + [f"int32 hi_{j}" for j in range(d)]
    params += ["int32 use_hi", "int32 nc", "int64 sp", "int64 sq"]
    return cupy.ElementwiseKernel(
        ", ".join(params), "raw Y o", "\n".join(ops), f"chunkmirage_interpolate_{d}d"
    )
