"""A page's call into Python by module path: the browser engine's
``Engine.call("pkg.mod:fn", args, arrays)``, with which a page of another project runs its
own Python in the Pyodide workers (the modules it gave ``Engine.start``), as the demos run
``chunkmirage_contrib``'s stitching and tracking steps.

The function named is called as ``fn(args, arrays)``: ``args`` the JSON arguments as a
dict, ``arrays`` the raw bytes sent beside them, in order (buffers; ``np.frombuffer`` reads
one). It returns a JSON value, one array (``bytes``, or anything with ``tobytes``: a numpy
array, a memoryview), or a pair ``(value, arrays)``, each array the same. Nothing here is
Pyodide's: the same dispatch runs natively, so a consumer tests its
functions with pytest and the worker only converts JavaScript's buffers.
"""

import importlib
import json
from collections.abc import Callable, Sequence
from typing import Any


def resolve(fn: str) -> Callable:
    """The function ``"pkg.mod:name"`` names, imported."""
    module, _, name = fn.partition(":")
    if not module or not name:
        raise ValueError(f"a function is named by its module and name, pkg.mod:fn, not {fn!r}")
    return getattr(importlib.import_module(module), name)


def _bytes(array: Any) -> bytes:
    return array.tobytes() if hasattr(array, "tobytes") else bytes(array)


def call(fn: str, args: str, arrays: Sequence[Any]):
    """Call ``fn`` with the JSON text ``args`` and the buffers ``arrays``, and marshal its
    answer as the worker sends it: JSON text, ``bytes`` (one array: bytes, or anything with
    ``tobytes``, a numpy array among them), or ``(JSON text, [bytes, ...])``."""
    out = resolve(fn)(json.loads(args), list(arrays))
    if isinstance(out, tuple):
        value, buffers = out
        return json.dumps(value), [_bytes(b) for b in buffers]
    if isinstance(out, (bytes, bytearray, memoryview)) or hasattr(out, "tobytes"):
        return _bytes(out)
    return json.dumps(out)
