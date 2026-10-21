"""chunkmirage.calls: a page's call into Python by module path (``Engine.call("pkg.mod:fn",
args, arrays)``), the function given the JSON arguments and the arrays, and its answer
marshalled as the Pyodide worker sends it (JSON, bytes, or JSON with arrays)."""

import json

import numpy as np
import pytest

from chunkmirage import calls

CONSUMER = """\
import numpy as np

def total(args, arrays):
    values = np.frombuffer(arrays[0], np.dtype(args["dtype"]))
    return {"n": int(values.size), "sum": float(values.sum()) * args.get("scale", 1)}

def doubled(args, arrays):
    return (np.frombuffer(arrays[0], np.float32) * 2).tobytes()

def split(args, arrays):
    values = np.frombuffer(arrays[0], np.float32)
    return {"halves": 2}, [values[: values.size // 2], values[values.size // 2 :].tobytes()]
"""


def test_a_consumers_function_is_called_by_module_path_with_its_arrays(tmp_path, monkeypatch):
    (tmp_path / "consumer_page.py").write_text(CONSUMER)
    monkeypatch.syspath_prepend(str(tmp_path))
    values = np.array([1, 2, 3, 4], np.float32)

    assert json.loads(calls.call("consumer_page:total", json.dumps({"dtype": "float32", "scale": 10}), [values.tobytes()])) == {"n": 4, "sum": 100.0}
    assert np.frombuffer(calls.call("consumer_page:doubled", "{}", [memoryview(values.tobytes())]), np.float32).tolist() == [2, 4, 6, 8]
    text, arrays = calls.call("consumer_page:split", "{}", [values.tobytes()])
    assert json.loads(text) == {"halves": 2}
    assert [np.frombuffer(a, np.float32).tolist() for a in arrays] == [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="pkg.mod:fn"):
        calls.call("consumer_page", "{}", [])
    with pytest.raises(AttributeError):
        calls.call("consumer_page:missing", "{}", [])
