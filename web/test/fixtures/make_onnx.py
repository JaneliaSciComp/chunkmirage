"""The two ONNX models test/onnx.test.ts runs through ONNX Runtime's WebAssembly backend,
written here as protobuf by hand (the onnx package is not a dependency of anything in this
repository): ``double.onnx`` multiplies its input by 2, ``pair.onnx`` concatenates its input
with itself along axis 1, two channels out. Both take float32 ``[1, 1, z, y, x]`` with z, y
and x free. ``python web/test/fixtures/make_onnx.py`` writes them beside this file."""

import struct
from pathlib import Path


def varint(n: int) -> bytes:
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        if not n:
            return out + bytes([b])
        out += bytes([b | 0x80])


def field(num: int, wire: int, payload: bytes) -> bytes:
    return varint((num << 3) | wire) + payload


def vint(num: int, n: int) -> bytes:
    return field(num, 0, varint(n))


def ld(num: int, b: bytes) -> bytes:
    return field(num, 2, varint(len(b)) + b)


def s(num: int, text: str) -> bytes:
    return ld(num, text.encode())


def dims(*names) -> bytes:
    """TensorShapeProto: dim (1) of Dimension { dim_value (1) | dim_param (2) }."""
    return b"".join(ld(1, vint(1, d) if isinstance(d, int) else s(2, d)) for d in names)


def tensor_value(name: str, *shape) -> bytes:
    """ValueInfoProto { name (1), type (2) } of a float32 tensor of ``shape``."""
    tensor = vint(1, 1) + ld(2, dims(*shape))  # TypeProto.Tensor { elem_type (1) = FLOAT, shape (2) }
    return ld(1, name.encode()) + ld(2, ld(1, tensor))  # TypeProto.tensor_type is field 1


def node(op: str, inputs, outputs, attrs: bytes = b"") -> bytes:
    """NodeProto { input (1)..., output (2)..., op_type (4), attribute (5)... }."""
    return b"".join(s(1, i) for i in inputs) + b"".join(s(2, o) for o in outputs) + s(4, op) + attrs


def attr_int(name: str, value: int) -> bytes:
    """AttributeProto { name (1), i (3), type (20) = INT }, as a NodeProto's field 5."""
    return ld(5, s(1, name) + vint(3, value) + vint(20, 2))


def scalar(name: str, value: float) -> bytes:
    """A float32 scalar TensorProto { data_type (2), name (8), raw_data (9) } as a GraphProto initializer (5)."""
    return ld(5, vint(2, 1) + s(8, name) + ld(9, struct.pack("<f", value)))


def model(graph_name: str, nodes: bytes, initializers: bytes, inputs: bytes, outputs: bytes) -> bytes:
    """ModelProto { ir_version (1) = 8, producer_name (2), graph (7), opset_import (8) = 13 }."""
    graph = nodes + s(2, graph_name) + initializers + inputs + outputs  # node (1), name (2), initializer (5), input (11), output (12)
    opset = ld(8, s(1, "") + vint(2, 13))  # OperatorSetIdProto { domain (1), version (2) }
    return vint(1, 8) + s(2, "chunkmirage-test") + ld(7, graph) + opset


if __name__ == "__main__":
    here = Path(__file__).parent
    x = ld(11, tensor_value("x", 1, 1, "z", "y", "x"))
    (here / "double.onnx").write_bytes(model(
        "double", ld(1, node("Mul", ["x", "two"], ["y"])), scalar("two", 2.0), x, ld(12, tensor_value("y", 1, 1, "z", "y", "x"))))
    (here / "pair.onnx").write_bytes(model(
        "pair", ld(1, node("Concat", ["x", "x"], ["y"], attr_int("axis", 1))), b"", x, ld(12, tensor_value("y", 1, 2, "z", "y", "x"))))
    for f in sorted(here.glob("*.onnx")):
        print(f.name, f.stat().st_size, "bytes")
