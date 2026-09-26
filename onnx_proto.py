"""Minimal ONNX protobuf reader. Enough to list nodes and load float weights."""

from __future__ import annotations

import numpy as np
from pathlib import Path


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, i
        shift += 7


def _skip(buf: bytes, i: int, wire: int) -> int:
    if wire == 0:
        _, i = _varint(buf, i)
        return i
    if wire == 1:
        return i + 8
    if wire == 2:
        n, i = _varint(buf, i)
        return i + n
    if wire == 5:
        return i + 4
    raise ValueError(f"unsupported wire type {wire}")


def fields(buf: bytes) -> list[tuple[int, int, bytes | int]]:
    """Return (field_number, wire_type, payload) for one message."""
    out: list[tuple[int, int, bytes | int]] = []
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field = key >> 3
        wire = key & 7
        if wire == 0:
            val, i = _varint(buf, i)
            out.append((field, wire, val))
        elif wire == 1:
            out.append((field, wire, buf[i : i + 8]))
            i += 8
        elif wire == 2:
            length, i = _varint(buf, i)
            out.append((field, wire, buf[i : i + length]))
            i += length
        elif wire == 5:
            out.append((field, wire, buf[i : i + 4]))
            i += 4
        else:
            i = _skip(buf, i, wire)
    return out


def _packed_varints(blob: bytes) -> list[int]:
    vals: list[int] = []
    i = 0
    while i < len(blob):
        v, i = _varint(blob, i)
        vals.append(v)
    return vals


def _zigzag(v: int) -> int:
    return (v >> 1) ^ -(v & 1)


def _signed(v: int) -> int:
    # Protobuf int64 is sign-extended into the varint.
    if v >= 2**63:
        return v - 2**64
    return v


def _string(blob: bytes | int) -> str:
    assert isinstance(blob, bytes)
    return blob.decode()


class Tensor:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array


_DTYPE = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    10: np.float16,
    11: np.float64,
    16: np.float32,  # bfloat16 handled separately
}


def parse_tensor(buf: bytes) -> Tensor:
    name = ""
    dims: list[int] = []
    data_type = 1
    raw: bytes | None = None
    float_data: list[float] = []
    int32_data: list[int] = []
    int64_data: list[int] = []
    for field, wire, val in fields(buf):
        if field == 1:  # dims
            if wire == 0:
                dims.append(_signed(int(val)))
            else:
                dims.extend(_signed(v) for v in _packed_varints(val))  # type: ignore[arg-type]
        elif field == 2 and wire == 0:
            data_type = int(val)
        elif field == 4 and wire == 5:  # float_data
            float_data.append(np.frombuffer(val, dtype="<f4")[0])  # type: ignore[arg-type]
        elif field == 4 and wire == 2:
            float_data.extend(np.frombuffer(val, dtype="<f4").tolist())  # type: ignore[arg-type]
        elif field == 5 and wire == 0:
            int32_data.append(_signed(int(val)))
        elif field == 5 and wire == 2:
            int32_data.extend(np.frombuffer(val, dtype="<i4").tolist())  # type: ignore[arg-type]
        elif field == 7 and wire == 0:
            int64_data.append(_signed(int(val)))
        elif field == 7 and wire == 2:
            int64_data.extend(_signed(v) for v in _packed_varints(val))  # type: ignore[arg-type]
        elif field == 8 and wire == 2:
            name = _string(val)
        elif field == 9 and wire == 2:
            raw = val  # type: ignore[assignment]
    if raw is not None:
        if data_type == 16:
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            array = u.view(np.float32)
        else:
            array = np.frombuffer(raw, dtype=_DTYPE[data_type])
    elif float_data:
        array = np.asarray(float_data, dtype=np.float32)
    elif int32_data:
        array = np.asarray(int32_data, dtype=np.int32)
    elif int64_data:
        array = np.asarray(int64_data, dtype=np.int64)
    else:
        array = np.zeros(int(np.prod(dims)) if dims else 0, dtype=_DTYPE.get(data_type, np.float32))
    if dims:
        array = array.reshape(dims)
    return Tensor(name, np.array(array, copy=True))


class Attribute:
    def __init__(self) -> None:
        self.name = ""
        self.f: float | None = None
        self.i: int | None = None
        self.s: bytes | None = None
        self.floats: list[float] = []
        self.ints: list[int] = []
        self.strings: list[bytes] = []


def parse_attribute(buf: bytes) -> Attribute:
    attr = Attribute()
    for field, wire, val in fields(buf):
        if field == 1 and wire == 2:
            attr.name = _string(val)
        elif field == 2 and wire == 5:
            attr.f = float(np.frombuffer(val, dtype="<f4")[0])  # type: ignore[arg-type]
        elif field == 3 and wire == 0:
            attr.i = _signed(int(val))
        elif field == 4 and wire == 2:
            attr.s = val  # type: ignore[assignment]
        elif field == 7 and wire == 5:
            attr.floats.append(float(np.frombuffer(val, dtype="<f4")[0]))  # type: ignore[arg-type]
        elif field == 7 and wire == 2:
            attr.floats.extend(np.frombuffer(val, dtype="<f4").tolist())  # type: ignore[arg-type]
        elif field == 8 and wire == 0:
            attr.ints.append(_signed(int(val)))
        elif field == 8 and wire == 2:
            attr.ints.extend(_signed(v) for v in _packed_varints(val))  # type: ignore[arg-type]
        elif field == 9 and wire == 2:
            attr.strings.append(val)  # type: ignore[arg-type]
    return attr


class Node:
    def __init__(self) -> None:
        self.op_type = ""
        self.name = ""
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.attrs: dict[str, Attribute] = {}


def parse_node(buf: bytes) -> Node:
    node = Node()
    for field, wire, val in fields(buf):
        if field == 1 and wire == 2:
            node.inputs.append(_string(val))
        elif field == 2 and wire == 2:
            node.outputs.append(_string(val))
        elif field == 3 and wire == 2:
            node.name = _string(val)
        elif field == 4 and wire == 2:
            node.op_type = _string(val)
        elif field == 5 and wire == 2:
            attr = parse_attribute(val)  # type: ignore[arg-type]
            node.attrs[attr.name] = attr
    return node


class ValueInfo:
    def __init__(self, name: str, shape: list[int | str], dtype: int) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


def parse_value_info(buf: bytes) -> ValueInfo:
    name = ""
    dtype = 1
    shape: list[int | str] = []
    for field, wire, val in fields(buf):
        if field == 1 and wire == 2:
            name = _string(val)
        elif field == 2 and wire == 2:
            for tf, tw, tv in fields(val):  # type: ignore[arg-type]
                if tf == 1 and tw == 2:  # tensor_type
                    for ef, ew, ev in fields(tv):  # type: ignore[arg-type]
                        if ef == 1 and ew == 0:
                            dtype = int(ev)
                        elif ef == 2 and ew == 2:
                            for df, dw, dv in fields(ev):  # type: ignore[arg-type]
                                if df == 1 and dw == 2:
                                    dim_val: int | str | None = None
                                    for sf, sw, sv in fields(dv):  # type: ignore[arg-type]
                                        if sf == 1 and sw == 0:
                                            dim_val = _signed(int(sv))
                                        elif sf == 2 and sw == 2:
                                            dim_val = _string(sv)
                                    shape.append(dim_val if dim_val is not None else -1)
    return ValueInfo(name, shape, dtype)


class Graph:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.initializers: dict[str, np.ndarray] = {}
        self.inputs: list[ValueInfo] = []
        self.outputs: list[ValueInfo] = []


def parse_graph(buf: bytes) -> Graph:
    graph = Graph()
    for field, wire, val in fields(buf):
        if wire != 2:
            continue
        if field == 1:
            graph.nodes.append(parse_node(val))  # type: ignore[arg-type]
        elif field == 5:
            tensor = parse_tensor(val)  # type: ignore[arg-type]
            graph.initializers[tensor.name] = tensor.array
        elif field == 11:
            graph.inputs.append(parse_value_info(val))  # type: ignore[arg-type]
        elif field == 12:
            graph.outputs.append(parse_value_info(val))  # type: ignore[arg-type]
    return graph


def load(path: str | Path) -> Graph:
    data = Path(path).read_bytes()
    for field, wire, val in fields(data):
        if field == 7 and wire == 2:
            return parse_graph(val)  # type: ignore[arg-type]
    raise ValueError(f"no graph in {path}")
