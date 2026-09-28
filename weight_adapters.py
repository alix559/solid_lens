"""Map an exported PP-OCRv6 ONNX checkpoint onto MAX module weights.

Parameter order follows the module tree, which is the same order the
Paddle exporter emitted Conv, ConvTranspose, BatchNormalization, MatMul,
and the decomposed LayerNorm scale and bias. Paddle linear weights are
stored as (in, out); MAX linear weights are (out, in).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from onnx_proto import Graph, Node, load


def _skip_identity(nodes: list[Node], index: int) -> int:
    while index < len(nodes) and nodes[index].op_type == "Identity":
        index += 1
    return index


def _vector(graph: Graph, node: Node) -> np.ndarray | None:
    """A 1-D learned vector on this node, such as a LayerNorm scale or bias."""
    for name in node.inputs:
        if name not in graph.initializers:
            continue
        array = np.asarray(graph.initializers[name], dtype=np.float32)
        if array.ndim == 1 and array.size > 1:
            return np.ascontiguousarray(array.reshape(-1))
    return None


def _bias_array(graph: Graph, nodes: list[Node], index: int) -> np.ndarray | None:
    nxt = _skip_identity(nodes, index + 1)
    if nxt >= len(nodes) or nodes[nxt].op_type != "Add":
        return None
    rhs = nodes[nxt].inputs[1]
    if rhs not in graph.initializers:
        return None
    array = np.asarray(graph.initializers[rhs], dtype=np.float32)
    if array.size <= 1 and array.ndim == 1:
        return None
    return np.ascontiguousarray(array.reshape(-1))


def checkpoint_arrays(path: Path) -> list[np.ndarray]:
    """Learned tensors in module state-dict order."""
    graph = load(path)
    nodes = graph.nodes
    arrays: list[np.ndarray] = []
    for index, node in enumerate(nodes):
        if node.op_type in ("Conv", "ConvTranspose"):
            arrays.append(np.ascontiguousarray(graph.initializers[node.inputs[1]], dtype=np.float32))
            if len(node.inputs) > 2 and node.inputs[2]:
                bias = graph.initializers[node.inputs[2]]
                arrays.append(np.ascontiguousarray(np.asarray(bias, dtype=np.float32).reshape(-1)))
            else:
                bias = _bias_array(graph, nodes, index)
                if bias is not None:
                    arrays.append(bias)
        elif node.op_type == "BatchNormalization":
            for name in node.inputs[1:5]:
                arrays.append(
                    np.ascontiguousarray(np.asarray(graph.initializers[name], dtype=np.float32).reshape(-1))
                )
        elif node.op_type == "MatMul" and node.inputs[1] in graph.initializers:
            weight = np.asarray(graph.initializers[node.inputs[1]], dtype=np.float32)
            arrays.append(np.ascontiguousarray(weight.T))
            bias = _bias_array(graph, nodes, index)
            if bias is None:
                raise ValueError(f"{node.name} is missing a bias")
            arrays.append(bias)
        elif node.op_type == "Mul":
            # Exported LayerNorm is mean/var, then scale (Mul) and bias (Add).
            gamma = _vector(graph, node)
            if gamma is None:
                continue
            nxt = _skip_identity(nodes, index + 1)
            if nxt >= len(nodes) or nodes[nxt].op_type != "Add":
                raise ValueError(f"{node.name} scale is missing a LayerNorm bias")
            beta = _vector(graph, nodes[nxt])
            if beta is None:
                raise ValueError(f"{node.name} scale is missing a LayerNorm bias")
            arrays.append(gamma)
            arrays.append(beta)
    return arrays


def _weights_in_definition_order(module, prefix: str = ""):
    """Depth-first weights. This matches the order the exporter wrote them."""
    for weight_name, weight in module.layer_weights.items():
        if weight_name in module._shared_weights:
            continue
        yield f"{prefix}{weight_name}", weight
    for local_name, child in module.sublayers.items():
        child_prefix = prefix if child._omit_module_attr_name else f"{prefix}{local_name}."
        yield from _weights_in_definition_order(child, child_prefix)


def load_checkpoint(module, path: Path) -> None:
    arrays = checkpoint_arrays(path)
    slots = list(_weights_in_definition_order(module))
    if len(arrays) != len(slots):
        raise ValueError(
            f"{path.name}: checkpoint has {len(arrays)} tensors, the MAX module has {len(slots)}"
        )
    state = {}
    for (name, weight), array in zip(slots, arrays, strict=True):
        expected = tuple(int(dim) for dim in weight.shape)
        if array.shape != expected:
            raise ValueError(f"{name}: checkpoint shape {array.shape} != module shape {expected}")
        state[name] = array
    module.load_state_dict(state, strict=True)
