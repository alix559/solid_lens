"""Activations used by the exported PP-OCRv6 graphs."""

from __future__ import annotations

import numpy as np
from max.graph import TensorValue, ops

from model_config import DEVICE, DTYPE


def hardsigmoid(x: TensorValue, alpha: float, beta: float = 0.5) -> TensorValue:
    zero = ops.constant(np.float32(0), DTYPE, device=DEVICE)
    one = ops.constant(np.float32(1), DTYPE, device=DEVICE)
    return ops.min(one, ops.max(zero, x * np.float32(alpha) + np.float32(beta)))


def hardswish(x: TensorValue, alpha: float) -> TensorValue:
    return x * hardsigmoid(x, alpha)


def gelu(x: TensorValue) -> TensorValue:
    return ops.gelu(x)


def relu(x: TensorValue) -> TensorValue:
    return ops.relu(x)
