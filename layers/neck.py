"""Inference DB head and nearest upsampling shared with the detection neck.

Two stride-2 transposed convolutions restore the input resolution.
"""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.layer import Module

from layers.activations import relu
from layers.conv import ConvTranspose2x2, conv2d


def _upsample(x: TensorValue, factor: int) -> TensorValue:
    if factor == 1:
        return x
    shape = x.shape
    return ops.resize_nearest(
        x,
        [shape[0], shape[1], shape[2] * factor, shape[3] * factor],
        coordinate_transform_mode=2,
        round_mode=2,
    )


class DBHead(Module):
    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.conv = conv2d(in_channels, hidden, 3, padding=1, bias=True)
        self.up1 = ConvTranspose2x2(hidden, hidden, bias=True)
        self.up2 = ConvTranspose2x2(hidden, 1, bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        x = relu(self.conv(x))
        x = relu(self.up1(x))
        return ops.sigmoid(self.up2(x))
