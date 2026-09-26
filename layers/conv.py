"""Convolution layers whose weights match the ONNX checkpoint layout.

``max.nn.Conv2d(permute=True)`` keeps filters in OIHW order, which is how
Paddle exported them. Stride-2 kernel-2 transposed convolutions do not
compile in this MAX build (``num_groups`` fails to infer). With the kernel
equal to the stride and no padding, each input pixel becomes a 2x2 block
and the four taps do not overlap, so the transpose is four pointwise mixes
interleaved on the spatial axes.
"""

from __future__ import annotations

from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.conv import Conv2d
from max.nn.layer import Module

from model_config import DEVICE, DTYPE


def conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int, int, int] = 0,
    groups: int = 1,
    bias: bool = True,
    device: DeviceRef = DEVICE,
) -> Conv2d:
    return Conv2d(
        kernel_size,
        in_channels,
        out_channels,
        DTYPE,
        stride=stride,
        padding=padding,
        num_groups=groups,
        device=device,
        has_bias=bias,
        permute=True,
    )


def same_padding(kernel: int) -> int:
    return kernel // 2


class ConvTranspose2x2(Module):
    """Stride-2, kernel-2, pad-0 conv-transpose, written as interleaved matmuls."""

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Weight(
            "weight",
            DTYPE,
            (in_channels, out_channels, 2, 2),
            DEVICE,
        )
        self.bias = (
            Weight("bias", DTYPE, (out_channels,), DEVICE) if bias else None
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x_nhwc = ops.permute(x, [0, 2, 3, 1])
        rows = []
        for iy in range(2):
            pair = []
            for ix in range(2):
                tap = ops.reshape(
                    self.weight[:, :, iy, ix],
                    [self.in_channels, self.out_channels],
                )
                pair.append(ops.matmul(x_nhwc, tap))
            rows.append(_interleave(pair[0], pair[1], axis=2))
        y = ops.permute(_interleave(rows[0], rows[1], axis=1), [0, 3, 1, 2])
        if self.bias is not None:
            y = y + ops.reshape(self.bias, [1, self.out_channels, 1, 1])
        return y


def _interleave(a: TensorValue, b: TensorValue, axis: int) -> TensorValue:
    """Interleave two NHWC tensors along height (axis 1) or width (axis 2)."""
    stacked = ops.concat(
        [ops.unsqueeze(a, axis + 1), ops.unsqueeze(b, axis + 1)],
        axis=axis + 1,
    )
    shape = stacked.shape
    if axis == 2:
        return ops.reshape(stacked, [shape[0], shape[1], shape[2] * 2, shape[4]])
    return ops.reshape(stacked, [shape[0], shape[1] * 2, shape[3], shape[4]])


def nchw_to_nhwc(x: TensorValue) -> TensorValue:
    return ops.permute(x, [0, 2, 3, 1])


def nhwc_to_nchw(x: TensorValue) -> TensorValue:
    return ops.permute(x, [0, 3, 1, 2])
