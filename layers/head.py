"""CTC head used by PP-OCRv6 tiny recognition at inference."""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear

from layers.activations import hardswish
from layers.conv import conv2d, nchw_to_nhwc, nhwc_to_nchw
from layers.lcnet import BatchNorm
from model_config import DEVICE, DTYPE


class CTCHead(Module):
    def __init__(
        self,
        channels: int,
        mid_channels: int,
        num_classes: int,
        alpha: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.dw = conv2d(
            channels,
            channels,
            (1, 5),
            padding=(0, 0, 2, 2),
            groups=channels,
            bias=False,
        )
        self.bn1 = BatchNorm(channels, eps)
        self.pw = conv2d(channels, channels, 1, bias=False)
        self.bn2 = BatchNorm(channels, eps)
        self.fc1 = Linear(channels, mid_channels, DTYPE, DEVICE, has_bias=True)
        self.fc2 = Linear(mid_channels, num_classes, DTYPE, DEVICE, has_bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        # x is NCHW with height 1 after the backbone pool.
        x = hardswish(self.bn1(self.dw(x)), self.alpha)
        x = hardswish(self.bn2(self.pw(x)), self.alpha)
        sequence = ops.permute(ops.squeeze(x, axis=2), [0, 2, 1])
        logits = self.fc2(self.fc1(sequence))
        return ops.softmax(logits, axis=2)


def avg_pool_height(x: TensorValue) -> TensorValue:
    """3x2 average pool. Recognition features are one row tall afterwards."""
    y = ops.avg_pool2d(
        nchw_to_nhwc(x),
        kernel_size=(3, 2),
        stride=(3, 2),
        padding=0,
        count_boundary=False,
    )
    y = nhwc_to_nchw(y)
    return ops.unsqueeze(ops.squeeze(y, axis=2), axis=2)
