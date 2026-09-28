"""RepLKPAN neck used by PP-OCRv6 medium detection.

The checkpoint is the reparameterized inference graph. Each dilated
large-kernel branch is one depthwise 9x9 plus a pointwise mix, and the
IntraCL batch-norm is folded into the closing 1x1. Submodules are stored
in the order the exporter wrote them.
"""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.layer import Module

from layers.activations import relu
from layers.conv import conv2d, same_padding
from layers.neck import _upsample


class LargeKernel(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        self.dw = conv2d(
            in_channels,
            in_channels,
            kernel,
            padding=same_padding(kernel),
            groups=in_channels,
            bias=True,
        )
        self.pw = conv2d(in_channels, out_channels, 1, bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.pw(self.dw(x))


class IntraCL(Module):
    """Strip and square context at 7, 5, and 3, added back onto the input."""

    def __init__(self, channels: int, reduce_factor: int = 2) -> None:
        super().__init__()
        hidden = channels // reduce_factor
        self.reduce = conv2d(channels, hidden, 1, bias=True)
        self.c7 = conv2d(hidden, hidden, 7, padding=3, bias=True)
        self.v7 = conv2d(hidden, hidden, (7, 1), padding=(3, 3, 0, 0), bias=True)
        self.q7 = conv2d(hidden, hidden, (1, 7), padding=(0, 0, 3, 3), bias=True)
        self.c5 = conv2d(hidden, hidden, 5, padding=2, bias=True)
        self.v5 = conv2d(hidden, hidden, (5, 1), padding=(2, 2, 0, 0), bias=True)
        self.q5 = conv2d(hidden, hidden, (1, 5), padding=(0, 0, 2, 2), bias=True)
        self.c3 = conv2d(hidden, hidden, 3, padding=1, bias=True)
        self.v3 = conv2d(hidden, hidden, (3, 1), padding=(1, 1, 0, 0), bias=True)
        self.q3 = conv2d(hidden, hidden, (1, 3), padding=(0, 0, 1, 1), bias=True)
        self.proj = conv2d(hidden, channels, 1, bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        hidden = self.reduce(x)
        hidden = self.c7(hidden) + self.v7(hidden) + self.q7(hidden)
        hidden = self.c5(hidden) + self.v5(hidden) + self.q5(hidden)
        hidden = self.c3(hidden) + self.v3(hidden) + self.q3(hidden)
        return x + relu(self.proj(hidden))


class RepLKPAN(Module):
    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int, kernel: int) -> None:
        super().__init__()
        fine, mid, low, coarse = in_channels
        inner = out_channels // 4
        # Coarsest lateral first. That is the order of the checkpoint.
        self.lateral5 = conv2d(coarse, out_channels, 1, bias=False)
        self.lateral4 = conv2d(low, out_channels, 1, bias=False)
        self.lateral3 = conv2d(mid, out_channels, 1, bias=False)
        self.lateral2 = conv2d(fine, out_channels, 1, bias=False)
        self.mix5 = LargeKernel(out_channels, inner, kernel)
        self.mix4 = LargeKernel(out_channels, inner, kernel)
        self.mix3 = LargeKernel(out_channels, inner, kernel)
        self.mix2 = LargeKernel(out_channels, inner, kernel)
        self.down3 = conv2d(inner, inner, 3, stride=2, padding=1, bias=False)
        self.down4 = conv2d(inner, inner, 3, stride=2, padding=1, bias=False)
        self.down5 = conv2d(inner, inner, 3, stride=2, padding=1, bias=False)
        self.lat2 = LargeKernel(inner, inner, kernel)
        self.lat3 = LargeKernel(inner, inner, kernel)
        self.lat4 = LargeKernel(inner, inner, kernel)
        self.lat5 = LargeKernel(inner, inner, kernel)
        self.incl5 = IntraCL(inner)
        self.incl4 = IntraCL(inner)
        self.incl3 = IntraCL(inner)
        self.incl2 = IntraCL(inner)

    def __call__(
        self, features: tuple[TensorValue, TensorValue, TensorValue, TensorValue]
    ) -> TensorValue:
        fine, mid, low, coarse = features
        top = self.lateral5(coarse)
        low_lat = self.lateral4(low)
        mid_lat = self.lateral3(mid)
        fine_lat = self.lateral2(fine)
        level4 = low_lat + _upsample(top, 2)
        level3 = mid_lat + _upsample(level4, 2)
        level2 = fine_lat + _upsample(level3, 2)

        feat5 = self.mix5(top)
        feat4 = self.mix4(level4)
        feat3 = self.mix3(level3)
        feat2 = self.mix2(level2)
        pan3 = feat3 + self.down3(feat2)
        pan4 = feat4 + self.down4(pan3)
        pan5 = feat5 + self.down5(pan4)

        out2 = self.incl2(self.lat2(feat2))
        out3 = self.incl3(self.lat3(pan3))
        out4 = self.incl4(self.lat4(pan4))
        out5 = self.incl5(self.lat5(pan5))
        return ops.concat(
            [
                _upsample(out5, 8),
                _upsample(out4, 4),
                _upsample(out3, 2),
                out2,
            ],
            axis=1,
        )
