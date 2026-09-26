"""RepLKFPN neck and the inference DB head.

The neck projects each backbone level to 64 channels, runs a residual SE,
then upsamples from coarse to fine. A fused 5x5 depthwise block (the
reparameterized dilated kernel) reduces every level to 16 channels. Those
maps are resized to the finest stride and concatenated for the DB head.
"""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.layer import Module

from layers.activations import relu
from layers.conv import ConvTranspose2x2, conv2d, same_padding
from layers.lcnet import SqueezeExcite


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


class Lateral(Module):
    def __init__(self, in_channels: int, out_channels: int, alpha: float) -> None:
        super().__init__()
        self.proj = conv2d(in_channels, out_channels, 1, bias=False)
        self.se = SqueezeExcite(out_channels, alpha, residual=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.se(self.proj(x))


class RepLKBlock(Module):
    def __init__(self, channels: int, kernel: int, out_channels: int, alpha: float) -> None:
        super().__init__()
        self.dw = conv2d(
            channels,
            channels,
            kernel,
            padding=same_padding(kernel),
            groups=channels,
            bias=True,
        )
        self.proj = conv2d(channels, out_channels, 1, bias=False)
        self.se = SqueezeExcite(out_channels, alpha, residual=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        return self.se(self.proj(self.dw(x)))


class RepLKFPN(Module):
    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        out_channels: int,
        kernel: int,
        reduced: int,
        alpha: float,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = in_channels
        # Coarsest level first: that is the order of the checkpoint.
        self.lateral4 = Lateral(c4, out_channels, alpha)
        self.lateral3 = Lateral(c3, out_channels, alpha)
        self.lateral2 = Lateral(c2, out_channels, alpha)
        self.lateral1 = Lateral(c1, out_channels, alpha)
        self.replk4 = RepLKBlock(out_channels, kernel, reduced, alpha)
        self.replk3 = RepLKBlock(out_channels, kernel, reduced, alpha)
        self.replk2 = RepLKBlock(out_channels, kernel, reduced, alpha)
        self.replk1 = RepLKBlock(out_channels, kernel, reduced, alpha)

    def __call__(self, features: tuple[TensorValue, TensorValue, TensorValue, TensorValue]) -> TensorValue:
        fine, mid, low, coarse = features
        p4 = self.lateral4(coarse)
        p3 = self.lateral3(low) + _upsample(p4, 2)
        p2 = self.lateral2(mid) + _upsample(p3, 2)
        p1 = self.lateral1(fine) + _upsample(p2, 2)
        maps = [
            _upsample(self.replk4(p4), 8),
            _upsample(self.replk3(p3), 4),
            _upsample(self.replk2(p2), 2),
            self.replk1(p1),
        ]
        return ops.concat(maps, axis=1)


class DBHead(Module):
    """Two stride-2 transposed convolutions restore the input resolution."""

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        self.conv = conv2d(in_channels, hidden, 3, padding=1, bias=True)
        self.up1 = ConvTranspose2x2(hidden, hidden, bias=True)
        self.up2 = ConvTranspose2x2(hidden, 1, bias=True)

    def __call__(self, x: TensorValue) -> TensorValue:
        x = relu(self.conv(x))
        x = relu(self.up1(x))
        return ops.sigmoid(self.up2(x))
