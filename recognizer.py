"""PP-OCRv6 medium text recognizer.

LCNetV4 plus the LightSVTR transformer neck and a CTC head.
"""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear

from layers.conv import nchw_to_nhwc, nhwc_to_nchw
from layers.lcnet import DetStem, stage
from layers.svtr import LightSVTR
from model_config import DEVICE, DTYPE, REC_BLOCKS, RecognizerConfig


class PPOCRV6Recognizer(Module):
    def __init__(self, config: RecognizerConfig | None = None) -> None:
        super().__init__()
        config = config or RecognizerConfig()
        self.stem = DetStem(config.stem_mid, config.stem_out)
        self.stage2 = stage(REC_BLOCKS["b2"], config.se_alpha, bias=True)
        self.stage3 = stage(REC_BLOCKS["b3"], config.se_alpha, bias=True)
        self.stage4 = stage(REC_BLOCKS["b4"], config.se_alpha, bias=True)
        self.stage5 = stage(REC_BLOCKS["b5"], config.se_alpha, bias=True)
        self.neck = LightSVTR(
            config.backbone_channels,
            config.svtr_dim,
            config.svtr_depth,
            config.svtr_heads,
            config.mlp_ratio,
            config.local_kernel,
            config.bn_eps,
            config.final_norm_eps,
        )
        self.head = Linear(
            config.svtr_dim,
            config.num_classes,
            DTYPE,
            DEVICE,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.stage2(self.stem(x))
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        sequence = self.neck(avg_pool_height(x))
        return ops.softmax(self.head(sequence), axis=2)


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
