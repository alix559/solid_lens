"""PP-OCRv6 medium architecture, taken from the PaddleOCR training configs.

Detection uses PPLCNetV4, a RepLKPAN neck with IntraCL, and the inference
DB head. Recognition uses the same backbone with height-only downsampling,
then an EncoderWithLightSVTR neck: a 1x7 local convolution plus global
transformer blocks, and a CTC head. The NRTR head exists only while training.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef

DEVICE = DeviceRef.CPU()
DTYPE = DType.float32


# kernel, in_channels, out_channels, stride, use_se
Block = tuple[int, int, int, int | tuple[int, int], bool]


DET_BLOCKS: dict[str, list[Block]] = {
    "s1": [(3, 128, 128, 1, True), (3, 128, 128, 1, False)],
    "s2": [
        (3, 128, 256, 2, False),
        (3, 256, 256, 1, True),
        (3, 256, 256, 1, False),
    ],
    "s3": [
        (3, 256, 512, 2, False),
        (3, 512, 512, 1, True),
        (3, 512, 512, 1, False),
        (3, 512, 512, 1, True),
        (3, 512, 512, 1, False),
    ],
    "s4": [
        (3, 512, 896, 2, False),
        (3, 896, 896, 1, True),
        (3, 896, 896, 1, False),
    ],
}

REC_BLOCKS: dict[str, list[Block]] = {
    "b2": [(3, 128, 128, 1, True)],
    "b3": [
        (3, 128, 256, 1, False),
        (3, 256, 256, 1, False),
        (3, 256, 256, 1, True),
    ],
    "b4": [
        (3, 256, 512, (2, 1), False),
        (3, 512, 512, 1, True),
        (3, 512, 512, 1, False),
        (3, 512, 512, 1, True),
        (3, 512, 512, 1, False),
        (3, 512, 512, 1, True),
        (3, 512, 512, 1, False),
    ],
    "b5": [
        (3, 512, 768, (2, 1), False),
        (3, 768, 768, 1, True),
        (3, 768, 768, 1, False),
    ],
}


@dataclass(frozen=True)
class DetectorConfig:
    stem_mid: int = 64
    stem_out: int = 128
    # RepLKPAN projects every level to this width, then to width // 4.
    fpn_channels: int = 256
    replk_kernel: int = 9
    head_channels: int = 64
    se_alpha: float = 1.0 / 6.0


@dataclass(frozen=True)
class RecognizerConfig:
    stem_mid: int = 64
    stem_out: int = 128
    backbone_channels: int = 768
    svtr_dim: int = 192
    svtr_depth: int = 2
    svtr_heads: int = 8
    mlp_ratio: float = 4.0
    local_kernel: int = 7
    num_classes: int = 18710
    image_height: int = 48
    image_width: int = 320
    se_alpha: float = 1.0 / 6.0
    # Conv batch-norm and the transformer block norms both use 1e-5.
    # The norm after the last block uses 1e-6.
    bn_eps: float = 1e-5
    final_norm_eps: float = 1e-6
