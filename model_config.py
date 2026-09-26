"""PP-OCRv6 tiny architecture, taken from the PaddleOCR training configs.

Detection uses PPLCNetV4, a RepLKFPN neck, and the inference DB head.
Recognition uses the same backbone with height-only downsampling, a
depthwise guide, and a CTC head. The NRTR head exists only while training.
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
    "s1": [(3, 32, 32, 1, True), (3, 32, 32, 1, False)],
    "s2": [
        (3, 32, 48, 2, False),
        (3, 48, 48, 1, True),
        (3, 48, 48, 1, False),
    ],
    "s3": [
        (3, 48, 64, 2, False),
        (3, 64, 64, 1, True),
        (3, 64, 64, 1, False),
        (3, 64, 64, 1, True),
        (3, 64, 64, 1, False),
    ],
    "s4": [
        (3, 64, 160, 2, False),
        (3, 160, 160, 1, True),
        (3, 160, 160, 1, False),
    ],
}

REC_BLOCKS: dict[str, list[Block]] = {
    "b2": [(3, 48, 48, 1, True)],
    "b3": [(3, 48, 48, 1, False)],
    "b4": [
        (3, 48, 96, (2, 1), False),
        (3, 96, 96, 1, True),
        (3, 96, 96, 1, False),
    ],
    "b5": [
        (3, 96, 160, (2, 1), False),
        (3, 160, 160, 1, True),
        (3, 160, 160, 1, False),
        (3, 160, 160, 1, False),
    ],
}


@dataclass(frozen=True)
class DetectorConfig:
    stem_mid: int = 16
    stem_out: int = 32
    fpn_channels: int = 64
    replk_kernel: int = 5
    head_channels: int = 16
    # Backbone SE uses Paddle's Hardsigmoid slope. The neck uses 1/5.
    se_alpha: float = 1.0 / 6.0
    neck_alpha: float = 0.2


@dataclass(frozen=True)
class RecognizerConfig:
    stem_mid: int = 24
    stem_out: int = 48
    guide_channels: int = 160
    mid_channels: int = 80
    num_classes: int = 6906
    image_height: int = 48
    image_width: int = 320
    se_alpha: float = 1.0 / 6.0
    bn_eps: float = 1e-5
