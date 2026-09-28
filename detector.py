"""PP-OCRv6 medium text detector."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.layer import Module

from layers.lcnet import DetStem, stage
from layers.neck import DBHead
from layers.pan import RepLKPAN
from model_config import DET_BLOCKS, DetectorConfig


class PPOCRV6Detector(Module):
    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        config = config or DetectorConfig()
        self.stem = DetStem(config.stem_mid, config.stem_out)
        self.stage1 = stage(DET_BLOCKS["s1"], config.se_alpha, bias=True)
        self.stage2 = stage(DET_BLOCKS["s2"], config.se_alpha, bias=True)
        self.stage3 = stage(DET_BLOCKS["s3"], config.se_alpha, bias=True)
        self.stage4 = stage(DET_BLOCKS["s4"], config.se_alpha, bias=True)
        self.neck = RepLKPAN((128, 256, 512, 896), config.fpn_channels, config.replk_kernel)
        self.head = DBHead(config.fpn_channels, config.head_channels)

    def __call__(self, x: TensorValue) -> TensorValue:
        fine = self.stage1(self.stem(x))
        mid = self.stage2(fine)
        low = self.stage3(mid)
        coarse = self.stage4(low)
        return self.head(self.neck((fine, mid, low, coarse)))
