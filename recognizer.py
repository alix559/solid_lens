"""PP-OCRv6 tiny text recognizer."""

from __future__ import annotations

from max.graph import TensorValue
from max.nn.layer import Module

from layers.head import CTCHead, avg_pool_height
from layers.lcnet import RecStem, stage
from model_config import REC_BLOCKS, RecognizerConfig


class PPOCRV6TinyRecognizer(Module):
    def __init__(self, config: RecognizerConfig | None = None) -> None:
        super().__init__()
        config = config or RecognizerConfig()
        self.stem = RecStem(config.stem_mid, config.stem_out, config.bn_eps)
        self.stage2 = stage(REC_BLOCKS["b2"], config.se_alpha, bias=True)
        self.stage3 = stage(REC_BLOCKS["b3"], config.se_alpha, bias=True)
        self.stage4 = stage(REC_BLOCKS["b4"], config.se_alpha, bias=True)
        self.stage5 = stage(REC_BLOCKS["b5"], config.se_alpha, bias=True)
        self.head = CTCHead(
            config.guide_channels,
            config.mid_channels,
            config.num_classes,
            config.se_alpha,
            config.bn_eps,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        x = self.stage2(self.stem(x))
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return self.head(avg_pool_height(x))
