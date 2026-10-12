"""A hybrid OCR engine inspired by the PaddleOCR v6 architecture

Detection is PPLCNetV4, a RepLKPAN neck, and the inference DB head. Recognition uses the same backbone with height-only downsampling, a LightSVTR neck, and CTC. The official ONNX checkpoints supply the weights. The notebooks in `nbs/` are the source; `nbdev-export` writes this package."""

__version__ = "0.0.1"
