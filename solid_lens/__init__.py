"""A hybrid OCR engine inspired by the PaddleOCR v6 architecture

Detection is PPLCNetV4, a RepLKPAN neck, and the inference DB head. Recognition uses the same backbone with height-only downsampling, a LightSVTR neck, and CTC. The official ONNX checkpoints supply the weights. The notebooks in `nbs/` are the source; `nbdev-export` writes this package.

Modules:

- `solid_lens.config`: PP-OCRv6 medium, as the PaddleOCR training configs describe it.
- `solid_lens.core`: Find the checkout that holds the weights.
- `solid_lens.detector`: PP-OCRv6 medium text detector.
- `solid_lens.download`: Fetch the ONNX checkpoints named in `requirements.txt`.
- `solid_lens.layers.activations`: Activations used by the exported PP-OCRv6 graphs.
- `solid_lens.layers.conv`: Convolutions whose weights match the ONNX checkpoint layout.
- `solid_lens.layers.lcnet`: PPLCNetV4 blocks for the medium detector and recognizer.
- `solid_lens.layers.neck`: Inference DB head, and the nearest upsample the detection neck shares.
- `solid_lens.layers.pan`: RepLKPAN neck used by PP-OCRv6 medium detection.
- `solid_lens.layers.svtr`: EncoderWithLightSVTR, the PP-OCRv6 medium recognition neck.
- `solid_lens.model`: Compile the PP-OCRv6 medium modules and bind their checkpoints.
- `solid_lens.ocr`: Read a page with the PP-OCRv6 medium detector and recognizer.
- `solid_lens.onnx`: Read an ONNX file far enough to list the nodes and load the float weights.
- `solid_lens.recognizer`: PP-OCRv6 medium text recognizer.
- `solid_lens.server`: POST one image or several to `/ocr` and get back each text fragment with its box and confidence.
- `solid_lens.weights`: Lay an exported PP-OCRv6 checkpoint onto MAX module weights."""

__version__ = "0.0.1"
