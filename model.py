"""Compile the PP-OCRv6 medium modules and bind their checkpoints."""

from __future__ import annotations

from pathlib import Path

from max.driver import CPU
from max.engine import InferenceSession
from max.graph import Graph, TensorType

from detector import PPOCRV6Detector
from model_config import DEVICE, DTYPE, RecognizerConfig
from recognizer import PPOCRV6Recognizer
from weight_adapters import load_checkpoint

ROOT = Path(__file__).resolve().parent


def _session(cache_dir: Path | None) -> InferenceSession:
    kwargs = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if any(cache_dir.iterdir()):
            kwargs["precompiled_mefs"] = cache_dir
        else:
            kwargs["export_mefs"] = cache_dir
    return InferenceSession(devices=[CPU()], **kwargs)


def _compile(module, shape: list[int], name: str, cache_dir: Path | None):
    graph = Graph(name, input_types=[TensorType(DTYPE, shape, device=DEVICE)])
    with graph:
        graph.output(module(graph.inputs[0].tensor))
    weights = module.state_dict()
    return _session(cache_dir).load(graph, weights_registry=weights)


def compile_detector(size: int, cache_dir: Path | None = None):
    module = PPOCRV6Detector()
    load_checkpoint(module, ROOT / "models" / "medium_det.onnx")
    return _compile(
        module,
        [1, 3, size, size],
        f"ppocrv6_medium_det_{size}",
        cache_dir,
    )


def compile_recognizer(cache_dir: Path | None = None):
    config = RecognizerConfig()
    module = PPOCRV6Recognizer(config)
    load_checkpoint(module, ROOT / "models" / "medium_rec.onnx")
    return _compile(
        module,
        [1, 3, config.image_height, config.image_width],
        "ppocrv6_medium_rec",
        cache_dir,
    )
