"""PP-OCRv6 tiny OCR on the MAX engine.

Detection and recognition are PP-OCRv6 tiny, written as MAX modules.
The official weights are loaded from the ONNX checkpoints. Box grouping
and CTC decode stay in Python.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from model import compile_detector, compile_recognizer

ROOT = Path(__file__).resolve().parent
DET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_charset(path: Path) -> list[str]:
    return path.read_text().splitlines()


def read_bmp(path: Path) -> np.ndarray:
    """Read a 24-bit or 32-bit BMP written by sips and return RGB uint8."""
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"{path} is not a BMP")
    pixel_offset = int.from_bytes(data[10:14], "little")
    width = int.from_bytes(data[18:22], "little", signed=True)
    height = int.from_bytes(data[22:26], "little", signed=True)
    bits = int.from_bytes(data[28:30], "little")
    top_down = height < 0
    height = abs(height)
    channels = bits // 8
    if bits not in (24, 32):
        raise ValueError(f"unsupported BMP depth {bits}")
    row_stride = (width * channels + 3) & ~3
    pixels = np.frombuffer(data, dtype=np.uint8, count=row_stride * height, offset=pixel_offset)
    rows = pixels.reshape(height, row_stride)[:, : width * channels].reshape(height, width, channels)
    if not top_down:
        rows = rows[::-1]
    rgb = rows[:, :, :3][:, :, ::-1]
    return np.ascontiguousarray(rgb)


def load_image(path: Path) -> np.ndarray:
    """Return an RGB uint8 image. sips converts JPEG and PNG into BMP."""
    if path.suffix.lower() == ".bmp":
        return read_bmp(path)
    converted = path.with_suffix(".solid_lens.bmp")
    subprocess.run(
        ["sips", "-s", "format", "bmp", str(path), "--out", str(converted)],
        check=True,
        capture_output=True,
    )
    try:
        return read_bmp(converted)
    finally:
        converted.unlink(missing_ok=True)


def resize_hwc(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if (src_h, src_w) == (out_h, out_w):
        return image
    y = (np.arange(out_h, dtype=np.float32) + 0.5) * src_h / out_h - 0.5
    x = (np.arange(out_w, dtype=np.float32) + 0.5) * src_w / out_w - 0.5
    y = np.clip(y, 0, src_h - 1)
    x = np.clip(x, 0, src_w - 1)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (y - y0).astype(np.float32)[:, None, None]
    wx = (x - x0).astype(np.float32)[None, :, None]
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bot = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(image.dtype)


def det_input(image_rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox onto a square canvas. The detector graph is compiled for that size."""
    height, width = image_rgb.shape[:2]
    scale = size / max(height, width)
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    resized = resize_hwc(image_rgb, resized_h, resized_w)
    canvas = np.zeros((size, size, 3), np.float32)
    top = (size - resized_h) // 2
    left = (size - resized_w) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    bgr = canvas[:, :, ::-1]
    normalized = (bgr / 255.0 - DET_MEAN) / DET_STD
    tensor = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1))[None], dtype=np.float32)
    return tensor, scale, left, top


def boxes_from_map(
    prob: np.ndarray,
    thresh: float,
    box_thresh: float,
    unclip_ratio: float,
) -> list[tuple[int, int, int, int, float]]:
    mask = prob > thresh
    height, width = mask.shape
    labels = np.zeros((height, width), np.int32)
    parent = [0]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    next_label = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        neighbors = []
        if y > 0 and labels[y - 1, x]:
            neighbors.append(labels[y - 1, x])
        if x > 0 and labels[y, x - 1]:
            neighbors.append(labels[y, x - 1])
        if not neighbors:
            next_label += 1
            parent.append(next_label)
            labels[y, x] = next_label
            continue
        root = find(neighbors[0])
        labels[y, x] = root
        for neighbor in neighbors[1:]:
            parent[find(neighbor)] = root

    boxes: list[tuple[int, int, int, int, float]] = []
    if next_label == 0:
        return boxes
    roots = np.array([find(label) for label in range(next_label + 1)], np.int32)
    flat = roots[labels]
    for label in range(1, next_label + 1):
        if roots[label] != label:
            continue
        ys_box, xs_box = np.nonzero(flat == label)
        if ys_box.size < 4:
            continue
        y0, y1 = int(ys_box.min()), int(ys_box.max()) + 1
        x0, x1 = int(xs_box.min()), int(xs_box.max()) + 1
        score = float(prob[y0:y1, x0:x1][flat[y0:y1, x0:x1] == label].mean())
        if score < box_thresh or (y1 - y0) < 2 or (x1 - x0) < 2:
            continue
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        half_h = (y1 - y0) * unclip_ratio / 2
        half_w = (x1 - x0) * unclip_ratio / 2
        boxes.append(
            (
                max(0, int(cx - half_w)),
                max(0, int(cy - half_h)),
                min(width, int(np.ceil(cx + half_w))),
                min(height, int(np.ceil(cy + half_h))),
                score,
            )
        )
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def recognize_crop(model, image_rgb: np.ndarray, charset: list[str]) -> str:
    height, width = image_rgb.shape[:2]
    if height < 2 or width < 2:
        return ""
    resized_w = int(np.ceil(48 * width / height))
    resized_w = max(8, min(320, resized_w))
    resized = resize_hwc(image_rgb, 48, resized_w).astype(np.float32)
    bgr = resized[:, :, ::-1]
    normalized = np.transpose((bgr / 255.0 - 0.5) / 0.5, (2, 0, 1))
    tensor = np.zeros((1, 3, 48, 320), np.float32)
    tensor[0, :, :, :resized_w] = normalized
    tensor = np.ascontiguousarray(tensor)
    probs = model.execute(tensor)[0].to_numpy()[0]
    ids = probs.argmax(axis=-1)
    text = []
    previous = -1
    for index in ids.tolist():
        if index != 0 and index != previous:
            text.append(charset[index - 1])
        previous = index
    return "".join(text).strip()


def read_page(
    image_rgb: np.ndarray,
    det_model,
    rec_model,
    charset: list[str],
    size: int,
) -> list[tuple[str, float, int, int, int, int]]:
    tensor, scale, pad_x, pad_y = det_input(image_rgb, size)
    prob = det_model.execute(tensor)[0].to_numpy()[0, 0]
    found = boxes_from_map(prob, thresh=0.2, box_thresh=0.4, unclip_ratio=1.4)
    lines = []
    height, width = image_rgb.shape[:2]
    for x0, y0, x1, y1, score in found:
        left = int(np.clip((x0 - pad_x) / scale, 0, width - 1))
        top = int(np.clip((y0 - pad_y) / scale, 0, height - 1))
        right = int(np.clip((x1 - pad_x) / scale, left + 1, width))
        bottom = int(np.clip((y1 - pad_y) / scale, top + 1, height))
        text = recognize_crop(rec_model, image_rgb[top:bottom, left:right], charset)
        if text:
            lines.append((text, score, left, top, right, bottom))
    return lines


def draw_boxes(image_rgb: np.ndarray, lines: list[tuple[str, float, int, int, int, int]]) -> np.ndarray:
    """Paint a green rectangle around each recognized line."""
    canvas = image_rgb.copy()
    height, width = canvas.shape[:2]
    color = np.array([32, 220, 80], np.uint8)
    for _, _, left, top, right, bottom in lines:
        x0 = int(np.clip(left, 0, width - 1))
        y0 = int(np.clip(top, 0, height - 1))
        x1 = int(np.clip(right, x0 + 1, width))
        y1 = int(np.clip(bottom, y0 + 1, height))
        thickness = 2
        canvas[y0 : y0 + thickness, x0:x1] = color
        canvas[y1 - thickness : y1, x0:x1] = color
        canvas[y0:y1, x0 : x0 + thickness] = color
        canvas[y0:y1, x1 - thickness : x1] = color
    return canvas


def write_bmp(path: Path, image_rgb: np.ndarray) -> None:
    height, width = image_rgb.shape[:2]
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    row_stride = (width * 3 + 3) & ~3
    pixels = np.zeros((height, row_stride), np.uint8)
    pixels[:, : width * 3] = bgr.reshape(height, width * 3)
    pixels = pixels[::-1].tobytes()
    header = bytearray(54)
    header[0:2] = b"BM"
    header[10:14] = (54).to_bytes(4, "little")
    header[14:18] = (40).to_bytes(4, "little")
    header[18:22] = width.to_bytes(4, "little", signed=True)
    header[22:26] = height.to_bytes(4, "little", signed=True)
    header[26:28] = (1).to_bytes(2, "little")
    header[28:30] = (24).to_bytes(2, "little")
    size = 54 + len(pixels)
    header[2:6] = size.to_bytes(4, "little")
    header[34:38] = len(pixels).to_bytes(4, "little")
    path.write_bytes(bytes(header) + pixels)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    bmp = path.with_suffix(".bmp")
    write_bmp(bmp, image_rgb)
    if path.suffix.lower() == ".bmp":
        return
    subprocess.run(
        ["sips", "-s", "format", path.suffix.lower().lstrip("."), str(bmp), "--out", str(path)],
        check=True,
        capture_output=True,
    )
    bmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP-OCRv6 tiny OCR with MAX")
    parser.add_argument("image", type=Path)
    parser.add_argument("--size", type=int, default=640, help="square detector input, a multiple of 32")
    parser.add_argument("--output", type=Path, help="image with boxes drawn; defaults to <image>.boxes.png")
    args = parser.parse_args()
    if args.size % 32 != 0:
        raise SystemExit("--size must be a multiple of 32")
    image = load_image(args.image)
    charset = load_charset(ROOT / "charset.txt")
    det_cache = ROOT / "models" / f"native_det_{args.size}.mef"
    rec_cache = ROOT / "models" / "native_rec_48x320.mef"
    print(
        "loading cached detector" if any(det_cache.glob("*.mef")) else "compiling PP-OCRv6 tiny detector",
        file=sys.stderr,
    )
    detector = compile_detector(args.size, cache_dir=det_cache)
    print(
        "loading cached recognizer" if any(rec_cache.glob("*.mef")) else "compiling PP-OCRv6 tiny recognizer",
        file=sys.stderr,
    )
    recognizer = compile_recognizer(cache_dir=rec_cache)
    lines = read_page(image, detector, recognizer, charset, args.size)
    output = args.output or args.image.with_name(f"{args.image.stem}.boxes.png")
    save_rgb(output, draw_boxes(image, lines))
    print(f"wrote {output}", file=sys.stderr)
    for text, score, *_ in lines:
        print(f"{score:.2f}  {text}")


if __name__ == "__main__":
    main()
