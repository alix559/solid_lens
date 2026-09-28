"""PP-OCRv6 medium OCR on the MAX engine.

Detection and recognition are written as MAX modules. The official weights
are loaded from the ONNX checkpoints. Box grouping and CTC decode stay in
Python. Recognition uses the LightSVTR transformer neck.
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


def load_charset(path: Path, *, add_space: bool = False) -> list[str]:
    chars = path.read_text().splitlines()
    if add_space:
        chars.append(" ")
    return chars


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


def _ctc_decode(ids: list[int], charset: list[str]) -> str:
    text = []
    previous = -1
    for index in ids:
        if index != 0 and index != previous:
            text.append(charset[index - 1])
        previous = index
    return "".join(text).strip()


def _recognize_ids(model, image_rgb: np.ndarray) -> list[int]:
    """Run one crop that already fits the 320-wide recognizer."""
    height, width = image_rgb.shape[:2]
    resized_w = int(np.ceil(48 * width / height))
    resized_w = max(8, min(320, resized_w))
    resized = resize_hwc(image_rgb, 48, resized_w).astype(np.float32)
    bgr = resized[:, :, ::-1]
    normalized = np.transpose((bgr / 255.0 - 0.5) / 0.5, (2, 0, 1))
    tensor = np.zeros((1, 3, 48, 320), np.float32)
    tensor[0, :, :, :resized_w] = normalized
    tensor = np.ascontiguousarray(tensor)
    probs = model.execute(tensor)[0].to_numpy()[0]
    # The recognizer downsamples width by 8, so padding past resized_w is blank.
    valid = max(1, resized_w // 8)
    return [int(index) for index in probs.argmax(axis=-1)[:valid]]


def recognize_crop(model, image_rgb: np.ndarray, charset: list[str]) -> str:
    height, width = image_rgb.shape[:2]
    if height < 2 or width < 2:
        return ""
    fitted = int(np.ceil(48 * width / height))
    if fitted <= 320:
        return _ctc_decode(_recognize_ids(model, image_rgb), charset)
    # A long line squashed into 320 columns loses characters. Slide a window
    # whose kept timesteps abut, and drop the edges where the crop has no context.
    drop = 5
    stride_net = (40 - 2 * drop) * 8
    window = max(8, int(round(320 * height / 48)))
    stride = max(4, int(round(stride_net * height / 48)))
    merged: list[int] = []
    x = 0
    first = True
    while x < width:
        piece = image_rgb[:, x : min(width, x + window)]
        seq = _recognize_ids(model, piece)
        last = x + window >= width
        if first and last:
            use = seq
        elif first:
            use = seq[:-drop] if len(seq) > drop else seq
        elif last:
            use = seq[drop:] if len(seq) > drop else seq
        elif len(seq) > 2 * drop:
            use = seq[drop:-drop]
        else:
            use = seq
        merged.extend(use)
        if last:
            break
        x += stride
        first = False
    return _ctc_decode(merged, charset)


def _map_boxes(
    prob: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> list[tuple[int, int, int, int, float]]:
    found = boxes_from_map(prob, thresh=0.2, box_thresh=0.45, unclip_ratio=1.4)
    mapped = []
    for x0, y0, x1, y1, score in found:
        left = int(np.clip((x0 - pad_x) / scale, 0, width - 1))
        top = int(np.clip((y0 - pad_y) / scale, 0, height - 1))
        right = int(np.clip((x1 - pad_x) / scale, left + 1, width))
        bottom = int(np.clip((y1 - pad_y) / scale, top + 1, height))
        mapped.append((left, top, right, bottom, score))
    return mapped


def _tile_starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _box_area(box: tuple[int, int, int, int, float]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(a: tuple[int, int, int, int, float], b: tuple[int, int, int, int, float]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def _same_box(a: tuple[int, int, int, int, float], b: tuple[int, int, int, int, float]) -> bool:
    inter = _intersection(a, b)
    if inter == 0:
        return False
    area_a = _box_area(a)
    area_b = _box_area(b)
    union = area_a + area_b - inter
    if union and inter / union > 0.3:
        return True
    smaller = min(area_a, area_b)
    return bool(smaller) and inter / smaller > 0.6


def _nms(boxes: list[tuple[int, int, int, int, float]]) -> list[tuple[int, int, int, int, float]]:
    ordered = sorted(boxes, key=lambda box: box[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    for box in ordered:
        if any(_same_box(box, other) for other in kept):
            continue
        kept.append(box)
    return kept


def detect_boxes(
    image_rgb: np.ndarray,
    det_model,
    size: int,
) -> list[tuple[int, int, int, int, float]]:
    """Boxes in original pixels.

    The compiled detector is a square. Letterboxing a whole page into it
    shrinks table cells until digits and short words disappear, so a large
    page is also scanned in overlapping tiles at twice the network size.
    """
    height, width = image_rgb.shape[:2]
    tensor, scale, pad_x, pad_y = det_input(image_rgb, size)
    prob = det_model.execute(tensor)[0].to_numpy()[0, 0]
    boxes = _map_boxes(prob, scale, pad_x, pad_y, width, height)

    tile = size * 2
    if max(height, width) <= tile:
        boxes.sort(key=lambda box: (box[1], box[0]))
        return boxes

    overlap = size // 2 + 40
    step = tile - overlap
    margin = 8
    extra: list[tuple[int, int, int, int, float]] = []
    ys = _tile_starts(height, tile, step)
    xs = _tile_starts(width, tile, step)
    print(f"scanning {len(ys) * len(xs)} tiles", file=sys.stderr)
    for top in ys:
        for left in xs:
            crop = image_rgb[top : top + tile, left : left + tile]
            crop_h, crop_w = crop.shape[:2]
            tensor, scale, pad_x, pad_y = det_input(crop, size)
            prob = det_model.execute(tensor)[0].to_numpy()[0, 0]
            at_left = left == 0
            at_top = top == 0
            at_right = left + crop_w >= width
            at_bottom = top + crop_h >= height
            for box in _map_boxes(prob, scale, pad_x, pad_y, crop_w, crop_h):
                x0, y0, x1, y1, score = box
                if (not at_left and x0 <= margin) or (not at_right and x1 >= crop_w - margin):
                    continue
                if (not at_top and y0 <= margin) or (not at_bottom and y1 >= crop_h - margin):
                    continue
                extra.append((x0 + left, y0 + top, x1 + left, y1 + top, score))
    extra = _nms(extra)
    boxes.extend(box for box in extra if all(not _same_box(box, kept) for kept in list(boxes)))
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def read_page(
    image_rgb: np.ndarray,
    det_model,
    rec_model,
    charset: list[str],
    size: int,
) -> list[tuple[str, float, int, int, int, int]]:
    lines = []
    for left, top, right, bottom, score in detect_boxes(image_rgb, det_model, size):
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
    parser = argparse.ArgumentParser(description="Run PP-OCRv6 medium OCR with MAX")
    parser.add_argument("image", type=Path)
    parser.add_argument("--size", type=int, default=640, help="square detector input, a multiple of 32")
    parser.add_argument("--output", type=Path, help="image with boxes drawn; defaults to <image>.boxes.png")
    args = parser.parse_args()
    if args.size % 32 != 0:
        raise SystemExit("--size must be a multiple of 32")
    image = load_image(args.image)
    charset = load_charset(ROOT / "models" / "ppocrv6_dict.txt", add_space=True)
    det_cache = ROOT / "models" / f"native_medium_det_{args.size}.mef"
    rec_cache = ROOT / "models" / "native_medium_rec_48x320.mef"
    print(
        "loading cached detector" if any(det_cache.glob("*.mef")) else "compiling PP-OCRv6 medium detector",
        file=sys.stderr,
    )
    detector = compile_detector(args.size, cache_dir=det_cache)
    print(
        "loading cached recognizer" if any(rec_cache.glob("*.mef")) else "compiling PP-OCRv6 medium recognizer",
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
