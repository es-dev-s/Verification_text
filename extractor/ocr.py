"""
PaddleOCR fallback for scanned / garbled PDF pages and image uploads.

Renders pages at high DPI, runs PP-OCR, and groups detections into readable
lines. Table-like layouts use column separators so downstream LLM extraction
gets one field group per line instead of a single mashed paragraph.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import fitz
import numpy as np
from PIL import Image

# Avoid slow model-host checks on cold start (PaddleX 3.x).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_DPI = 200
TITLE_BAND = 1.0
MAX_OCR_WIDTH = 1600
OCR_TIMEOUT_SEC = 120
_COLUMN_GAP_WIDTH_RATIO = 1.15
_COLUMN_GAP_MIN_PX = 18.0
# Dense transcript tables need tight vertical banding so rows don't collapse.
_LINE_Y_OVERLAP_RATIO = 0.55
_LINE_Y_MID_RATIO = 0.35
# Mobile PP-OCR models — much faster on CPU than medium/server.
_DET_MODEL = os.environ.get("PADDLEOCR_DET_MODEL", "PP-OCRv5_mobile_det")
_REC_MODEL = os.environ.get("PADDLEOCR_REC_MODEL", "en_PP-OCRv5_mobile_rec")

_PaddleOCR: Any = None
_OCR_ENGINE: Any = None
_OCR_LOCK = threading.Lock()


class OcrUnavailableError(RuntimeError):
    """PaddleOCR or its dependencies are missing or failed to load."""


def _load_bindings() -> bool:
    global _PaddleOCR
    if _PaddleOCR is not None:
        return True
    try:
        from paddleocr import PaddleOCR as _POCR
    except ImportError:
        return False
    _PaddleOCR = _POCR
    return True


def _paddle_lang() -> str:
    return (os.environ.get("PADDLEOCR_LANG") or "en").strip() or "en"


def _get_engine() -> Any:
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    if not _load_bindings():
        raise OcrUnavailableError(
            "OCR extras are not installed. Run: pip install paddleocr paddlepaddle"
        )
    # PaddlePaddle 3.3.x + oneDNN can crash on CPU; disabling mkldnn is the supported workaround.
    # Mobile det/rec + capped side length keeps CPU OCR usable for multi-page transcripts.
    _OCR_ENGINE = _PaddleOCR(
        lang=_paddle_lang(),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        text_detection_model_name=_DET_MODEL,
        text_recognition_model_name=_REC_MODEL,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
    )
    return _OCR_ENGINE


def ocr_is_available() -> bool:
    return ocr_status()["available"]


def ocr_status() -> dict[str, Any]:
    if not _load_bindings():
        return {
            "available": False,
            "engine": "paddleocr",
            "reason": "paddleocr not installed in this Python",
        }
    try:
        with _OCR_LOCK:
            _get_engine()
    except OcrUnavailableError as exc:
        return {"available": False, "engine": "paddleocr", "reason": str(exc)}
    except Exception as exc:
        return {
            "available": False,
            "engine": "paddleocr",
            "reason": f"PaddleOCR init failed: {exc}",
        }
    return {"available": True, "engine": "paddleocr", "reason": "ready"}


def ocr_page(
    page: fitz.Page,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    timeout: float = OCR_TIMEOUT_SEC,
    band: float = TITLE_BAND,
) -> list[dict[str, Any]]:
    """OCR one PDF page and return native-shaped spans (one span per visual line)."""
    del timeout  # reserved for API compatibility; Paddle has no per-call timeout hook

    rect = page.rect
    clip = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y0 + rect.height * min(max(band, 0.35), 1.0),
    )
    image_rgb, image_w, image_h = _render_page_rgb(page, clip, dpi)
    x_scale = clip.width / max(image_w, 1)
    y_scale = clip.height / max(image_h, 1)
    return _ocr_rgb_array(
        image_rgb,
        page_number=page_number,
        clip_origin=(clip.x0, clip.y0),
        x_scale=x_scale,
        y_scale=y_scale,
    )


def ocr_image_file_spans(
    path: str,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return OCR spans and average confidence (0–100) for an image file."""
    del timeout
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    if image.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(image.width)
        image = image.resize(
            (MAX_OCR_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )

    spans = _ocr_rgb_array(np.array(image), page_number=page_number)
    confidences = [
        float(span["confidence"])
        for span in spans
        if span.get("confidence") is not None
    ]
    avg = round(sum(confidences) / len(confidences), 1) if confidences else None
    return spans, avg


def _render_page_rgb(
    page: fitz.Page,
    clip: fitz.Rect,
    dpi: int,
) -> tuple[np.ndarray, int, int]:
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csRGB,
        clip=clip,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    if image.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(image.width)
        image = image.resize(
            (MAX_OCR_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )
    array = np.array(image)
    return array, image.width, image.height


def _ocr_rgb_array(
    image_rgb: np.ndarray,
    *,
    page_number: int,
    clip_origin: tuple[float, float] = (0.0, 0.0),
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> list[dict[str, Any]]:
    _ensure_paddle()
    with _OCR_LOCK:
        raw = list(_get_engine().predict(image_rgb))
    detections = _parse_paddle_results(raw)
    lines = _group_detections_into_lines(detections)
    return _lines_to_spans(
        lines,
        page_number=page_number,
        clip_origin=clip_origin,
        x_scale=x_scale,
        y_scale=y_scale,
    )


def _ensure_paddle() -> None:
    if not _load_bindings():
        raise OcrUnavailableError(
            "OCR extras are not installed. Run: pip install paddleocr paddlepaddle"
        )
    try:
        _get_engine()
    except Exception as exc:
        raise OcrUnavailableError(f"PaddleOCR is unavailable: {exc}") from exc


def _parse_paddle_results(results: list[Any]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for page_result in results:
        payload = _result_payload(page_result)
        texts = _as_list(payload.get("rec_texts"))
        scores = _as_list(payload.get("rec_scores"))
        boxes = _as_list(payload.get("rec_boxes"))
        if not boxes:
            boxes = _as_list(payload.get("dt_polys"))
        for index, text in enumerate(texts):
            cleaned = (str(text) if text is not None else "").strip()
            if not cleaned:
                continue
            box = boxes[index] if index < len(boxes) else None
            bbox_px = _normalize_box(box)
            if bbox_px is None:
                continue
            score_raw = scores[index] if index < len(scores) else 0.0
            detections.append(
                {
                    "text": cleaned,
                    "bbox_px": bbox_px,
                    "confidence": _score_to_percent(score_raw),
                }
            )
    return detections


def _as_list(value: Any) -> list[Any]:
    """Convert Paddle outputs (list / ndarray / None) without boolean tests on arrays."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _result_payload(page_result: Any) -> dict[str, Any]:
    if isinstance(page_result, dict):
        inner = page_result.get("res")
        return inner if isinstance(inner, dict) else page_result
    json_attr = getattr(page_result, "json", None)
    if isinstance(json_attr, dict):
        inner = json_attr.get("res")
        return inner if isinstance(inner, dict) else json_attr
    if hasattr(page_result, "get"):
        try:
            inner = page_result.get("res")
            if isinstance(inner, dict):
                return inner
        except Exception:
            pass
        try:
            return dict(page_result)
        except Exception:
            return {}
    return {}


def _normalize_box(box: Any) -> list[float] | None:
    if box is None:
        return None
    if isinstance(box, np.ndarray):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in box
    ):
        x0, y0, x1, y1 = (float(v) for v in box)
        return [x0, y0, x1, y1]
    if isinstance(box, (list, tuple)) and len(box) > 0:
        first = box[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, (list, tuple)):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _score_to_percent(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1.0:
        score *= 100.0
    return round(max(0.0, min(score, 100.0)), 1)


def _group_detections_into_lines(
    detections: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not detections:
        return []
    ordered = sorted(
        detections,
        key=lambda item: (
            (item["bbox_px"][1] + item["bbox_px"][3]) / 2.0,
            item["bbox_px"][0],
        ),
    )
    lines: list[list[dict[str, Any]]] = []
    for detection in ordered:
        placed = False
        for line in lines:
            if _same_text_line(detection, line):
                line.append(detection)
                placed = True
                break
        if not placed:
            lines.append([detection])
    for line in lines:
        line.sort(key=lambda item: item["bbox_px"][0])
    lines.sort(
        key=lambda cluster: (
            min(item["bbox_px"][1] for item in cluster),
            min(item["bbox_px"][0] for item in cluster),
        )
    )
    return lines


def _same_text_line(detection: dict[str, Any], line: list[dict[str, Any]]) -> bool:
    dx0, dy0, dx1, dy1 = detection["bbox_px"]
    d_mid = (dy0 + dy1) / 2.0
    d_h = max(dy1 - dy0, 1.0)
    ly0 = min(item["bbox_px"][1] for item in line)
    ly1 = max(item["bbox_px"][3] for item in line)
    l_mid = (ly0 + ly1) / 2.0
    l_h = max(ly1 - ly0, max(item["bbox_px"][3] - item["bbox_px"][1] for item in line), 1.0)
    overlap = min(dy1, ly1) - max(dy0, ly0)
    # Require clear vertical overlap; reject stacked table rows.
    if overlap < _LINE_Y_OVERLAP_RATIO * min(d_h, l_h):
        return False
    if abs(d_mid - l_mid) > _LINE_Y_MID_RATIO * max(d_h, l_h):
        return False
    return True


def _format_line_text(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""
    if len(cells) == 1:
        return cells[0]["text"]
    widths = [max(cell["bbox_px"][2] - cell["bbox_px"][0], 1.0) for cell in cells]
    median_w = sorted(widths)[len(widths) // 2]
    gap_threshold = max(_COLUMN_GAP_MIN_PX, median_w * _COLUMN_GAP_WIDTH_RATIO)
    parts: list[str] = []
    previous_x1: float | None = None
    for cell in cells:
        text = cell["text"]
        x0, _, x1, _ = cell["bbox_px"]
        if previous_x1 is not None and parts:
            gap = x0 - previous_x1
            if gap >= gap_threshold:
                parts.append(" | ")
            elif not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(text)
        previous_x1 = x1
    return "".join(parts).strip()


def _lines_to_spans(
    lines: list[list[dict[str, Any]]],
    *,
    page_number: int,
    clip_origin: tuple[float, float],
    x_scale: float,
    y_scale: float,
) -> list[dict[str, Any]]:
    ox, oy = clip_origin
    spans: list[dict[str, Any]] = []
    for line_index, cells in enumerate(lines):
        text = _format_line_text(cells)
        if not text:
            continue
        boxes = [cell["bbox_px"] for cell in cells]
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        confidences = [float(cell.get("confidence") or 0.0) for cell in cells]
        font_size = round(max(y1 - y0, 1.0) * y_scale, 2)
        spans.append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": font_size,
                "bold": False,
                "italic": False,
                "bbox": [
                    round(ox + x0 * x_scale, 2),
                    round(oy + y0 * y_scale, 2),
                    round(ox + x1 * x_scale, 2),
                    round(oy + y1 * y_scale, 2),
                ],
                "block": line_index // 8,
                "line": line_index,
                "source": "ocr",
                "confidence": round(sum(confidences) / max(len(confidences), 1), 1),
            }
        )
    return spans


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 2)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 2)
