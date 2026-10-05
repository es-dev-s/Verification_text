"""
Tesseract OCR for scanned / garbled PDF pages and standalone images.

Controlled pipeline:
1. Render the PDF page at a fixed 400 DPI grayscale (no RGB color noise).
2. Light preprocess (autocontrast + long ruled-line suppression).
3. Dual-pass ``image_to_data`` so dense transcript tables keep word boxes.
4. Map every pixel bbox back into native PDF coordinates.
5. Apply conservative course-code OCR repairs (I/O/A/S digit confusions).

Wide horizontal gaps are preserved later as column separators when lines are
reconstructed in ``pdf_utils``.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import fitz
import numpy as np

Image = None
ImageFilter = None
ImageOps = None
pytesseract = None
Output = None

TITLE_BAND = 1.0
DEFAULT_DPI = 400
OCR_TIMEOUT_SEC = float(os.environ.get("TESSERACT_TIMEOUT_SEC", "180") or 180)
_ENGINE_LOCK = threading.Lock()
_ENGINE_ERROR: str | None = None
_TESSERACT_READY = False

# Known academic course-code prefixes (order matters: longer first).
_CODE_PREFIXES = ("MATH", "GSB", "GSC", "GSA", "QM", "CS", "EE")
_PREFIX_FIXES = (
    ("BE", "EE"),
    ("FE", "EE"),
    ("RE", "EE"),
    ("HE", "EE"),
    ("BF", "EE"),
    ("AE", "EE"),
    ("OM", "QM"),
)
# OCR confusions inside the numeric tail of a course code.
_DIGIT_TRANS = str.maketrans(
    {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "|": "1",
        "Z": "2",
        "A": "4",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8",
    }
)
_CODEISH_RE = re.compile(r"^[A-Za-z]{2,5}[0-9A-Za-z|/]{2,6}$")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class OcrUnavailableError(RuntimeError):
    """Tesseract or its runtime dependencies are missing / failed to init."""


def _load_pillow() -> bool:
    global Image, ImageFilter, ImageOps
    if Image is not None:
        return True
    try:
        from PIL import Image as _Image
        from PIL import ImageFilter as _ImageFilter
        from PIL import ImageOps as _ImageOps
    except ImportError:
        return False
    Image = _Image
    ImageFilter = _ImageFilter
    ImageOps = _ImageOps
    return True


def _load_pytesseract() -> tuple[bool, str | None]:
    global pytesseract, Output
    if pytesseract is not None:
        return True, None
    try:
        import pytesseract as _pt
        from pytesseract import Output as _Output
    except ImportError as exc:
        return False, f"pytesseract missing: {exc}"
    pytesseract = _pt
    Output = _Output
    return True, None


def _configure_tesseract_cmd() -> str | None:
    """Resolve tesseract binary; honor TESSERACT_CMD when set."""
    assert pytesseract is not None
    configured = os.environ.get("TESSERACT_CMD", "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return configured
    which = shutil.which("tesseract")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return which
    # Common Windows install paths.
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return None


def _tesseract_importable() -> tuple[bool, str | None]:
    if not _load_pillow():
        return False, "Pillow missing in this Python"
    ok, reason = _load_pytesseract()
    if not ok:
        return False, reason
    cmd = _configure_tesseract_cmd()
    if not cmd:
        return False, "tesseract binary not found on PATH (set TESSERACT_CMD)"
    try:
        ver = pytesseract.get_tesseract_version()
    except Exception as exc:
        return False, f"tesseract not runnable: {exc}"
    return True, f"tesseract {ver}"


def ocr_is_available() -> bool:
    return ocr_status()["available"]


def ocr_status() -> dict[str, Any]:
    ok, reason = _tesseract_importable()
    if not ok:
        return {"available": False, "engine": "tesseract", "reason": reason}
    if _ENGINE_ERROR:
        return {
            "available": False,
            "engine": "tesseract",
            "reason": _ENGINE_ERROR,
        }
    return {"available": True, "engine": "tesseract", "reason": reason or "ready"}


def _ensure_tesseract() -> None:
    global _TESSERACT_READY, _ENGINE_ERROR
    if _TESSERACT_READY:
        return
    with _ENGINE_LOCK:
        if _TESSERACT_READY:
            return
        ok, reason = _tesseract_importable()
        if not ok:
            _ENGINE_ERROR = reason
            raise OcrUnavailableError(
                "OCR extras are not installed. "
                f"Install Tesseract + pip install pytesseract Pillow ({reason})"
            )
        _ENGINE_ERROR = None
        _TESSERACT_READY = True


def ocr_page(
    page: fitz.Page,
    page_number: int,
    dpi: int | None = None,
    timeout: float = OCR_TIMEOUT_SEC,
    band: float = TITLE_BAND,
) -> list[dict[str, Any]]:
    """OCR one page and return spans with PDF-page bounding boxes."""
    _ensure_tesseract()
    _load_pillow()

    dpi = int(dpi or _env_int("TESSERACT_DPI", DEFAULT_DPI))
    rect = page.rect
    clip = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y0 + rect.height * min(max(band, 0.35), 1.0),
    )
    image = _render_page_gray(page, dpi=dpi, clip=clip)
    print(
        f"[ocr] page={page_number} starting Tesseract "
        f"(image={image.width}x{image.height}, dpi={dpi}, timeout={timeout:.0f}s)",
        flush=True,
    )

    detections = _predict_page_image(image, timeout=timeout)
    scale = 72.0 / float(dpi)
    spans: list[dict[str, Any]] = []
    for index, item in enumerate(detections):
        text = _maybe_correct_token(item["text"])
        conf = item["confidence"]
        x0, y0, x1, y1 = item["bbox"]
        # Shrink tall Tesseract boxes toward the baseline so adjacent
        # transcript rows do not falsely overlap during line clustering.
        bx0, by0, bx1, by1 = _shrink_bbox_for_rows(
            [
                clip.x0 + x0 * scale,
                clip.y0 + y0 * scale,
                clip.x0 + x1 * scale,
                clip.y0 + y1 * scale,
            ]
        )
        bbox = [round(bx0, 2), round(by0, 2), round(bx1, 2), round(by1, 2)]
        font_size = round(max(bbox[3] - bbox[1], 0.0), 2)
        spans.append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": font_size,
                "bold": False,
                "italic": False,
                "bbox": bbox,
                "block": item.get("block", 0),
                "line": item.get("line", index),
                "source": "ocr",
                "confidence": conf,
            }
        )
    spans = _reassign_ocr_lines(spans)
    spans.sort(key=lambda span: (span["bbox"][1], span["bbox"][0]))
    _log_ocr_result(page_number, spans)
    return spans


def ocr_pil_image(
    image: Any,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """OCR a PIL image; return spans and average confidence (0–100)."""
    _ensure_tesseract()
    _load_pillow()

    if image.mode != "L":
        image = image.convert("L")

    print(
        f"[ocr] image page={page_number} starting Tesseract "
        f"(image={image.width}x{image.height}, timeout={timeout:.0f}s)",
        flush=True,
    )
    detections = _predict_page_image(image, timeout=timeout)
    spans: list[dict[str, Any]] = []
    confs: list[float] = []
    for index, item in enumerate(detections):
        conf = item["confidence"]
        confs.append(conf)
        x0, y0, x1, y1 = _shrink_bbox_for_rows(item["bbox"])
        spans.append(
            {
                "text": _maybe_correct_token(item["text"]),
                "page": page_number,
                "font_name": "ocr",
                "font_size": round(max(y1 - y0, 0.0), 2),
                "bold": False,
                "italic": False,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "block": item.get("block", 0),
                "line": item.get("line", index),
                "source": "ocr",
                "confidence": conf,
            }
        )
    spans = _reassign_ocr_lines(spans)
    spans.sort(key=lambda span: (span["bbox"][1], span["bbox"][0]))
    avg = round(sum(confs) / len(confs), 1) if confs else None
    _log_ocr_result(page_number, spans)
    return spans, avg


def _shrink_bbox_for_rows(bbox: list[float]) -> list[float]:
    """Collapse inflated OCR boxes onto a thin mid-band for row separation."""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    height = max(y1 - y0, 1.0)
    mid = (y0 + y1) / 2.0
    half = max(height * 0.18, 0.6)
    return [x0, mid - half, x1, mid + half]


def _reassign_ocr_lines(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rebuild line ids from center-Y gaps after dual-pass merge.

    Tesseract line_num values are not comparable across OCR passes, so we
    re-cluster by vertical gaps using the shrunk boxes.
    """
    if not spans:
        return spans
    ordered = sorted(
        spans,
        key=lambda span: (
            (span["bbox"][1] + span["bbox"][3]) / 2.0,
            span["bbox"][0],
        ),
    )
    heights = [max(span["bbox"][3] - span["bbox"][1], 1.0) for span in ordered]
    median_h = sorted(heights)[len(heights) // 2]
    gap_limit = max(2.2, 0.85 * median_h)

    line_id = 0
    prev_mid = None
    for span in ordered:
        mid = (span["bbox"][1] + span["bbox"][3]) / 2.0
        if prev_mid is not None and abs(mid - prev_mid) > gap_limit:
            line_id += 1
        span["line"] = line_id
        span["block"] = 0
        prev_mid = mid if prev_mid is None else (0.7 * prev_mid + 0.3 * mid)
    return ordered


def _log_ocr_result(page_number: int, spans: list[dict[str, Any]]) -> None:
    confs = [
        float(span["confidence"])
        for span in spans
        if span.get("confidence") is not None
    ]
    avg = round(sum(confs) / len(confs), 1) if confs else None
    preview_lines = [str(span.get("text") or "") for span in spans[:20]]
    preview = "\n".join(preview_lines)
    more = "" if len(spans) <= 20 else f"\n... ({len(spans) - 20} more spans)"
    print(
        f"[ocr] page={page_number} finished spans={len(spans)} "
        f"avg_confidence={avg}\n----- OCR TEXT PREVIEW (page {page_number}) -----\n"
        f"{preview}{more}\n----- END OCR PREVIEW -----",
        flush=True,
    )


def _render_page_gray(
    page: fitz.Page,
    *,
    dpi: int,
    clip: fitz.Rect,
) -> Any:
    """Controlled 400-DPI grayscale render; geometry maps 1:1 via dpi/72."""
    scale = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csGRAY,
        clip=clip,
    )
    return Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)


def _preprocess(image: Any) -> Any:
    """Light cleanup: lift contrast, soften watermark, drop long ruled lines."""
    if not _env_flag("TESSERACT_PREPROCESS", True):
        return image
    cleaned = ImageOps.autocontrast(image, cutoff=_env_float("TESSERACT_AUTOCONTRAST", 0.8))
    cleaned = cleaned.filter(ImageFilter.MedianFilter(size=3))
    if _env_flag("TESSERACT_REMOVE_GRID", True):
        cleaned = _remove_ruled_lines(cleaned)
    return cleaned


def _remove_ruled_lines(image: Any) -> Any:
    """Replace long horizontal/vertical table rules with page background."""
    arr = np.asarray(image, dtype=np.uint8)
    ink = arr < 160
    height, width = ink.shape
    if height < 32 or width < 32:
        return image
    row_dense = ink.sum(axis=1) > 0.55 * width
    col_dense = ink.sum(axis=0) > 0.45 * height
    if not row_dense.any() and not col_dense.any():
        return image
    out = arr.copy()
    bg = int(np.percentile(arr, 90))
    out[row_dense, :] = bg
    out[:, col_dense] = bg
    return Image.fromarray(out, mode="L")


def _predict_page_image(image: Any, *, timeout: float) -> list[dict[str, Any]]:
    def _run() -> list[dict[str, Any]]:
        primary = _tesseract_words(
            _preprocess(image),
            config=_primary_config(image),
        )
        secondary = _tesseract_words(
            image,
            config=_secondary_config(image),
        )
        # Course-number column specialist (helps cracked codes like EEAS33).
        column = _course_column_words(image)
        merged = _merge_detections(primary, secondary)
        merged = _merge_detections(merged, column)
        merged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return merged

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=max(1.0, float(timeout)))
        except TimeoutError as exc:
            raise RuntimeError(f"Tesseract timed out after {timeout:.0f}s") from exc


def _primary_config(image: Any) -> str:
    dpi = _estimate_dpi(image)
    # PSM 6 + preprocess: strong on dense multi-column transcript tables.
    return (
        f"--oem 3 --psm 6 --dpi {dpi} "
        "-c preserve_interword_spaces=1"
    )


def _secondary_config(image: Any) -> str:
    dpi = _estimate_dpi(image)
    # PSM 4 + Sauvola: better per-row separation and GPA column recall.
    return (
        f"--oem 3 --psm 4 --dpi {dpi} "
        "-c thresholding_method=2 "
        "-c preserve_interword_spaces=1"
    )


def _estimate_dpi(image: Any) -> int:
    # Prefer explicit env; otherwise assume the controlled 400 DPI render.
    return _env_int("TESSERACT_DPI", DEFAULT_DPI)


def _tesseract_words(image: Any, *, config: str) -> list[dict[str, Any]]:
    assert pytesseract is not None and Output is not None
    lang = os.environ.get("TESSERACT_LANG", "eng")
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=Output.DICT,
    )
    detections: list[dict[str, Any]] = []
    count = len(data["text"])
    for index in range(count):
        text = (data["text"][index] or "").strip()
        if not text:
            continue
        conf = float(data["conf"][index])
        if conf < 0:
            continue
        left = float(data["left"][index])
        top = float(data["top"][index])
        width = float(data["width"][index])
        height = float(data["height"][index])
        if width <= 1 or height <= 1:
            continue
        detections.append(
            {
                "text": text,
                "confidence": round(conf, 1),
                "bbox": [left, top, left + width, top + height],
                "block": int(data["block_num"][index]),
                "line": int(data["line_num"][index]),
            }
        )
    return detections


def _course_column_words(image: Any) -> list[dict[str, Any]]:
    """OCR the approximate Course No. column at higher local contrast."""
    width, height = image.size
    left = int(0.14 * width)
    right = int(0.34 * width)
    top = int(0.20 * height)
    bottom = int(0.90 * height)
    if right - left < 40 or bottom - top < 40:
        return []
    crop = image.crop((left, top, right, bottom))
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = ImageOps.expand(crop, border=16, fill=255)
    dpi = _estimate_dpi(image)
    config = (
        f"--oem 3 --psm 6 --dpi {dpi} "
        "-c preserve_interword_spaces=1"
    )
    words = _tesseract_words(crop, config=config)
    shifted: list[dict[str, Any]] = []
    for item in words:
        x0, y0, x1, y1 = item["bbox"]
        shifted.append(
            {
                "text": item["text"],
                "confidence": item["confidence"],
                "bbox": [
                    x0 - 16 + left,
                    y0 - 16 + top,
                    x1 - 16 + left,
                    y1 - 16 + top,
                ],
                "block": item.get("block", 0),
                "line": item.get("line", 0),
            }
        )
    return shifted


def _merge_detections(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep primary layout; add secondary words that do not overlap heavily."""
    merged = [dict(item) for item in primary]
    for item in secondary:
        if _overlaps_any(item["bbox"], merged, iou_thresh=0.35):
            # Prefer higher-confidence text on near-duplicate boxes.
            _maybe_upgrade_text(merged, item)
            continue
        merged.append(dict(item))
    return merged


def _overlaps_any(
    bbox: list[float],
    items: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> bool:
    for item in items:
        if _iou(bbox, item["bbox"]) >= iou_thresh:
            return True
    return False


def _maybe_upgrade_text(
    items: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    for item in items:
        if _iou(item["bbox"], candidate["bbox"]) < 0.35:
            continue
        cand_code = _normalize_course_code(candidate["text"])
        item_code = _normalize_course_code(item["text"])
        if cand_code and not item_code:
            item["text"] = candidate["text"]
            item["confidence"] = max(
                float(item.get("confidence") or 0.0),
                float(candidate.get("confidence") or 0.0),
            )
            return
        if float(candidate.get("confidence") or 0.0) > float(
            item.get("confidence") or 0.0
        ) + 8:
            item["text"] = candidate["text"]
            item["confidence"] = candidate["confidence"]
        return


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _maybe_correct_token(text: str) -> str:
    """Replace a whole token when it is clearly a mangled course code."""
    cleaned = text.strip()
    if not cleaned or not _CODEISH_RE.match(cleaned):
        return cleaned
    normalized = _normalize_course_code(cleaned)
    if not normalized:
        return cleaned
    # Keep original when already well-formed.
    if cleaned.upper() == normalized:
        return normalized
    return normalized


def _normalize_course_code(token: str) -> str | None:
    raw = re.sub(r"[^A-Za-z0-9]", "", token or "").upper()
    if len(raw) < 5:
        return None

    prefix = None
    rest = None
    for candidate in _CODE_PREFIXES:
        if raw.startswith(candidate):
            prefix, rest = candidate, raw[len(candidate) :]
            break
    if prefix is None:
        for bad, good in _PREFIX_FIXES:
            if raw.startswith(bad):
                prefix, rest = good, raw[len(bad) :]
                break
    if prefix is None or rest is None:
        return None

    digits = rest.translate(_DIGIT_TRANS)
    match = re.search(r"(\d{3,4})", digits)
    if not match:
        return None
    return f"{prefix}{match.group(1)}"
