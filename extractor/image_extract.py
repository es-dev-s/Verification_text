"""
OCR a standalone image file (PNG/JPG) via Tesseract.
"""

from __future__ import annotations

from typing import Any

from extractor.ocr import (
    MAX_OCR_WIDTH,
    OCR_TIMEOUT_SEC,
    PSM_NO_OSD,
    TESSERACT_OEM,
    _ensure_tesseract,
    _load_bindings,
    _median,
    _safe_confidence,
)


def ocr_image_file(
    path: str,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return OCR spans and average confidence (0–100) for an image file."""
    _ensure_tesseract()
    _load_bindings()

    from extractor import ocr as ocr_mod

    Image = ocr_mod.Image
    pytesseract = ocr_mod.pytesseract
    Output = ocr_mod.Output

    image = Image.open(path)
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    if image.mode != "L":
        image = image.convert("L")
    if image.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(image.width)
        image = image.resize(
            (MAX_OCR_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )

    data = pytesseract.image_to_data(
        image,
        config=f"--psm {PSM_NO_OSD} --oem {TESSERACT_OEM}",
        output_type=Output.DICT,
        timeout=max(1, int(timeout)),
    )

    lines: list[str] = []
    confs: list[float] = []
    n_items = len(data.get("text", []))
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for index in range(n_items):
        text = (data["text"][index] or "").strip()
        conf = _safe_confidence(data["conf"][index])
        if not text or conf < 0:
            continue
        confs.append(conf)
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped.setdefault(key, []).append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": float(data["height"][index] or 0),
                "bold": False,
                "italic": False,
                "bbox": [
                    float(data["left"][index]),
                    float(data["top"][index]),
                    float(data["left"][index] + data["width"][index]),
                    float(data["top"][index] + data["height"][index]),
                ],
                "block": key[0],
                "line": key[2],
                "source": "ocr",
                "confidence": conf,
            }
        )

    spans: list[dict[str, Any]] = []
    for key in sorted(grouped):
        words = sorted(grouped[key], key=lambda item: (item["bbox"][0], item["bbox"][1]))
        text = " ".join(word["text"] for word in words if word["text"]).strip()
        if not text:
            continue
        lines.append(text)
        boxes = [word["bbox"] for word in words]
        word_confs = [float(word.get("confidence") or 0.0) for word in words]
        spans.append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": _median([w["font_size"] for w in words]),
                "bold": False,
                "italic": False,
                "bbox": [
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                "block": key[0],
                "line": key[2],
                "source": "ocr",
                "confidence": round(sum(word_confs) / max(len(word_confs), 1), 1),
            }
        )

    avg = round(sum(confs) / len(confs), 1) if confs else None
    return spans, avg
