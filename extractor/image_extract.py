"""
OCR a standalone image file (PNG/JPG) via Tesseract.
"""

from __future__ import annotations

from typing import Any

from extractor.ocr import OCR_TIMEOUT_SEC, _load_pillow, ocr_pil_image


def ocr_image_file(
    path: str,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return OCR spans and average confidence (0–100) for an image file."""
    if not _load_pillow():
        raise RuntimeError("Pillow is required for image OCR")

    from extractor import ocr as ocr_mod

    Image = ocr_mod.Image
    image = Image.open(path)
    return ocr_pil_image(image, page_number=page_number, timeout=timeout)
