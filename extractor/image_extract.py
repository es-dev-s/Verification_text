"""
OCR a standalone image file (PNG/JPG) via PaddleOCR.
"""

from __future__ import annotations

from typing import Any

from extractor.ocr import OCR_TIMEOUT_SEC, ocr_image_file_spans


def ocr_image_file(
    path: str,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return OCR spans and average confidence (0–100) for an image file."""
    return ocr_image_file_spans(path, page_number=page_number, timeout=timeout)
