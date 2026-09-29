"""
Full-document PDF text extraction for CV parsing.

Reuses native PyMuPDF extraction and Tesseract OCR from this package.
Unlike title extraction, every uploaded page is processed — no front-matter
cap and no OCR page budget.
"""

from __future__ import annotations

from typing import Any

from extractor.ocr import OcrUnavailableError, ocr_page, ocr_status
from extractor.pdf_utils import (
    classify_page,
    extract_page_spans,
    extract_usable_native_spans,
    open_pdf,
    reconstruct_text,
    spans_to_lines,
)

OCR_TIMEOUT_SEC = 45.0


def extract_full_document(
    pdf_path: str,
    *,
    use_ocr: bool = True,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Extract text from every page.

    Returns normalized ``raw_text`` plus per-page ``source`` (native|ocr|empty|…)
    and ``ocr_confidence`` (0–100 average when OCR was used, else null).
    """
    doc = open_pdf(pdf_path)
    try:
        spans: list[dict[str, Any]] = []
        pages: list[dict[str, Any]] = []
        warnings: list[str] = []
        page_geom: dict[int, tuple[float, float]] = {}
        document_page_count = doc.page_count

        for page_index in range(document_page_count):
            page = doc[page_index]
            page_number = page_index + 1
            page_geom[page_number] = (float(page.rect.width), float(page.rect.height))
            page_spans, report = _extract_one_page(page, page_number, use_ocr, warnings)
            spans.extend(page_spans)
            pages.append(report)

        lines = spans_to_lines(spans)
        for line in lines:
            width, height = page_geom.get(line["page"], (612.0, 792.0))
            line["page_width"] = round(width, 2)
            line["page_height"] = round(height, 2)

        raw_text = reconstruct_text(lines)
        ocr_info = ocr_status()
        return {
            "ok": bool(raw_text.strip()),
            "filename": filename,
            "raw_text": raw_text,
            "page_count": document_page_count,
            "pages": pages,
            "pdf_type": _document_type(pages),
            "span_count": len(spans),
            "line_count": len(lines),
            "lines": lines,
            "ocr_available": ocr_info["available"],
            "ocr_reason": ocr_info["reason"],
            "warnings": warnings,
        }
    finally:
        doc.close()


def _extract_one_page(
    page: Any,
    page_number: int,
    use_ocr: bool,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kind = classify_page(page)
    method = "empty"
    error: str | None = None
    page_spans: list[dict[str, Any]] = []

    if kind == "native":
        page_spans = extract_page_spans(page, page_number)
        method = "native"
    elif kind in {"scanned", "garbled"}:
        native_spans = extract_usable_native_spans(page, page_number)
        if native_spans:
            page_spans = native_spans
            method = "native"
        elif not use_ocr:
            method = "ocr_skipped"
            error = "OCR skipped"
        else:
            page_spans, method, error = _run_ocr(page, page_number, warnings)
    else:
        method = "empty"

    source = _page_source(method)
    confidences = [
        float(span["confidence"])
        for span in page_spans
        if span.get("source") == "ocr" and span.get("confidence") is not None
    ]
    ocr_confidence = (
        round(sum(confidences) / len(confidences), 1) if confidences else None
    )

    report = {
        "page": page_number,
        "kind": kind,
        "source": source,
        "method": method,
        "ocr_confidence": ocr_confidence,
        "span_count": len(page_spans),
        "char_count": sum(len(span.get("text") or "") for span in page_spans),
        "error": error,
    }
    return page_spans, report


def _run_ocr(
    page: Any,
    page_number: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str, str | None]:
    try:
        spans = ocr_page(page, page_number, timeout=OCR_TIMEOUT_SEC)
        return spans, "ocr", None
    except OcrUnavailableError as exc:
        warnings.append(f"Page {page_number}: {exc}")
        return [], "ocr_unavailable", str(exc)
    except Exception as exc:
        message = f"OCR failed: {exc}"
        warnings.append(f"Page {page_number}: {message}")
        return [], "ocr_failed", message


def _page_source(method: str) -> str:
    if method == "native":
        return "native"
    if method == "ocr":
        return "ocr"
    if method in {"ocr_skipped", "ocr_unavailable", "ocr_failed"}:
        return method
    return "empty"


def _document_type(pages: list[dict[str, Any]]) -> str:
    kinds = {page["kind"] for page in pages if page["kind"] != "empty"}
    if not kinds:
        return "empty"
    if kinds == {"native"}:
        return "native"
    if kinds <= {"scanned", "garbled"}:
        return "scanned"
    return "mixed"
