"""
Full-document PDF text extraction for CV parsing.

Reuses native PyMuPDF extraction and PaddleOCR from this package.
Unlike title extraction, every uploaded page is processed — no front-matter
cap and no OCR page budget.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from extractor.ocr import (
    OcrUnavailableError,
    ocr_prepared,
    ocr_status,
    prepare_ocr_page,
)
from extractor.pdf_utils import (
    classify_page,
    extract_page_spans,
    extract_usable_native_spans,
    open_pdf,
    reconstruct_text,
    spans_to_lines,
)

OCR_TIMEOUT_SEC = 120.0


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

        plans = _plan_pages(doc, use_ocr, warnings)
        page_results = _execute_plans(pdf_path, doc, plans, warnings)

        for page_index in range(document_page_count):
            page = doc[page_index]
            page_number = page_index + 1
            page_geom[page_number] = (float(page.rect.width), float(page.rect.height))
            page_spans, report = page_results[page_index]
            spans.extend(page_spans)
            pages.append(report)

        lines = _spans_to_document_lines(spans)
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


def _plan_pages(
    doc: Any,
    use_ocr: bool,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Classify each page and decide native vs OCR without running Paddle yet."""
    plans: list[dict[str, Any]] = []
    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        kind = classify_page(page)
        plan: dict[str, Any] = {
            "page_index": page_index,
            "page_number": page_number,
            "kind": kind,
            "action": "empty",
            "spans": [],
            "error": None,
        }

        if kind == "native":
            plan["action"] = "native"
            plan["spans"] = extract_page_spans(page, page_number)
        elif kind in {"scanned", "garbled"}:
            native_spans = extract_usable_native_spans(page, page_number)
            if native_spans:
                plan["action"] = "native"
                plan["spans"] = native_spans
            elif not use_ocr:
                plan["action"] = "ocr_skipped"
                plan["error"] = "OCR skipped"
            else:
                plan["action"] = "ocr"
        plans.append(plan)
    return plans


def _render_ocr_page_from_path(
    pdf_path: str,
    page_index: int,
    page_number: int,
) -> dict[str, Any]:
    """Render in a worker thread with its own Document (PyMuPDF is not thread-safe)."""
    doc = open_pdf(pdf_path)
    try:
        return prepare_ocr_page(doc[page_index], page_number)
    finally:
        doc.close()


def _execute_plans(
    pdf_path: str,
    doc: Any,
    plans: list[dict[str, Any]],
    warnings: list[str],
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Run OCR with one-page render prefetch overlapping the current predict()."""
    results: list[tuple[list[dict[str, Any]], dict[str, Any]]] = [
        ([], {}) for _ in plans
    ]
    ocr_indices = [i for i, plan in enumerate(plans) if plan["action"] == "ocr"]

    with ThreadPoolExecutor(max_workers=1) as render_pool:
        prefetch: Future[dict[str, Any]] | None = None
        prefetch_index: int | None = None

        def _start_prefetch(from_pos: int) -> None:
            nonlocal prefetch, prefetch_index
            for pos in range(from_pos, len(ocr_indices)):
                idx = ocr_indices[pos]
                plan = plans[idx]
                prefetch = render_pool.submit(
                    _render_ocr_page_from_path,
                    pdf_path,
                    plan["page_index"],
                    plan["page_number"],
                )
                prefetch_index = idx
                return
            prefetch = None
            prefetch_index = None

        if ocr_indices:
            _start_prefetch(0)

        ocr_pos = 0
        for index, plan in enumerate(plans):
            if plan["action"] != "ocr":
                results[index] = _report_from_plan(plan)
                continue

            try:
                if prefetch is not None and prefetch_index == index:
                    prepared = prefetch.result()
                else:
                    prepared = prepare_ocr_page(
                        doc[plan["page_index"]],
                        plan["page_number"],
                    )
                ocr_pos += 1
                _start_prefetch(ocr_pos)
                page_spans = ocr_prepared(prepared)
                plan = {
                    **plan,
                    "spans": page_spans,
                    "action": "ocr",
                    "error": None,
                }
            except OcrUnavailableError as exc:
                warnings.append(f"Page {plan['page_number']}: {exc}")
                plan = {
                    **plan,
                    "spans": [],
                    "action": "ocr_unavailable",
                    "error": str(exc),
                }
                ocr_pos += 1
                _start_prefetch(ocr_pos)
            except Exception as exc:
                message = f"OCR failed: {exc}"
                warnings.append(f"Page {plan['page_number']}: {message}")
                plan = {
                    **plan,
                    "spans": [],
                    "action": "ocr_failed",
                    "error": message,
                }
                ocr_pos += 1
                _start_prefetch(ocr_pos)

            results[index] = _report_from_plan(plan)

    return results


def _report_from_plan(
    plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_spans = list(plan.get("spans") or [])
    method = str(plan.get("action") or "empty")
    confidences = [
        float(span["confidence"])
        for span in page_spans
        if span.get("source") == "ocr" and span.get("confidence") is not None
    ]
    ocr_confidence = (
        round(sum(confidences) / len(confidences), 1) if confidences else None
    )
    report = {
        "page": plan["page_number"],
        "kind": plan["kind"],
        "source": _page_source(method),
        "method": method,
        "ocr_confidence": ocr_confidence,
        "span_count": len(page_spans),
        "char_count": sum(len(span.get("text") or "") for span in page_spans),
        "error": plan.get("error"),
    }
    return page_spans, report


def _spans_to_document_lines(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep PaddleOCR visual lines intact; only re-cluster native PDF spans."""
    if spans and all(span.get("source") == "ocr" for span in spans):
        ordered = sorted(
            spans,
            key=lambda item: (
                int(item.get("page") or 0),
                float(item["bbox"][1]),
                float(item["bbox"][0]),
            ),
        )
        lines: list[dict[str, Any]] = []
        for index, span in enumerate(ordered):
            text = (span.get("text") or "").strip()
            if not text:
                continue
            lines.append(
                {
                    "text": text,
                    "page": int(span.get("page") or 1),
                    "font_name": span.get("font_name") or "ocr",
                    "font_size": float(span.get("font_size") or 0.0),
                    "bold": bool(span.get("bold")),
                    "italic": bool(span.get("italic")),
                    "bbox": list(span.get("bbox") or [0, 0, 0, 0]),
                    "block": span.get("block", 0),
                    "line": span.get("line", index),
                    "source": "ocr",
                    "span_count": 1,
                }
            )
        return lines
    return spans_to_lines(spans)


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
