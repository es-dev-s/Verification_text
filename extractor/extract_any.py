"""
Unified multi-format text extraction (no Gemini).
"""

from __future__ import annotations

import os
from typing import Any

from extractor.docx_extract import extract_docx
from extractor.image_extract import ocr_image_file
from extractor.pdf_utils import reconstruct_text, spans_to_lines
from extractor.text_extract import extract_full_document


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}


def detect_kind(filename: str | None, path: str) -> str:
    name = (filename or path or "").lower()
    _, ext = os.path.splitext(name)
    if ext in PDF_EXTS:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in DOCX_EXTS:
        return "docx"
    raise ValueError(f"Unsupported format: {ext or 'unknown'}. Accepted: PDF, PNG, JPG, DOCX")


def extract_text_file(path: str, *, filename: str | None = None) -> dict[str, Any]:
    kind = detect_kind(filename, path)

    if kind == "pdf":
        result = extract_full_document(path, filename=filename)
        pages = result.get("pages") or []
        methods = {p.get("method") for p in pages if p.get("method")}
        method = "+".join(sorted(m for m in methods if m)) or "pdf"
        confs = [
            float(p["ocr_confidence"])
            for p in pages
            if p.get("ocr_confidence") is not None
        ]
        return {
            "ok": bool((result.get("raw_text") or "").strip()),
            "filename": filename or result.get("filename"),
            "raw_text": result.get("raw_text") or "",
            "method": method,
            "ocr_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "page_count": result.get("page_count"),
            "pages": pages,
            "warnings": result.get("warnings") or [],
            "pdf_type": result.get("pdf_type"),
        }

    if kind == "image":
        spans, avg = ocr_image_file(path)
        lines = spans_to_lines(spans)
        raw_text = reconstruct_text(lines) if lines else "\n".join(
            s["text"] for s in spans if s.get("text")
        )
        if not raw_text.strip() and spans:
            raw_text = "\n".join(s["text"] for s in spans)
        return {
            "ok": bool(raw_text.strip()),
            "filename": filename,
            "raw_text": raw_text,
            "method": "ocr",
            "ocr_confidence": avg,
            "page_count": 1,
            "pages": [
                {
                    "page": 1,
                    "method": "ocr",
                    "source": "ocr",
                    "ocr_confidence": avg,
                    "char_count": len(raw_text),
                }
            ],
            "warnings": [],
        }

    return extract_docx(path, filename=filename)
