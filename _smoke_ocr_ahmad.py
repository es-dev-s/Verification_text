"""OCR Ahmad transcript and count subjects from extracted text (no LLM)."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

# Windows consoles often can't print OCR oddities; keep UTF-8.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from extractor.text_extract import extract_full_document

PDF = Path(__file__).resolve().parent / "6.Ahmad bs  transcript.pdf"
OUT = Path(__file__).resolve().parent / "_ahmad_ocr_out.txt"

_CODE_RE = re.compile(r"(?<![A-Z])([A-Z]{2,5})\s?(\d{4})([A-Z]?)\b", re.I)
_FALSE = {"FALL", "SPRING", "SUMMER", "TOTAL", "DATE", "PAGE", "TERM", "IMAD", "MOH"}


def main() -> None:
    t0 = time.time()
    result = extract_full_document(str(PDF), filename=PDF.name)
    elapsed = time.time() - t0
    text = result.get("raw_text") or ""
    OUT.write_text(text, encoding="utf-8")

    pages = result.get("pages") or []
    methods = {p.get("method") for p in pages}
    print(f"ok={result.get('ok')} pages={result.get('page_count')} methods={methods}")
    print(f"elapsed_sec={elapsed:.1f} chars={len(text)} warnings={result.get('warnings')}")

    codes_in_order: list[str] = []
    for m in _CODE_RE.finditer(text):
        prefix = m.group(1).upper()
        if prefix in _FALSE:
            continue
        code = (prefix + m.group(2) + (m.group(3) or "")).upper()
        codes_in_order.append(code)

    unique: list[str] = []
    seen: set[str] = set()
    for code in codes_in_order:
        if code not in seen:
            seen.add(code)
            unique.append(code)

    has_internship = bool(re.search(r"\bInternship\b", text, re.I))

    print(f"\nunique_course_codes={len(unique)}")
    print(f"course_code_mentions={len(codes_in_order)}")
    print(f"internship_line={has_internship}")
    print("codes:", ", ".join(unique))
    print("\n--- extracted text ---")
    print(text)


if __name__ == "__main__":
    main()
