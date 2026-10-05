"""
Smoke + accuracy test: Tesseract OCR on the Ahmad BS transcript PDF.

Checks that every subject course code and all Term/Cum GPA values appear
in the geometry-preserving reconstructed text.
"""

from __future__ import annotations

import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ground truth from the scanned Ahmad Bilal BS (EE) transcript.
EXPECTED_CODES = {
    # Fall 2003
    "EE1003",
    "GSA1003",
    "GSA1023",
    "GSC1113",
    "GSC1223",
    # Spring 2004
    "CS1104",
    "GSA1013",
    "GSC1013",
    "GSC1123",
    "MATH2104",
    # Summer 2004
    "EE2201",
    "EE2203",
    "GSA2063",
    "MATH2114",
    # Fall 2004
    "EE2211",
    "EE2213",
    "EE3211",
    "EE3213",
    "MATH2124",
    "MATH3003",
    # Spring 2005
    "EE2331",
    "EE2333",
    "EE2413",
    "EE2513",
    "EE3101",
    "EE3103",
    "GSB2153",
    # Fall 2005
    "EE3231",
    "EE3233",
    "EE3711",
    "EE3713",
    "EE4701",
    "EE4703",
    "GSA2163",
    # Spring 2006
    "EE3243",
    "EE3331",
    "EE3333",
    "EE3813",
    "EE4523",
    "EE4733",
    # Fall 2006
    "EE3723",
    "EE4533",
    "EE4633",
    "EE4913",
    "MATH1303",
    "QM4023",
    # Spring 2007
    "EE4213",
    "EE4543",
    "EE4743",
    "EE4923",
    "QM3023",
}

EXPECTED_GPAS = {
    "3.00",
    "2.69",
    "2.86",
    "2.60",
    "2.77",
    "2.13",
    "3.24",
    "2.75",
    "3.40",
    "3.13",
    "2.90",
    "2.91",
    "3.80",
    "3.01",
}

CODE_RE = re.compile(
    r"(?:^|[^A-Z0-9])((?:MATH|GSB|GSC|GSA|QM|CS|EE)\d{3,4})(?=[^A-Z0-9]|$)",
    re.I,
)
GPA_RE = re.compile(r"(?<![0-9])(\d\.\d{2})(?![0-9])")


def find_pdf() -> str:
    preferred = [
        "ahmad bs transcript.pdf",
        "Ahmad bs transcript.pdf",
        "6.Ahmad bs  transcript.pdf",
        "6.Ahmad bs transcript.pdf",
    ]
    for name in preferred:
        path = os.path.join(ROOT, name)
        if os.path.isfile(path):
            return path

    for name in os.listdir(ROOT):
        lower = name.lower()
        if lower.endswith(".pdf") and "ahmad" in lower and "transcript" in lower:
            return os.path.join(ROOT, name)

    raise FileNotFoundError(
        f"Could not find Ahmad transcript PDF in {ROOT}. "
        "Expected something like 'ahmad bs transcript.pdf'."
    )


def main() -> int:
    from extractor.ocr import ocr_page, ocr_status
    from extractor.pdf_utils import open_pdf, reconstruct_text, spans_to_lines

    pdf_path = find_pdf()
    print(f"PDF: {pdf_path}")
    print(f"OCR status: {ocr_status()}")

    doc = open_pdf(pdf_path)
    try:
        print(f"Pages: {doc.page_count}")
        all_spans: list[dict] = []
        t0 = time.perf_counter()

        for page_index in range(doc.page_count):
            page_number = page_index + 1
            page = doc[page_index]
            print(f"\n--- OCR page {page_number}/{doc.page_count} ---")
            page_t0 = time.perf_counter()
            spans = ocr_page(page, page_number, timeout=180)
            elapsed = time.perf_counter() - page_t0
            confs = [
                float(s["confidence"])
                for s in spans
                if s.get("confidence") is not None
            ]
            avg = round(sum(confs) / len(confs), 1) if confs else None
            print(f"spans={len(spans)}  avg_confidence={avg}  took={elapsed:.1f}s")
            for span in spans[:12]:
                print(f"  [{span['confidence']:>5}] {span['text']}")
            if len(spans) > 12:
                print(f"  ... ({len(spans) - 12} more)")
            all_spans.extend(spans)

        total = time.perf_counter() - t0
        lines = spans_to_lines(all_spans)
        text = reconstruct_text(lines)

        print("\n========== RECONSTRUCTED TEXT ==========\n")
        print(text if text.strip() else "(empty)")
        print("\n========================================")

        found_codes = {m.group(1).upper() for m in CODE_RE.finditer(text)}
        found_gpas = set(GPA_RE.findall(text))
        missing_codes = sorted(EXPECTED_CODES - found_codes)
        missing_gpas = sorted(EXPECTED_GPAS - found_gpas)
        internship_ok = "internship" in text.lower()

        print(
            f"Done. pages={doc.page_count} spans={len(all_spans)} "
            f"lines={len(lines)} chars={len(text)} total={total:.1f}s"
        )
        print(
            f"Course codes: {len(EXPECTED_CODES) - len(missing_codes)}/"
            f"{len(EXPECTED_CODES)}"
        )
        print(f"GPA values: {len(EXPECTED_GPAS) - len(missing_gpas)}/{len(EXPECTED_GPAS)}")
        print(f"Internship present: {internship_ok}")
        if missing_codes:
            print("MISSING CODES:", ", ".join(missing_codes))
        if missing_gpas:
            print("MISSING GPAs:", ", ".join(missing_gpas))

        ok = (
            bool(text.strip())
            and not missing_codes
            and not missing_gpas
            and internship_ok
        )
        return 0 if ok else 1
    finally:
        doc.close()


if __name__ == "__main__":
    raise SystemExit(main())
