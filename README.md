# Titlextractor (text extraction service)

Multi-format **text extraction only** (native PDF text, OCR, DOCX). No Gemini.
The Verification Engine API workers call this service, then Node runs Gemini parse jobs.

## Formats

| Format | Method |
|--------|--------|
| PDF | Per page: native text, Tesseract OCR when little/no text |
| PNG / JPG | Tesseract OCR |
| DOCX | Paragraphs + tables; Tesseract OCR on embedded images if almost no text |

## Setup

```bash
pip install -r requirements.txt
```

OCR uses **Tesseract** via `pytesseract`. Install the Tesseract binary on the host
(or set `TESSERACT_CMD`). Scanned PDF pages are rendered at **400 DPI grayscale**,
word boxes are mapped back to PDF coordinates, and dual-pass PSM settings keep
dense transcript tables readable. Optional knobs: `TESSERACT_DPI`,
`TESSERACT_LANG`, `TESSERACT_TIMEOUT_SEC`, `TESSERACT_PREPROCESS`,
`TESSERACT_REMOVE_GRID`.

## Run

```bash
python app.py
```

`GET /health` → `{"ok": true}`

### Product API

`POST /extract-text` with form field `file` (PDF / PNG / JPG / DOCX):

```json
{
  "ok": true,
  "raw_text": "...",
  "method": "native+ocr",
  "ocr_confidence": 82.5,
  "pages": [{"page": 1, "method": "native", "ocr_confidence": null}],
  "warnings": []
}
```

```bash
curl -F "file=@resume.pdf" http://localhost:5000/extract-text
```

Debug UI at `http://localhost:5000` exercises the same `/extract-text` endpoint.

## Production

See `Dockerfile` / `docker-compose.yml`. Gunicorn timeout should cover OCR-heavy PDFs
(default 900s). Per-page Tesseract timeout defaults to 180s (`TESSERACT_TIMEOUT_SEC`).
Gemini API keys belong in the Verification Engine `api/.env`, not here.
