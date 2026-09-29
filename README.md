# Titlextractor (text extraction service)

Multi-format **text extraction only** (native PDF text, OCR, DOCX). No Gemini.
The Verification Engine API workers call this service, then Node runs Gemini parse jobs.

## Formats

| Format | Method |
|--------|--------|
| PDF | Per page: native text, OCR when little/no text |
| PNG / JPG | OCR |
| DOCX | Paragraphs + tables; OCR embedded images if almost no text |

## Setup

```bash
pip install -r requirements.txt
```

Tesseract must be on `PATH` for OCR (images / scanned PDFs / DOCX images).

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

See `Dockerfile` / `docker-compose.yml`. Gunicorn timeout should cover OCR-heavy PDFs (e.g. 180s).
Gemini API keys belong in the Verification Engine `api/.env`, not here.
