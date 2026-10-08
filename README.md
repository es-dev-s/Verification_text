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

OCR uses **PaddleOCR** (installed via `requirements.txt`). First run downloads model weights; set `PADDLEOCR_LANG` (default `en`) if needed.

### OCR speed settings (optional env vars)

| Variable | Default | Meaning |
|----------|---------|---------|
| `PADDLEOCR_WORKERS` | `min(2, physical_cores // 2)`, at least 1 | Number of PaddleOCR engines in the pool. Pages of one PDF and concurrent requests are OCR'd in parallel, one engine per page. Each engine holds its own copy of the models, so every extra worker costs extra RAM (roughly a few hundred MB). |
| `PADDLEOCR_CPU_THREADS` | `physical_cores // (workers × gunicorn workers)`, at least 1 | CPU threads per engine. |
| `PADDLEOCR_ENABLE_MKLDNN` | `auto` | oneDNN CPU acceleration. `1` forces on, `0` forces off. `auto` turns it on except on paddlepaddle 3.3.x, which crashes with oneDNN (use 3.2.2 from `requirements.txt` to get it). If oneDNN fails at load, warm-up or inference, the service falls back to oneDNN off automatically and logs `[ocr] ...`. |
| `PADDLEOCR_DEVICE` | `auto` | `auto` uses the first GPU on a CUDA (`paddlepaddle-gpu`) build, otherwise CPU. `cpu`, `gpu` or `gpu:N` to pin it. A GPU that fails to load falls back to CPU. |
| `PADDLEOCR_DET_MODEL` / `PADDLEOCR_REC_MODEL` | `PP-OCRv5_mobile_det` / `en_PP-OCRv5_mobile_rec` | Model names (unchanged). |

On startup the service logs one line such as
`[ocr] pool ready in 12.3s: workers=2 device=cpu mkldnn=on (...) cpu_threads=2 ...`
so you can confirm what was picked.

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
