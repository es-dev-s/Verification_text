"""
app.py

Flask entry point for multi-format text extraction (no Gemini).

POST /extract-text  — PDF / PNG / JPG / DOCX → raw_text + method + ocr_confidence
"""

import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VENV_SITE = _ROOT / ".venv" / "Lib" / "site-packages"
if _VENV_SITE.is_dir() and str(_VENV_SITE) not in sys.path:
    sys.path.insert(0, str(_VENV_SITE))

from flask import Flask, request, render_template, jsonify

from extractor.extract_any import extract_text_file

app = Flask(__name__)

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET", "HEAD"])
def health():
    return jsonify({"ok": True})


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin") or "*"
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Vary"] = "Origin"
    return response


@app.route("/extract-text", methods=["POST", "OPTIONS"])
def extract_text():
    if request.method == "OPTIONS":
        return ("", 204)

    uploaded = request.files.get("file") or request.files.get("pdf")
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "No file uploaded", "ok": False}), 400

    original = uploaded.filename or "upload.bin"
    _, ext = os.path.splitext(original.lower())
    if ext not in ALLOWED_EXT:
        return jsonify({
            "ok": False,
            "error": "Accepted formats: PDF, PNG, JPG, DOCX",
        }), 400

    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)
    uploaded.save(temp_path)

    try:
        result = extract_text_file(temp_path, filename=original)
        raw = result.get("raw_text") or ""
        _safe_print(
            f"[extract-text] {original} | method={result.get('method')} | "
            f"chars={len(raw)} | ok={result.get('ok')} | "
            f"pages={result.get('page_count')} | "
            f"warnings={len(result.get('warnings') or [])}"
        )
        preview = raw[:2000]
        _safe_print(
            "----- EXTRACT-TEXT RETURNED -----\n"
            f"{preview}"
            + ("\n... (truncated)" if len(raw) > 2000 else "")
            + "\n----- END EXTRACT-TEXT -----"
        )
        if result.get("warnings"):
            for warning in result["warnings"][:10]:
                _safe_print(f"[extract-text] warning: {warning}")
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Extraction failed: {exc}"}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _safe_print(text: str) -> None:
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    line = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(line)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000") or "5000")
    # Reloader kills in-flight OCR (several minutes) and the browser shows
    # "Failed to fetch" even when PaddleOCR was still working.
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
