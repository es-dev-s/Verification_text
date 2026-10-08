"""
PaddleOCR fallback for scanned / garbled PDF pages and image uploads.

Renders pages at high DPI, runs PP-OCR, and groups detections into readable
lines. Table-like layouts use column separators so downstream LLM extraction
gets one field group per line instead of a single mashed paragraph.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import fitz
import numpy as np
from PIL import Image

# Avoid slow model-host checks on cold start (PaddleX 3.x).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_DPI = 200
TITLE_BAND = 1.0
MAX_OCR_WIDTH = 1600
OCR_TIMEOUT_SEC = 120
_COLUMN_GAP_WIDTH_RATIO = 1.15
_COLUMN_GAP_MIN_PX = 18.0
# Dense transcript tables need tight vertical banding so rows don't collapse.
_LINE_Y_OVERLAP_RATIO = 0.55
_LINE_Y_MID_RATIO = 0.35
# Near-blank pages: skip Paddle when there is almost no ink.
_BLANK_MAX_STD = 8.0
_BLANK_MIN_MEAN = 245.0
# Mobile PP-OCR models — much faster on CPU than medium/server.
_DET_MODEL = os.environ.get("PADDLEOCR_DET_MODEL", "PP-OCRv5_mobile_det")
_REC_MODEL = os.environ.get("PADDLEOCR_REC_MODEL", "en_PP-OCRv5_mobile_rec")

_PaddleOCR: Any = None

# Pool of independent PaddleOCR instances. A single instance is not thread-safe,
# so each engine is checked out by exactly one thread at a time; different
# engines run in parallel (pages of one document and concurrent requests).
_POOL: "_EnginePool | None" = None
_POOL_INIT_LOCK = threading.Lock()
_POOL_STATE: dict[str, Any] = {"state": "idle", "reason": "not loaded yet"}
_INIT_RETRY_COOLDOWN_SEC = 30.0
_LAST_INIT_FAILURE: dict[str, Any] = {"at": 0.0, "error": None}
# Set after oneDNN fails at runtime so later pool builds skip it.
_MKLDNN_DISABLED_AT_RUNTIME = False
# oneDNN/PIR crash on CPU: Paddle 3.3.0 and 3.3.1 (PaddlePaddle/Paddle#77340,
# PaddleOCR#18162); 3.2.x is the newest release confirmed fine.
_MKLDNN_BAD_SERIES = ((3, 3),)
_ONEDNN_ERROR_MARKERS = ("onednn", "mkldnn", "convertpirattribute2runtimeattribute")


class OcrUnavailableError(RuntimeError):
    """PaddleOCR or its dependencies are missing or failed to load."""


def _log(message: str) -> None:
    text = f"[ocr] {message}"
    try:
        print(text, flush=True)
    except Exception:
        try:
            sys.stdout.write(text.encode("ascii", errors="replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:
            pass


def _load_bindings() -> bool:
    global _PaddleOCR
    if _PaddleOCR is not None:
        return True
    try:
        from paddleocr import PaddleOCR as _POCR
    except ImportError:
        return False
    _PaddleOCR = _POCR
    return True


def _paddle_lang() -> str:
    return (os.environ.get("PADDLEOCR_LANG") or "en").strip() or "en"


def _env_int(name: str) -> int | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        _log(f"ignoring {name}={raw!r} (not an integer)")
        return None
    return value if value >= 1 else None


def _paddle_version() -> tuple[str | None, tuple[int, int] | None]:
    """Installed paddlepaddle version without importing paddle itself."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None, None
    for dist in ("paddlepaddle", "paddlepaddle-gpu"):
        try:
            text = version(dist)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
        parts = text.split(".")
        try:
            return text, (int(parts[0]), int("".join(ch for ch in parts[1] if ch.isdigit()) or 0))
        except (IndexError, ValueError):
            return text, None
    return None, None


def _mkldnn_decision() -> tuple[bool, str]:
    """
    Decide whether to try oneDNN (MKLDNN) on CPU.

    PADDLEOCR_ENABLE_MKLDNN=1/true/on forces it on, 0/false/off forces it off,
    unset/auto enables it unless the installed Paddle is a known-bad series.
    A failed init or warm-up always falls back to oneDNN off.
    """
    raw = (os.environ.get("PADDLEOCR_ENABLE_MKLDNN") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False, "forced off by PADDLEOCR_ENABLE_MKLDNN"
    if _MKLDNN_DISABLED_AT_RUNTIME:
        return False, "disabled after a oneDNN runtime failure"
    if raw in {"1", "true", "yes", "on"}:
        return True, "forced on by PADDLEOCR_ENABLE_MKLDNN"
    text, series = _paddle_version()
    if series is None:
        return True, f"auto (paddlepaddle {text or 'unknown'}; falls back if it fails)"
    if series in _MKLDNN_BAD_SERIES:
        return False, f"auto off: paddlepaddle {text} has the oneDNN/PIR crash"
    return True, f"auto on (paddlepaddle {text})"


def _physical_cores() -> int:
    logical = os.cpu_count() or 1
    physical: int | None = None
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    if not physical:
        try:
            pairs: set[tuple[str, str]] = set()
            phys_id = "0"
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        pairs.add((phys_id, line.split(":", 1)[1].strip()))
            physical = len(pairs) or None
        except Exception:
            physical = None
    cores = physical or logical
    # Respect CPU affinity / container cpusets where the OS exposes them.
    try:
        allowed = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        cores = min(cores, allowed)
    except Exception:
        pass
    return max(1, int(cores))


def _process_count() -> int:
    """Gunicorn runs WEB_CONCURRENCY worker processes, each with its own pool."""
    if "gunicorn" in sys.modules:
        return _env_int("WEB_CONCURRENCY") or 1
    return 1


def _pool_size() -> int:
    configured = _env_int("PADDLEOCR_WORKERS")
    if configured:
        return configured
    # Each engine holds its own copy of the models (~hundreds of MB), so stay small.
    return max(1, min(2, _physical_cores() // 2))


def _cpu_threads(pool_size: int) -> int:
    configured = _env_int("PADDLEOCR_CPU_THREADS")
    if configured:
        return configured
    return max(1, _physical_cores() // max(1, pool_size * _process_count()))


def _gpu_available() -> bool:
    try:
        import paddle  # type: ignore

        if not paddle.device.is_compiled_with_cuda():
            return False
        return int(paddle.device.cuda.device_count()) > 0
    except Exception:
        return False


def _requested_device() -> str:
    """PADDLEOCR_DEVICE=cpu|gpu|gpu:N|auto (auto: first GPU on a CUDA build, else CPU)."""
    raw = (os.environ.get("PADDLEOCR_DEVICE") or "auto").strip().lower() or "auto"
    if raw == "auto":
        return "gpu:0" if _gpu_available() else "cpu"
    if raw.startswith("gpu") and not _gpu_available():
        _log(f"PADDLEOCR_DEVICE={raw} but no CUDA build/GPU is available; using cpu")
        return "cpu"
    return raw


def _warmup_image() -> np.ndarray:
    """Small synthetic page with real glyphs so both det and rec actually run."""
    from PIL import ImageDraw, ImageFont

    image = Image.new("RGB", (720, 160), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font: Any = ImageFont.load_default(size=40)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 20), "University Transcript", fill=(0, 0, 0), font=font)
    draw.text((20, 90), "CS 1010  Grade A  3.0", fill=(0, 0, 0), font=font)
    return np.array(image)


def _build_engine(device: str, mkldnn: bool, cpu_threads: int) -> Any:
    # Mobile det/rec + capped side length keeps CPU OCR usable for multi-page transcripts.
    kwargs: dict[str, Any] = {
        "lang": _paddle_lang(),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": mkldnn,
        "text_detection_model_name": _DET_MODEL,
        "text_recognition_model_name": _REC_MODEL,
        "text_det_limit_side_len": 960,
        "text_det_limit_type": "max",
        "device": device,
    }
    if device == "cpu":
        kwargs["cpu_threads"] = cpu_threads
    return _PaddleOCR(**kwargs)


def _build_and_warm(device: str, mkldnn: bool, cpu_threads: int) -> Any:
    engine = _build_engine(device, mkldnn, cpu_threads)
    list(engine.predict(_warmup_image()))
    return engine


def _looks_like_onednn_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _ONEDNN_ERROR_MARKERS)


class _EnginePool:
    def __init__(self, engines: list[Any], config: dict[str, Any]) -> None:
        self.size = len(engines)
        self.config = config
        self._idle: "queue.Queue[Any]" = queue.Queue()
        for engine in engines:
            self._idle.put(engine)

    @contextmanager
    def engine(self) -> Iterator[Any]:
        engine = self._idle.get()
        try:
            yield engine
        finally:
            self._idle.put(engine)


def _create_pool() -> _EnginePool:
    """Build N engines; the first one decides device/oneDNN via warm-up fallbacks."""
    size = _pool_size()
    threads = _cpu_threads(size)
    device = _requested_device()
    want_mkldnn, mkldnn_reason = _mkldnn_decision()

    attempts: list[tuple[str, bool]] = []
    if device != "cpu":
        attempts.append((device, False))
    if want_mkldnn:
        attempts.append(("cpu", True))
    attempts.append(("cpu", False))

    started = time.perf_counter()
    first: Any = None
    chosen: tuple[str, bool] | None = None
    last_exc: BaseException | None = None
    for attempt_device, attempt_mkldnn in attempts:
        try:
            first = _build_and_warm(attempt_device, attempt_mkldnn, threads)
            chosen = (attempt_device, attempt_mkldnn)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            label = f"device={attempt_device} mkldnn={'on' if attempt_mkldnn else 'off'}"
            _log(f"engine init/warm-up failed ({label}): {type(exc).__name__}: {exc}")
            if attempt_mkldnn:
                _log("falling back to oneDNN off")
            elif attempt_device != "cpu":
                _log("falling back to cpu")
    if chosen is None:
        raise OcrUnavailableError(f"PaddleOCR init failed: {last_exc}")

    engines = [first]
    for index in range(1, size):
        try:
            engines.append(_build_and_warm(chosen[0], chosen[1], threads))
        except Exception as exc:  # noqa: BLE001
            _log(
                f"could not create OCR engine {index + 1}/{size} "
                f"({type(exc).__name__}: {exc}); continuing with {len(engines)}"
            )
            break

    config = {
        "workers": len(engines),
        "device": chosen[0],
        "mkldnn": chosen[1],
        "mkldnn_reason": mkldnn_reason if chosen[1] == want_mkldnn else "fell back to off",
        "cpu_threads": threads if chosen[0] == "cpu" else None,
        "det_model": _DET_MODEL,
        "rec_model": _REC_MODEL,
    }
    _log(
        f"pool ready in {time.perf_counter() - started:.1f}s: workers={config['workers']} "
        f"device={config['device']} mkldnn={'on' if config['mkldnn'] else 'off'} "
        f"({config['mkldnn_reason']}) cpu_threads={config['cpu_threads']} "
        f"det={_DET_MODEL} rec={_REC_MODEL}"
    )
    return _EnginePool(engines, config)


def _get_pool() -> _EnginePool:
    """Return the shared pool, building it once (concurrent callers wait for it)."""
    global _POOL
    pool = _POOL
    if pool is not None:
        return pool
    if not _load_bindings():
        raise OcrUnavailableError(
            "OCR extras are not installed. Run: pip install paddleocr paddlepaddle"
        )
    with _POOL_INIT_LOCK:
        if _POOL is not None:
            return _POOL
        since_failure = time.monotonic() - float(_LAST_INIT_FAILURE["at"] or 0.0)
        if _LAST_INIT_FAILURE["error"] and since_failure < _INIT_RETRY_COOLDOWN_SEC:
            # Don't hammer a broken install with a full model load on every page.
            raise OcrUnavailableError(str(_LAST_INIT_FAILURE["error"]))
        _POOL_STATE.update(state="loading", reason="loading models")
        try:
            _POOL = _create_pool()
        except Exception as exc:  # noqa: BLE001
            message = str(exc) if isinstance(exc, OcrUnavailableError) else f"PaddleOCR init failed: {exc}"
            _LAST_INIT_FAILURE.update(at=time.monotonic(), error=message)
            _POOL_STATE.update(state="failed", reason=message)
            raise OcrUnavailableError(message) from exc
        _LAST_INIT_FAILURE.update(at=0.0, error=None)
        _POOL_STATE.update(state="ready", reason="ready")
        return _POOL


def _disable_mkldnn_and_reset(failed_pool: _EnginePool) -> None:
    """oneDNN broke during real inference: rebuild the pool without it."""
    global _POOL, _MKLDNN_DISABLED_AT_RUNTIME
    with _POOL_INIT_LOCK:
        if _POOL is failed_pool:
            _MKLDNN_DISABLED_AT_RUNTIME = True
            _POOL = None
            _POOL_STATE.update(state="idle", reason="reloading without oneDNN")
            _log("oneDNN failed during inference; rebuilding OCR pool with oneDNN off")


def _predict(image_rgb: np.ndarray) -> list[Any]:
    pool = _get_pool()
    try:
        with pool.engine() as engine:
            return list(engine.predict(image_rgb))
    except Exception as exc:  # noqa: BLE001
        if not (pool.config.get("mkldnn") and _looks_like_onednn_error(exc)):
            raise
        _disable_mkldnn_and_reset(pool)
    with _get_pool().engine() as engine:
        return list(engine.predict(image_rgb))


def ocr_pool_size() -> int:
    """Engines in the pool (configured size until the pool is built)."""
    pool = _POOL
    return pool.size if pool is not None else _pool_size()


def warm_ocr_engine() -> dict[str, Any]:
    """Build the OCR engine pool once so the first real request skips cold model init."""
    try:
        _get_pool()
    except Exception:  # noqa: BLE001
        pass
    return ocr_status()


def ocr_is_available() -> bool:
    return ocr_status()["available"]


def ocr_status() -> dict[str, Any]:
    """Non-blocking status: never waits on model loading or running OCR."""
    if not _load_bindings():
        return {
            "available": False,
            "engine": "paddleocr",
            "reason": "paddleocr not installed in this Python",
        }
    state = _POOL_STATE.get("state")
    reason = str(_POOL_STATE.get("reason") or "")
    if state == "failed":
        return {"available": False, "engine": "paddleocr", "reason": reason}
    status: dict[str, Any] = {
        "available": True,
        "engine": "paddleocr",
        "reason": "ready" if state == "ready" else reason,
    }
    pool = _POOL
    if pool is not None:
        status["config"] = dict(pool.config)
    return status


def ocr_page(
    page: fitz.Page,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    timeout: float = OCR_TIMEOUT_SEC,
    band: float = TITLE_BAND,
) -> list[dict[str, Any]]:
    """OCR one PDF page and return native-shaped spans (one span per visual line)."""
    del timeout  # reserved for API compatibility; Paddle has no per-call timeout hook
    prepared = prepare_ocr_page(page, page_number, dpi=dpi, band=band)
    return ocr_prepared(prepared)


def prepare_ocr_page(
    page: fitz.Page,
    page_number: int,
    dpi: int = DEFAULT_DPI,
    band: float = TITLE_BAND,
) -> dict[str, Any]:
    """Render a PDF page to RGB for OCR (no OCR engine needed; thread-safe per Document)."""
    rect = page.rect
    clip = fitz.Rect(
        rect.x0,
        rect.y0,
        rect.x1,
        rect.y0 + rect.height * min(max(band, 0.35), 1.0),
    )
    image_rgb, image_w, image_h = _render_page_rgb(page, clip, dpi)
    return {
        "image_rgb": image_rgb,
        "page_number": page_number,
        "clip_origin": (clip.x0, clip.y0),
        "x_scale": clip.width / max(image_w, 1),
        "y_scale": clip.height / max(image_h, 1),
    }


def ocr_prepared(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    """Run PaddleOCR on a prepare_ocr_page() result."""
    return _ocr_rgb_array(
        prepared["image_rgb"],
        page_number=int(prepared["page_number"]),
        clip_origin=prepared.get("clip_origin", (0.0, 0.0)),
        x_scale=float(prepared.get("x_scale", 1.0)),
        y_scale=float(prepared.get("y_scale", 1.0)),
    )


def ocr_image_file_spans(
    path: str,
    *,
    page_number: int = 1,
    timeout: float = OCR_TIMEOUT_SEC,
) -> tuple[list[dict[str, Any]], float | None]:
    """Return OCR spans and average confidence (0–100) for an image file."""
    del timeout
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    if image.width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(image.width)
        image = image.resize(
            (MAX_OCR_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )

    spans = _ocr_rgb_array(np.array(image), page_number=page_number)
    confidences = [
        float(span["confidence"])
        for span in spans
        if span.get("confidence") is not None
    ]
    avg = round(sum(confidences) / len(confidences), 1) if confidences else None
    return spans, avg


def _render_page_rgb(
    page: fitz.Page,
    clip: fitz.Rect,
    dpi: int,
) -> tuple[np.ndarray, int, int]:
    # Keep requested DPI for recognition quality; only downscale after render.
    scale = max(int(dpi), 72) / 72.0
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csRGB,
        clip=clip,
    )
    # Direct numpy path — skip PIL when already at/under the OCR width.
    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    ).copy()
    width, height = pixmap.width, pixmap.height
    if width > MAX_OCR_WIDTH:
        ratio = MAX_OCR_WIDTH / float(width)
        new_h = max(1, int(height * ratio))
        image = Image.fromarray(array)
        image = image.resize((MAX_OCR_WIDTH, new_h), Image.BILINEAR)
        array = np.array(image)
        width, height = image.width, image.height
    return array, width, height


def _is_near_blank(image_rgb: np.ndarray) -> bool:
    """True when the page is essentially empty (no ink worth OCR)."""
    if image_rgb.size == 0:
        return True
    # Sample a coarse grid for speed on large pages.
    sample = image_rgb[::8, ::8]
    if sample.size == 0:
        return True
    gray = sample.mean(axis=2) if sample.ndim == 3 else sample.astype(np.float64)
    return float(gray.std()) < _BLANK_MAX_STD and float(gray.mean()) > _BLANK_MIN_MEAN


def _ocr_rgb_array(
    image_rgb: np.ndarray,
    *,
    page_number: int,
    clip_origin: tuple[float, float] = (0.0, 0.0),
    x_scale: float = 1.0,
    y_scale: float = 1.0,
) -> list[dict[str, Any]]:
    if _is_near_blank(image_rgb):
        return []
    _ensure_paddle()
    raw = _predict(image_rgb)
    detections = _parse_paddle_results(raw)
    lines = _group_detections_into_lines(detections)
    return _lines_to_spans(
        lines,
        page_number=page_number,
        clip_origin=clip_origin,
        x_scale=x_scale,
        y_scale=y_scale,
    )


def _ensure_paddle() -> None:
    if not _load_bindings():
        raise OcrUnavailableError(
            "OCR extras are not installed. Run: pip install paddleocr paddlepaddle"
        )
    try:
        _get_pool()
    except Exception as exc:
        raise OcrUnavailableError(f"PaddleOCR is unavailable: {exc}") from exc


def _parse_paddle_results(results: list[Any]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for page_result in results:
        payload = _result_payload(page_result)
        texts = _as_list(payload.get("rec_texts"))
        scores = _as_list(payload.get("rec_scores"))
        boxes = _as_list(payload.get("rec_boxes"))
        if not boxes:
            boxes = _as_list(payload.get("dt_polys"))
        for index, text in enumerate(texts):
            cleaned = (str(text) if text is not None else "").strip()
            if not cleaned:
                continue
            box = boxes[index] if index < len(boxes) else None
            bbox_px = _normalize_box(box)
            if bbox_px is None:
                continue
            score_raw = scores[index] if index < len(scores) else 0.0
            detections.append(
                {
                    "text": cleaned,
                    "bbox_px": bbox_px,
                    "confidence": _score_to_percent(score_raw),
                }
            )
    return detections


def _as_list(value: Any) -> list[Any]:
    """Convert Paddle outputs (list / ndarray / None) without boolean tests on arrays."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _result_payload(page_result: Any) -> dict[str, Any]:
    if isinstance(page_result, dict):
        inner = page_result.get("res")
        return inner if isinstance(inner, dict) else page_result
    json_attr = getattr(page_result, "json", None)
    if isinstance(json_attr, dict):
        inner = json_attr.get("res")
        return inner if isinstance(inner, dict) else json_attr
    if hasattr(page_result, "get"):
        try:
            inner = page_result.get("res")
            if isinstance(inner, dict):
                return inner
        except Exception:
            pass
        try:
            return dict(page_result)
        except Exception:
            return {}
    return {}


def _normalize_box(box: Any) -> list[float] | None:
    if box is None:
        return None
    if isinstance(box, np.ndarray):
        box = box.tolist()
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in box
    ):
        x0, y0, x1, y1 = (float(v) for v in box)
        return [x0, y0, x1, y1]
    if isinstance(box, (list, tuple)) and len(box) > 0:
        first = box[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, (list, tuple)):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _score_to_percent(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1.0:
        score *= 100.0
    return round(max(0.0, min(score, 100.0)), 1)


def _group_detections_into_lines(
    detections: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if not detections:
        return []
    ordered = sorted(
        detections,
        key=lambda item: (
            (item["bbox_px"][1] + item["bbox_px"][3]) / 2.0,
            item["bbox_px"][0],
        ),
    )
    lines: list[list[dict[str, Any]]] = []
    for detection in ordered:
        placed = False
        for line in lines:
            if _same_text_line(detection, line):
                line.append(detection)
                placed = True
                break
        if not placed:
            lines.append([detection])
    for line in lines:
        line.sort(key=lambda item: item["bbox_px"][0])
    lines.sort(
        key=lambda cluster: (
            min(item["bbox_px"][1] for item in cluster),
            min(item["bbox_px"][0] for item in cluster),
        )
    )
    return lines


def _same_text_line(detection: dict[str, Any], line: list[dict[str, Any]]) -> bool:
    dx0, dy0, dx1, dy1 = detection["bbox_px"]
    d_mid = (dy0 + dy1) / 2.0
    d_h = max(dy1 - dy0, 1.0)
    ly0 = min(item["bbox_px"][1] for item in line)
    ly1 = max(item["bbox_px"][3] for item in line)
    l_mid = (ly0 + ly1) / 2.0
    l_h = max(ly1 - ly0, max(item["bbox_px"][3] - item["bbox_px"][1] for item in line), 1.0)
    overlap = min(dy1, ly1) - max(dy0, ly0)
    # Require clear vertical overlap; reject stacked table rows.
    if overlap < _LINE_Y_OVERLAP_RATIO * min(d_h, l_h):
        return False
    if abs(d_mid - l_mid) > _LINE_Y_MID_RATIO * max(d_h, l_h):
        return False
    return True


def _format_line_text(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""
    if len(cells) == 1:
        return cells[0]["text"]
    widths = [max(cell["bbox_px"][2] - cell["bbox_px"][0], 1.0) for cell in cells]
    median_w = sorted(widths)[len(widths) // 2]
    gap_threshold = max(_COLUMN_GAP_MIN_PX, median_w * _COLUMN_GAP_WIDTH_RATIO)
    parts: list[str] = []
    previous_x1: float | None = None
    for cell in cells:
        text = cell["text"]
        x0, _, x1, _ = cell["bbox_px"]
        if previous_x1 is not None and parts:
            gap = x0 - previous_x1
            if gap >= gap_threshold:
                parts.append(" | ")
            elif not parts[-1].endswith(" "):
                parts.append(" ")
        parts.append(text)
        previous_x1 = x1
    return "".join(parts).strip()


def _lines_to_spans(
    lines: list[list[dict[str, Any]]],
    *,
    page_number: int,
    clip_origin: tuple[float, float],
    x_scale: float,
    y_scale: float,
) -> list[dict[str, Any]]:
    ox, oy = clip_origin
    spans: list[dict[str, Any]] = []
    for line_index, cells in enumerate(lines):
        text = _format_line_text(cells)
        if not text:
            continue
        boxes = [cell["bbox_px"] for cell in cells]
        x0 = min(box[0] for box in boxes)
        y0 = min(box[1] for box in boxes)
        x1 = max(box[2] for box in boxes)
        y1 = max(box[3] for box in boxes)
        confidences = [float(cell.get("confidence") or 0.0) for cell in cells]
        font_size = round(max(y1 - y0, 1.0) * y_scale, 2)
        spans.append(
            {
                "text": text,
                "page": page_number,
                "font_name": "ocr",
                "font_size": font_size,
                "bold": False,
                "italic": False,
                "bbox": [
                    round(ox + x0 * x_scale, 2),
                    round(oy + y0 * y_scale, 2),
                    round(ox + x1 * x_scale, 2),
                    round(oy + y1 * y_scale, 2),
                ],
                "block": line_index // 8,
                "line": line_index,
                "source": "ocr",
                "confidence": round(sum(confidences) / max(len(confidences), 1), 1),
            }
        )
    return spans


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 2)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 2)
