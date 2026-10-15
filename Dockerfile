# titlextractor — Flask + PaddleOCR (CPU) text extraction service.
# Railway: set the service Root Directory to "titlextractor"; Railway injects PORT.
#
# Python 3.11: paddlepaddle 3.2.2 ships manylinux wheels for cp311.
# paddlepaddle is pinned to 3.2.x on purpose: 3.3.0/3.3.1 crash on CPU with
# oneDNN (see _MKLDNN_BAD_SERIES in extractor/ocr.py).
FROM python:3.11-slim-bookworm

# Runtime libraries for paddlepaddle (OpenMP) and OpenCV (GL/X11 libs for opencv-contrib-python).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PADDLE_PDX_CACHE_HOME keeps the PaddleX model cache inside the image (filled at build time below).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PADDLE_PDX_CACHE_HOME=/opt/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLEOCR_LANG=en \
    PADDLEOCR_DEVICE=cpu \
    PADDLEOCR_WORKERS=1 \
    OMP_NUM_THREADS=2 \
    FLAGS_allocator_strategy=auto_growth

WORKDIR /app

COPY requirements.txt .
# opencv-contrib-python (not -headless) on purpose: paddlex checks for that exact
# distribution name and refuses to build the OCR pipeline without it. The GL/X
# libraries it needs are installed above.
RUN pip install -r requirements.txt \
    && python -c "import cv2, fitz, docx, flask, gunicorn, paddle, paddleocr, psutil; print('imports ok, paddle', paddle.__version__, 'cv2', cv2.__version__)"

COPY app.py gunicorn.conf.py entrypoint.py ./
COPY extractor ./extractor
COPY templates ./templates

# Download and warm the same PaddleOCR models/config the app uses, so the first
# request does not download anything at runtime. Fails the build if OCR cannot load.
RUN python -c "from extractor.ocr import warm_ocr_engine; s = warm_ocr_engine(); print('ocr status:', s); assert s.get('available'), s" \
    && (du -sh /opt/paddlex || true)

# Defaults; Railway's PORT and any service variables override these at runtime.
ENV PORT=5000 \
    WEB_CONCURRENCY=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=600 \
    FORWARDED_ALLOW_IPS=*

EXPOSE 5000

# entrypoint.py exec's gunicorn on 0.0.0.0:$PORT (gunicorn becomes PID 1).
ENTRYPOINT ["python", "/app/entrypoint.py"]
