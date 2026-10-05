FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py gunicorn.conf.py entrypoint.py ./
COPY extractor ./extractor
COPY templates ./templates

# Local default. A hosted platform injects PORT at runtime and that value wins.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FORWARDED_ALLOW_IPS=*

EXPOSE 5000

# python then exec's gunicorn, so gunicorn is PID 1 and the bind is explicit.
ENTRYPOINT ["python", "/app/entrypoint.py"]
