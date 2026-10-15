"""Container start. Gunicorn must be reachable from the platform proxy.

A bind of 127.0.0.1 accepts connections only from inside the container.
The host proxy connects to the container's network interface, gets
connection refused, and answers 502. This process replaces itself with
Gunicorn on 0.0.0.0:$PORT so that address cannot be dropped by a shell
command or a config-file override.
"""

import os
import sys

port = os.environ.get("PORT", "5000").strip() or "5000"
workers = os.environ.get("WEB_CONCURRENCY", "1").strip() or "1"
# Threads let /health and other requests answer while a long OCR runs; the OCR
# engine pool inside the process is shared and thread-safe.
threads = os.environ.get("GUNICORN_THREADS", "4").strip() or "4"
timeout = os.environ.get("GUNICORN_TIMEOUT", "600").strip() or "600"
gunicorn_path = os.path.join(
    os.path.dirname(sys.executable),
    "gunicorn",
)

# A host GUNICORN_CMD_ARGS like --bind=127.0.0.1:5000 would otherwise
# override the config file and cause the 502 you saw in deploy logs.
os.environ.pop("GUNICORN_CMD_ARGS", None)

print(
    f"titlextractor listening on 0.0.0.0:{port} (workers={workers}, threads={threads})",
    flush=True,
)

os.execvp(
    gunicorn_path,
    [
        gunicorn_path,
        "--bind",
        f"0.0.0.0:{port}",
        "--workers",
        workers,
        "--worker-class",
        "gthread",
        "--threads",
        threads,
        "--timeout",
        timeout,
        "--graceful-timeout",
        "30",
        "--forwarded-allow-ips",
        "*",
        "--access-logfile",
        "-",
        "--error-logfile",
        "-",
        "app:app",
    ],
)

sys.exit(1)
