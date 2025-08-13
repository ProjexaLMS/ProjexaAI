# ---------- Single Dockerfile (no external scripts) ----------
FROM ollama/ollama:latest

# ---- App/config envs ----
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJEXA_DISPLAY_NAME=projexa \
    OLLAMA_BACKEND_MODEL=llama3.2:3b \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    PROJEXA_MAX_BYTES=524288 \
    PROJEXA_MAX_WORDS_OUT=500 \
    PROJEXA_STREAM_TIMEOUT_S=120 \
    OLLAMA_TIMEOUT_S=60 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# ---- System deps + venv support ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# ---- Create isolated venv (bypasses PEP 668) ----
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ---- App setup ----
WORKDIR /app

# Install Python deps first to leverage layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your app (expects main.py exposing "app = FastAPI(...)")
COPY . /app

# ---- Healthcheck for Ollama daemon ----
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
  CMD curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || exit 1

# ---- Expose ports ----
EXPOSE 8000 11434

# ---- Embedded entrypoint (no separate file) ----
# Starts Ollama, waits up to 60s, pulls model (idempotent), then launches uvicorn as PID 1.
RUN set -eux; \
  cat > /usr/local/bin/projexa_entrypoint.sh <<'SH'; \
  #!/usr/bin/env bash
  set -euo pipefail

  # Start Ollama in background
  ollama serve &

  # Wait for Ollama to be ready
  echo "Waiting for Ollama on ${OLLAMA_HOST}..."
  for i in {1..60}; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null; then
      echo "Ollama is up."
      break
    fi
    sleep 1
  done

  # Pull the model (safe if already present)
  echo "Pulling model: ${OLLAMA_BACKEND_MODEL}"
  ollama pull "${OLLAMA_BACKEND_MODEL}" || true

  # Start FastAPI (make uvicorn PID 1)
  echo "Starting FastAPI on 0.0.0.0:8000..."
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300
  SH
RUN chmod +x /usr/local/bin/projexa_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/projexa_entrypoint.sh"]
# ---------- End Dockerfile ----------
