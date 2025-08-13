# ---------- Projexa: single-container Ollama + FastAPI ----------
FROM ollama/ollama:latest

# Minimal system deps + venv (avoids PEP 668)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code (expects main.py with `app = FastAPI(...)`)
COPY . /app

# Healthcheck for local Ollama daemon
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
  CMD curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || exit 1

# No EXPOSE — service is internal unless you map a port at run-time

# Reset parent ENTRYPOINT (the base image sets it to `ollama`)
ENTRYPOINT []

# Start Ollama, wait for readiness, pull model, run FastAPI on 8000
CMD ["bash","-lc","\
ollama serve & \
echo 'Waiting for Ollama on http://127.0.0.1:11434...'; \
for i in $(seq 1 60); do \
  curl -fsS http://127.0.0.1:11434/api/tags >/dev/null && break || sleep 1; \
done; \
echo 'Pulling model: llama3.2:3b'; \
(ollama pull llama3.2:3b || true); \
echo 'Starting FastAPI on 0.0.0.0:8000 (internal only)...'; \
exec uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 600 \
"]
# ---------- End ----------
