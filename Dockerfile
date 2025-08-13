FROM ollama/ollama:latest

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

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
  CMD curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || exit 1

EXPOSE 8000 11434

# CMD runs both Ollama and FastAPI in one shell
CMD bash -lc '\
  ollama serve & \
  echo "Waiting for Ollama on ${OLLAMA_HOST}..."; \
  for i in $(seq 1 60); do \
    curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null && break || sleep 1; \
  done; \
  echo "Pulling model: ${OLLAMA_BACKEND_MODEL}"; \
  (ollama pull "${OLLAMA_BACKEND_MODEL}" || true); \
  echo "Starting FastAPI on 0.0.0.0:8000..."; \
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300'
