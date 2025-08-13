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
    OLLAMA_TIMEOUT_S=60

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8000 11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || exit 1

CMD bash -lc '\
  ollama serve & \
  echo "Waiting for Ollama on ${OLLAMA_HOST}..."; \
  for i in $(seq 1 60); do curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null && break || sleep 1; done; \
  echo "Pulling model: ${OLLAMA_BACKEND_MODEL}"; (ollama pull "${OLLAMA_BACKEND_MODEL}" || true); \
  echo "Starting FastAPI on 0.0.0.0:8000..."; \
  exec uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300'
