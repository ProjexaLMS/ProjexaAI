#!/usr/bin/env bash
set -euo pipefail

# Start Ollama in the background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start on ${OLLAMA_HOST}..."
for i in {1..60}; do
  if curl -s "${OLLAMA_HOST}/api/tags" >/dev/null; then
    echo "Ollama is up."
    break
  fi
  sleep 1
done

# Pull the backend model (idempotent if already present)
echo "Pulling model: ${OLLAMA_BACKEND_MODEL}"
ollama pull "${OLLAMA_BACKEND_MODEL}" || true

# Run FastAPI (uvicorn)
echo "Starting FastAPI on 0.0.0.0:8000..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
