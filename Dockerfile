FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJEXA_DISPLAY_NAME=projexa \
    OLLAMA_BACKEND_MODEL=llama3.2:3b \
    OLLAMA_HOST=http://127.0.0.1:11434

# System deps
RUN apt-get update && apt-get install -y \
    curl ca-certificates python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install Ollama daemon + CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Your FastAPI app (the file that contains `app = FastAPI(...)`)
# Rename this if your filename is different.
COPY app.py .

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000 11434
CMD ["/start.sh"]
