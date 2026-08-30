# Dockerfile — M2: containerize the inference service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=/app/models/model.pt \
    PYTHONPATH=/app/src

EXPOSE 8000

# Container-level healthcheck (also used by docker-compose)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
