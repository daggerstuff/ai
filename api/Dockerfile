FROM python:3.13-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy and install Python dependencies
COPY ai/api/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache --system -r /app/requirements.txt

# Copy the entire ai/ directory so the training_corpus/pal_framework imports resolve
COPY ai/ /app/ai/

# Create a non-root user
RUN groupadd -g 42420 appuser && \
    useradd -u 42420 -g appuser -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose the PAL API port
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')" || exit 1

# Use --app-dir to guarantee module resolution regardless of namespace packages
CMD ["uvicorn", "pal_inference_service:app", "--host", "0.0.0.0", "--port", "8010", "--app-dir", "/app/ai/api"]
