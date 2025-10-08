# Dockerfile for Pixel Voice API and MCP Server using uv
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIXEL_VOICE_ENV=production
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies including uv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy project configuration first for better caching
COPY pyproject.toml .
COPY requirements_api.txt .
COPY requirements/pixel_voice_pipeline.txt ./requirements/

# Create virtual environment and install dependencies using uv
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies using uv
RUN uv pip install -e .
RUN uv pip install -r requirements/pixel_voice_pipeline.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/voice_raw data/voice_segments data/voice_transcripts \
    data/voice_transcripts_filtered data/voice_features data/dialogue_pairs \
    data/therapeutic_pairs data/voice_consistency data/voice_optimized \
    logs reports

# Expose ports
EXPOSE 8000 8001

# Create startup script using uv
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    uv run python start_api.py\n\
elif [ "$1" = "mcp" ]; then\n\
    uv run python start_mcp.py\n\
else\n\
    echo "Usage: docker run <image> [api|mcp]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
