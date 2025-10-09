# Multi-stage Dockerfile for Pixelated Empathy AI
# Optimized for production deployment with security and performance

# Build arguments
ARG PYTHON_VERSION=3.11
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_BRANCH
ARG VERSION

# Base image for Python dependencies
FROM python:${PYTHON_VERSION}-slim as python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster Python package management
RUN pip install uv

# Development stage
FROM python-base as development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv sync --dev

# Copy source code
COPY . .

# Production dependencies stage
FROM python-base as deps

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev

# Production stage
FROM python:${PYTHON_VERSION}-slim as production

# Build metadata
LABEL org.opencontainers.image.title="Pixelated Empathy AI" \
      org.opencontainers.image.description="AI-powered empathetic conversation system" \
      org.opencontainers.image.vendor="Pixelated Team" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/pixelated/empathy-ai"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    BUILD_DATE="${BUILD_DATE}" \
    GIT_COMMIT="${GIT_COMMIT}" \
    GIT_BRANCH="${GIT_BRANCH}" \
    VERSION="${VERSION}"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

# Create application directory
WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=deps --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development override
FROM development as dev
USER root
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*
USER appuser
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
