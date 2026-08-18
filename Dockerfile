# Multi-stage Dockerfile for Pixelated Empathy AI
# Optimized for production deployment with security and performance

# Build arguments
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_BRANCH
ARG VERSION

# Base image for Python dependencies
FROM ${BASE_IMAGE} AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# NVIDIA image is Ubuntu-based, so apt-get is correct
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user if it doesn't exist
RUN id -u ubuntu >/dev/null 2>&1 || (groupadd -r ubuntu && useradd -m -r -g ubuntu -s /bin/bash ubuntu)


# Install uv for faster Python package management
RUN pip install uv

# Development stage
FROM python-base AS development

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
# system base image has torch, but uv might try to reinstall if not careful.
# For now, we rely on uv sync to ensure consistent environment in .venv
RUN uv sync --dev

# Copy source code
COPY . .

# Production dependencies stage
FROM python-base AS deps

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --no-dev

# Production stage
FROM ${BASE_IMAGE} AS production

# Redeclare build arguments for this stage
ARG BUILD_DATE
ARG GIT_COMMIT
ARG GIT_BRANCH
ARG VERSION

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
    PYTHONPATH="/app" \
    BUILD_DATE="${BUILD_DATE}" \
    GIT_COMMIT="${GIT_COMMIT}" \
    GIT_BRANCH="${GIT_BRANCH}" \
    VERSION="${VERSION}"

# Install runtime dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Container runtime runs as a fixed non-root UID for deployment consistency.
RUN groupadd -g 42420 appuser && \
    useradd -u 42420 -g 42420 -m -s /bin/bash appuser

# Create application directory
WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy application code
COPY --chown=ubuntu:ubuntu . .

# Switch to non-root runtime user
USER 42420

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command (using uv for all Python commands)
CMD ["uv", "run", "python", "-m", "ai.api.main"]

# Development override
FROM development AS dev
USER root
RUN apt-get update && apt-get install --no-install-recommends -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*
USER ubuntu
CMD ["uv", "run", "python", "-m", "ai.api.main"]
