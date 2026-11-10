# Pinned by digest (3.12-slim as of 2026-11-09) so the build is reproducible;
# bump the digest when the tag is intentionally moved.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution (pinned by digest for reproducibility)
COPY --from=ghcr.io/astral-sh/uv@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /bin/uv

# Run as a non-root user with a writable home (uv cache) and app directory.
RUN groupadd --system appuser \
    && useradd --system --create-home --gid appuser appuser

WORKDIR /app
RUN chown appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY --chown=appuser:appuser ./ ./

CMD ["python", "-m", "pipelines.cli"]
