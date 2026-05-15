"""Sentry initialization helpers for Python services."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import sentry_sdk
except ImportError:
    sentry_sdk = None


_SENTRY_INITIALIZED = False


def resolve_sentry_dsn() -> str | None:
    """Resolve Sentry DSN from environment for Python runtimes."""
    return (
        os.getenv("SENTRY_DSN")
        or os.getenv("PUBLIC_SENTRY_DSN")
        or os.getenv("SENTRY_PUBLIC_DSN")
        or os.getenv("VITE_SENTRY_DSN")
    )


def _resolve_release() -> str | None:
    return (
        os.getenv("SENTRY_RELEASE")
        or os.getenv("PUBLIC_SENTRY_RELEASE")
        or os.getenv("GIT_COMMIT")
        or os.getenv("GITHUB_SHA")
        or os.getenv("CI_COMMIT_SHA")
    )


def _resolve_environment() -> str:
    return os.getenv("SENTRY_ENVIRONMENT", os.getenv("NODE_ENV", "production"))


def _resolve_debug_enabled() -> bool:
    return os.getenv("SENTRY_DEBUG", "0") == "1"


def initialize_sentry_logging(service_name: str | None = None) -> bool:
    """Initialize Sentry with logs forwarding enabled.

    Returns True when initialization succeeds; False when DSN is missing or SDK is
    unavailable.
    """
    if sentry_sdk is None:
        return False

    dsn = resolve_sentry_dsn()
    if not dsn:
        return False

    global _SENTRY_INITIALIZED
    if _SENTRY_INITIALIZED:
        return True

    sentry_sdk.init(
        dsn=dsn,
        enable_logs=True,
        traces_sample_rate=0.0,
        environment=_resolve_environment(),
        release=_resolve_release(),
        debug=_resolve_debug_enabled(),
        send_default_pii=True,
        server_name=service_name,
    )

    _SENTRY_INITIALIZED = True
    return True


def send_sample_log_messages() -> dict[str, Any]:
    """Emit sample logs for quick verification."""
    if sentry_sdk is None:
        return {"status": "skipped", "reason": "sentry-sdk not installed"}

    sentry_sdk.logger.info("This is an info log message")
    sentry_sdk.logger.warning("This is a warning message")
    sentry_sdk.logger.error("This is an error message")
    return {"status": "sent", "method": "sentry_sdk.logger"}


def attach_python_logging() -> None:
    logging.getLogger().setLevel(logging.INFO)
