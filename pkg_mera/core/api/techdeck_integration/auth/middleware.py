"""Compatibility wrapper for legacy imports in ``ai.pkg_mera.core.api.techdeck_integration.auth.middleware``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _load_core_middleware():
    from ai.api.techdeck_integration.auth.middleware import JWTAuthMiddleware as CoreJWTAuthMiddleware

    return CoreJWTAuthMiddleware


def _require_dependency() -> Any:
    try:
        return _load_core_middleware()
    except Exception as exc:
        raise RuntimeError(
            "ai.api.techdeck_integration.auth.middleware is unavailable because optional dependencies are missing."
        ) from exc


class JWTAuthMiddleware:
    """Compatibility wrapper that delegates to the primary middleware implementation."""

    def __init__(self, app: Callable | None = None, config: Any | None = None):
        middleware_cls = _require_dependency()
        self._impl = middleware_cls(app, config)

    def __call__(self, environ: dict[str, Any], start_response: Callable):
        return self._impl(environ, start_response)


__all__ = ["JWTAuthMiddleware"]
