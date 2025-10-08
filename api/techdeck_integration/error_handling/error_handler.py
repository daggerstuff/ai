"""Minimal error handler shim for test collection.

Provides ErrorHandler with register and handle_exception used during imports.
"""
from typing import Any
from .custom_errors import TechDeckBaseError


class ErrorHandler:
    def __init__(self, app=None):
        self.app = app

    def register(self, app: Any) -> None:
        """Register error handlers on the Flask app (no-op shim)."""
        self.app = app

    def handle_exception(self, exc: Exception) -> dict:
        """Convert exception to dict for API responses (simple shim)."""
        if isinstance(exc, TechDeckBaseError):
            return exc.to_dict()
        return {
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(exc)
            }
        }
