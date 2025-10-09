"""Minimal error handler for TechDeck integration used during tests.

This provides a lightweight TechDeckErrorHandler class so the Flask
app factory and tests can import the symbol without pulling in heavy
dependencies. The real implementation lives elsewhere; this shim keeps
test collection working.
"""
from typing import Any, Dict, Optional
from flask import Flask


class TechDeckErrorHandler:
    """Simple error handler placeholder used for pytest collection.

    The real error handler would register handlers on the Flask app and
    perform structured JSON responses. For tests we only need the class
    to exist and accept a config object.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def register(self, app: Flask) -> None:
        """Register error handlers on the Flask app (no-op placeholder)."""
        # No-op for test-time import; real implementation registers handlers
        # with app.errorhandler(...)
        return None
