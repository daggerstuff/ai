"""Lightweight WebSocketManager shim for test collection.

Provides a minimal WebSocketManager that matches the public API used
by the app factory (init and attach to app). The full implementation
is in progress; here we provide enough for imports during pytest
collection.
"""
from typing import Any, Optional
from flask import Flask


class WebSocketManager:
    def __init__(self, config: Optional[Any] = None):
        self.config = config or {}

    def init_app(self, app: Flask) -> None:
        # No-op initialization for tests
        app.websocket_manager = self
        return None
