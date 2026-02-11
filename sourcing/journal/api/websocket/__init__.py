"""
WebSocket support for real-time updates.

This module provides WebSocket endpoints for streaming progress updates.
"""

from ai.sourcing.journal.api.websocket.manager import ConnectionManager
from ai.sourcing.journal.api.websocket.routes import router as websocket_router

__all__ = ["ConnectionManager", "websocket_router"]

