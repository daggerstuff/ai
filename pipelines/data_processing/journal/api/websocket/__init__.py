"""
WebSocket support for real-time updates.

This module provides WebSocket endpoints for streaming progress updates.
"""

from ai.pipelines.data_processing.journal.api.websocket.manager import ConnectionManager
from ai.pipelines.data_processing.journal.api.websocket.routes import router as websocket_router

__all__ = ["ConnectionManager", "websocket_router"]
