"""
WebSocket support for real-time updates.

This module provides WebSocket endpoints for streaming progress updates.
"""

from ai.journal_dataset_research.api.websocket.manager import ConnectionManager
from ai.journal_dataset_research.api.websocket.routes import router as websocket_router

__all__ = ["ConnectionManager", "websocket_router"]

