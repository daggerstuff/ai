"""
WebSocket support for real-time updates.

This module provides WebSocket endpoints for streaming progress updates.
"""

from ai.pipelines.data_processing.journal.api.websocket.manager import ConnectionManager

__all__ = ["ConnectionManager"]
