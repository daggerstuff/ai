"""
WebSocket connection manager.

This module manages WebSocket connections and broadcasts progress updates.
"""

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a WebSocket connection for a session."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_personal_message(self, message: dict, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")

    async def broadcast_to_session(self, session_id: str, message: dict) -> None:
        """Broadcast a message to all connections for a session."""
        async with self._lock:
            connections = self.active_connections.get(session_id, set()).copy()

        disconnected = set()
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(connection)

        # Clean up disconnected connections
        if disconnected:
            async with self._lock:
                if session_id in self.active_connections:
                    self.active_connections[session_id] -= disconnected
                    if not self.active_connections[session_id]:
                        del self.active_connections[session_id]

    async def get_connection_count(self, session_id: str) -> int:
        """Get the number of active connections for a session."""
        async with self._lock:
            return len(self.active_connections.get(session_id, set()))

    def get_all_session_ids(self) -> Set[str]:
        """Get all session IDs with active connections."""
        return set(self.active_connections.keys())


# Global connection manager instance
manager = ConnectionManager()

