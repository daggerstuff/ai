"""
WebSocket connection manager.

This module manages WebSocket connections and broadcasts progress updates.
Includes security enforcement: max connections, origin validation, audit logging.
"""

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Maximum connections per session (prevents resource exhaustion)
MAX_CONNECTIONS_PER_SESSION = 50
# Maximum total connections across all sessions
MAX_TOTAL_CONNECTIONS = 500


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        # Track user_id per websocket for rate limiting and audit
        self._ws_users: dict[int, dict[str, Any]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str | None = None,
        origin: str | None = None,
        allowed_origins: list[str] | None = None,
    ) -> bool:
        """
        Accept a WebSocket connection for a session.

        Returns True if connection was accepted, False if rejected.
        """
        # Origin validation
        if allowed_origins and origin:
            origin_host = origin.split("://")[-1].split("/")[0]
            if not any(origin_host == o.split("://")[-1].split("/")[0] for o in allowed_origins):
                logger.warning(f"WebSocket rejected: origin {origin} not in allowed_origins")
                return False

        # Check total connection limit
        total = sum(len(conns) for conns in self.active_connections.values())
        if total >= MAX_TOTAL_CONNECTIONS:
            logger.warning(f"WebSocket rejected: total connection limit {MAX_TOTAL_CONNECTIONS} reached")
            return False

        # Check per-session connection limit
        async with self._lock:
            session_conns = self.active_connections.get(session_id, set())
            if len(session_conns) >= MAX_CONNECTIONS_PER_SESSION:
                logger.warning(
                    f"WebSocket rejected: session {session_id} connection limit {MAX_CONNECTIONS_PER_SESSION} reached"
                )
                return False

        await websocket.accept()

        async with self._lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
            # Track user for this websocket
            self._ws_users[id(websocket)] = {
                "user_id": user_id or "anonymous",
                "session_id": session_id,
            }

        logger.info(f"WebSocket connected for session {session_id} user={user_id}")
        return True

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            self._ws_users.pop(id(websocket), None)
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

    def get_all_session_ids(self) -> set[str]:
        """Get all session IDs with active connections."""
        return set(self.active_connections.keys())


# Global connection manager instance
manager = ConnectionManager()
