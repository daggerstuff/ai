"""
Progress streaming system for MCP Server.

This module provides progress update broadcasting, progress resource updates,
and progress subscription mechanism for async operations.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode

logger = logging.getLogger(__name__)


class ProgressStatus(str, Enum):
    """Progress update status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update message."""

    operation_id: str
    session_id: str
    status: ProgressStatus
    progress_percent: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress update to dictionary."""
        return {
            "operation_id": self.operation_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ProgressStreamer:
    """
    Progress streaming system for broadcasting progress updates.

    This class manages progress subscriptions and broadcasts progress updates
    to subscribed listeners. It also updates progress resources automatically.
    """

    def __init__(self) -> None:
        """Initialize progress streamer."""
        # Subscriptions: operation_id -> set of callbacks
        self._subscriptions: Dict[str, Set[Callable[[ProgressUpdate], None]]] = {}
        # Session subscriptions: session_id -> set of callbacks
        self._session_subscriptions: Dict[str, Set[Callable[[ProgressUpdate], None]]] = {}
        # Operation status tracking: operation_id -> ProgressStatus
        self._operation_status: Dict[str, ProgressStatus] = {}
        # Progress history: operation_id -> list of ProgressUpdate
        self._progress_history: Dict[str, List[ProgressUpdate]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        operation_id: str,
        callback: Callable[[ProgressUpdate], None],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Subscribe to progress updates for an operation.

        Args:
            operation_id: Operation ID to subscribe to
            callback: Callback function to call on progress updates
            session_id: Optional session ID for session-wide subscriptions
        """
        async with self._lock:
            if operation_id not in self._subscriptions:
                self._subscriptions[operation_id] = set()
            self._subscriptions[operation_id].add(callback)

            if session_id:
                if session_id not in self._session_subscriptions:
                    self._session_subscriptions[session_id] = set()
                self._session_subscriptions[session_id].add(callback)

            logger.debug(f"Subscribed to progress updates for operation {operation_id}")

    async def unsubscribe(
        self,
        operation_id: str,
        callback: Callable[[ProgressUpdate], None],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Unsubscribe from progress updates for an operation.

        Args:
            operation_id: Operation ID to unsubscribe from
            callback: Callback function to remove
            session_id: Optional session ID for session-wide subscriptions
        """
        async with self._lock:
            if operation_id in self._subscriptions:
                self._subscriptions[operation_id].discard(callback)
                if not self._subscriptions[operation_id]:
                    del self._subscriptions[operation_id]

            if session_id and session_id in self._session_subscriptions:
                self._session_subscriptions[session_id].discard(callback)
                if not self._session_subscriptions[session_id]:
                    del self._session_subscriptions[session_id]

            logger.debug(f"Unsubscribed from progress updates for operation {operation_id}")

    async def broadcast_update(self, update: ProgressUpdate) -> None:
        """
        Broadcast progress update to all subscribers.

        Args:
            update: Progress update to broadcast
        """
        async with self._lock:
            # Update operation status
            self._operation_status[update.operation_id] = update.status

            # Add to progress history
            if update.operation_id not in self._progress_history:
                self._progress_history[update.operation_id] = []
            self._progress_history[update.operation_id].append(update)

            # Keep only last 100 updates per operation
            if len(self._progress_history[update.operation_id]) > 100:
                self._progress_history[update.operation_id] = self._progress_history[
                    update.operation_id
                ][-100:]

        # Broadcast to operation-specific subscribers
        if update.operation_id in self._subscriptions:
            callbacks = self._subscriptions[update.operation_id].copy()
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.warning(
                        f"Error calling progress callback for operation {update.operation_id}: {e}"
                    )

        # Broadcast to session-wide subscribers
        if update.session_id in self._session_subscriptions:
            callbacks = self._session_subscriptions[update.session_id].copy()
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.warning(
                        f"Error calling progress callback for session {update.session_id}: {e}"
                    )

        logger.debug(
            f"Broadcast progress update for operation {update.operation_id}: "
            f"{update.status.value} - {update.progress_percent}%"
        )

    async def get_operation_status(self, operation_id: str) -> Optional[ProgressStatus]:
        """
        Get current status of an operation.

        Args:
            operation_id: Operation ID

        Returns:
            Current status or None if operation not found
        """
        async with self._lock:
            return self._operation_status.get(operation_id)

    async def get_progress_history(
        self, operation_id: str, limit: Optional[int] = None
    ) -> List[ProgressUpdate]:
        """
        Get progress history for an operation.

        Args:
            operation_id: Operation ID
            limit: Optional limit on number of updates to return

        Returns:
            List of progress updates
        """
        async with self._lock:
            history = self._progress_history.get(operation_id, [])
            if limit:
                return history[-limit:]
            return history.copy()

    async def get_session_progress(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[ProgressUpdate]:
        """
        Get all progress updates for a session.

        Args:
            session_id: Session ID
            limit: Optional limit on number of updates to return

        Returns:
            List of progress updates for the session
        """
        async with self._lock:
            session_updates: List[ProgressUpdate] = []
            for operation_id, history in self._progress_history.items():
                for update in history:
                    if update.session_id == session_id:
                        session_updates.append(update)

            # Sort by timestamp
            session_updates.sort(key=lambda u: u.timestamp)

            if limit:
                return session_updates[-limit:]
            return session_updates

    async def clear_operation(self, operation_id: str) -> None:
        """
        Clear operation data (status and history).

        Args:
            operation_id: Operation ID to clear
        """
        async with self._lock:
            self._operation_status.pop(operation_id, None)
            self._progress_history.pop(operation_id, None)
            self._subscriptions.pop(operation_id, None)

        logger.debug(f"Cleared operation data for {operation_id}")

    async def clear_session(self, session_id: str) -> None:
        """
        Clear all operations for a session.

        Args:
            session_id: Session ID to clear
        """
        async with self._lock:
            # Find all operations for this session
            operations_to_clear = []
            for operation_id, history in self._progress_history.items():
                if history and history[0].session_id == session_id:
                    operations_to_clear.append(operation_id)

            # Clear each operation
            for operation_id in operations_to_clear:
                self._operation_status.pop(operation_id, None)
                self._progress_history.pop(operation_id, None)
                self._subscriptions.pop(operation_id, None)

            # Clear session subscriptions
            self._session_subscriptions.pop(session_id, None)

        logger.debug(f"Cleared all operations for session {session_id}")

    def get_subscription_count(self, operation_id: Optional[str] = None) -> int:
        """
        Get number of active subscriptions.

        Args:
            operation_id: Optional operation ID to count subscriptions for

        Returns:
            Number of subscriptions
        """
        if operation_id:
            return len(self._subscriptions.get(operation_id, set()))
        return sum(len(subs) for subs in self._subscriptions.values())

