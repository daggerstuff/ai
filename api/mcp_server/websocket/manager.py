"""
WebSocket Management for MCP Server.

This module handles real-time communication between agents, clients, and the
MCP server using Flask-SocketIO.
"""

import logging
from typing import Any

from flask_socketio import SocketIO, emit, join_room

# Internal imports
from ai.api.mcp_server.config import MCPConfig

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections and real-time events."""

    def __init__(self, socket_io: SocketIO, config: MCPConfig):
        self.socketio = socket_io
        self.config = config
        self.active_clients: dict[str, dict[str, Any]] = {}  # sid -> metadata

    def handle_agent_status_subscribe(self, sid: str, data: dict[str, Any]) -> None:
        """Handle agent status subscription."""
        agent_id = data.get("agent_id")
        if not agent_id:
            logger.warning(f"Client {sid} attempted to subscribe to agent status without agent_id")
            return

        join_room(f"agent_status_{agent_id}", sid=sid)
        logger.info(f"Client {sid} subscribed to agent {agent_id} status updates")

        # Initial status could be fetched and emitted here
        emit("subscription_confirmed", {"target": f"agent_status_{agent_id}"}, to=sid)

    def handle_task_progress_subscribe(self, sid: str, data: dict[str, Any]) -> None:
        """Handle task progress subscription."""
        task_id = data.get("task_id")
        if not task_id:
            logger.warning(f"Client {sid} attempted to subscribe to task progress without task_id")
            return

        join_room(f"task_progress_{task_id}", sid=sid)
        logger.info(f"Client {sid} subscribed to task {task_id} progress updates")

        emit("subscription_confirmed", {"target": f"task_progress_{task_id}"}, to=sid)

    def handle_pipeline_updates_subscribe(self, sid: str, data: dict[str, Any]) -> None:
        """Handle pipeline updates subscription."""
        execution_id = data.get("execution_id")
        if not execution_id:
            logger.warning(f"Client {sid} attempted to subscribe to pipeline updates without execution_id")
            return

        join_room(f"pipeline_updates_{execution_id}", sid=sid)
        logger.info(f"Client {sid} subscribed to pipeline {execution_id} status updates")

        emit("subscription_confirmed", {"target": f"pipeline_updates_{execution_id}"}, to=sid)

    def broadcast_agent_update(self, agent_id: str, update_data: dict[str, Any]) -> None:
        """Broadcast agent status update to registered observers."""
        self.socketio.emit("agent_status_update", update_data, to=f"agent_status_{agent_id}")
        logger.debug(f"Broadcasted status update for agent {agent_id}")

    def broadcast_task_update(self, task_id: str, update_data: dict[str, Any]) -> None:
        """Broadcast task progress update to registered observers."""
        self.socketio.emit("task_progress_update", update_data, to=f"task_progress_{task_id}")
        logger.debug(f"Broadcasted progress update for task {task_id}")

    def broadcast_pipeline_update(self, execution_id: str, update_data: dict[str, Any]) -> None:
        """Broadcast pipeline update to registered observers."""
        self.socketio.emit("pipeline_update", update_data, to=f"pipeline_updates_{execution_id}")
        logger.debug(f"Broadcasted update for pipeline {execution_id}")

    def handle_client_connect(self, sid: str, environ: dict[str, Any]) -> None:
        """Process new WebSocket client connection."""
        self.active_clients[sid] = {
            "connected_at": environ.get("REMOTE_ADDR"),
            "user_agent": environ.get("HTTP_USER_AGENT"),
        }
        logger.info(f"WebSocket client {sid} connected")

    def handle_client_disconnect(self, sid: str) -> None:
        """Process WebSocket client disconnection."""
        if sid in self.active_clients:
            del self.active_clients[sid]
        logger.info(f"WebSocket client {sid} disconnected")
