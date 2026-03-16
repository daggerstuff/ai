"""
WebSocket routes for real-time updates.

This module provides WebSocket endpoints for streaming progress updates.
"""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from ai.sourcing.journal.api.dependencies import (
    get_command_handler_service,
    get_current_user,
)
from ai.sourcing.journal.api.websocket.manager import manager
from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/progress/{session_id}")
async def websocket_progress(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """
    WebSocket endpoint for real-time progress updates.

    Connects to a session and streams progress updates as they occur.
    Query parameters (optional):
    - token: JWT token for authentication (can be passed via query string)
    """
    # Optional: Check for token in query params
    token = websocket.query_params.get("token")
    if token:
        try:
            from ai.sourcing.journal.api.auth.jwt import get_user_from_token

            user = get_user_from_token(token)
            logger.info(f"WebSocket authenticated for user {user.get('user_id')}")
        except Exception as e:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            logger.warning(f"WebSocket authentication failed: {e}")
            return

    # Connect to session
    await manager.connect(websocket, session_id)

    try:
        # Send initial progress state
        service = CommandHandlerService()
        try:
            progress_data = service.get_progress(session_id)
            await manager.send_personal_message(
                {
                    "type": "progress_update",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": progress_data,
                },
                websocket,
            )
        except Exception as e:
            logger.error(f"Error sending initial progress: {e}")
            await manager.send_personal_message(
                {
                    "type": "error",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Failed to load progress: {str(e)}",
                },
                websocket,
            )

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for ping or close message
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await manager.send_personal_message(
                            {
                                "type": "pong",
                                "timestamp": datetime.now().isoformat(),
                            },
                            websocket,
                        )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_personal_message(
                    {
                        "type": "ping",
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await manager.disconnect(websocket, session_id)


@router.websocket("/ws/progress/{session_id}/poll")
async def websocket_progress_poll(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """
    WebSocket endpoint for polling progress updates.

    Polls progress at regular intervals and sends updates.
    Query parameters (optional):
    - interval: Polling interval in seconds (default: 5)
    - token: JWT token for authentication
    """
    # Get interval from query params
    interval = int(websocket.query_params.get("interval", "5"))

    # Optional: Check for token in query params
    token = websocket.query_params.get("token")
    if token:
        try:
            from ai.sourcing.journal.api.auth.jwt import get_user_from_token

            user = get_user_from_token(token)
            logger.info(f"WebSocket authenticated for user {user.get('user_id')}")
        except Exception as e:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            logger.warning(f"WebSocket authentication failed: {e}")
            return

    # Connect to session
    await manager.connect(websocket, session_id)

    try:
        service = CommandHandlerService()
        last_metrics = None

        while True:
            try:
                # Get current progress
                progress_data = service.get_progress_metrics(session_id)

                # Check if metrics have changed
                current_metrics = {
                    "sources_identified": progress_data["sources_identified"],
                    "datasets_evaluated": progress_data["datasets_evaluated"],
                    "datasets_acquired": progress_data["datasets_acquired"],
                    "integration_plans_created": progress_data["integration_plans_created"],
                }

                if current_metrics != last_metrics:
                    await manager.send_personal_message(
                        {
                            "type": "progress_update",
                            "session_id": session_id,
                            "timestamp": datetime.now().isoformat(),
                            "data": progress_data,
                        },
                        websocket,
                    )
                    last_metrics = current_metrics

                # Wait for next poll
                await asyncio.sleep(interval)

                # Check for close message
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    if message.get("type") == "close":
                        break
                except asyncio.TimeoutError:
                    pass
                except json.JSONDecodeError:
                    pass

            except Exception as e:
                logger.error(f"Error polling progress: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Error polling progress: {str(e)}",
                    },
                    websocket,
                )
                await asyncio.sleep(interval)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await manager.disconnect(websocket, session_id)

