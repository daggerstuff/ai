"""
WebSocket routes for real-time updates.

Security: JWT auth mandatory, rate limiting, origin validation, audit logging.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from ai.pipelines.data_processing.journal.api.auth.jwt import get_user_from_token
from ai.pipelines.data_processing.journal.api.config import get_settings
from ai.pipelines.data_processing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.pipelines.data_processing.journal.api.websocket.manager import manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# WebSocket-specific rate limit state (user_id -> [timestamps])
_ws_connect_timestamps: dict[str, list[float]] = {}
WS_RATE_LIMIT_PER_MINUTE = 30


def _check_ws_rate_limit(user_id: str) -> bool:
    """Check if user has exceeded WebSocket connection rate limit."""
    import time

    now = time.time()
    window = 60.0  # 1 minute

    if user_id not in _ws_connect_timestamps:
        _ws_connect_timestamps[user_id] = []

    # Prune old timestamps
    _ws_connect_timestamps[user_id] = [ts for ts in _ws_connect_timestamps[user_id] if now - ts < window]

    if len(_ws_connect_timestamps[user_id]) >= WS_RATE_LIMIT_PER_MINUTE:
        return False

    _ws_connect_timestamps[user_id].append(now)
    return True


async def _authenticate_ws(websocket: WebSocket) -> dict | None:
    """Authenticate WebSocket connection. Returns user dict or None if failed."""
    token = websocket.query_params.get("token")
    if not token:
        logger.warning("WebSocket connection rejected: no token provided")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None

    try:
        user = get_user_from_token(token)
        user_id = user.get("user_id")
        if not user_id:
            logger.warning("WebSocket connection rejected: no user_id in token")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        if not _check_ws_rate_limit(user_id):
            logger.warning(f"WebSocket connection rejected: rate limit exceeded for user {user_id}")
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return None

        logger.info(f"WebSocket authenticated for user {user_id}")
        return user
    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None


@router.websocket("/ws/progress/{session_id}")
async def websocket_progress(
    websocket: WebSocket,
    session_id: str,
) -> None:
    """WebSocket endpoint for real-time progress updates."""
    user = await _authenticate_ws(websocket)
    if user is None:
        return

    origin = websocket.headers.get("origin")
    accepted = await manager.connect(
        websocket,
        session_id,
        user_id=user.get("user_id"),
        origin=origin,
        allowed_origins=settings.cors_origins,
    )
    if not accepted:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    try:
        # Send initial progress state
        service = CommandHandlerService()
        try:
            progress_data = service.get_progress(session_id)
            await manager.send_personal_message(
                {
                    "type": "progress_update",
                    "session_id": session_id,
                    "timestamp": datetime.now(UTC).isoformat(),
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
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": f"Failed to load progress: {e!s}",
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
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                            websocket,
                        )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")
            except TimeoutError:
                # Send ping to keep connection alive
                await manager.send_personal_message(
                    {
                        "type": "ping",
                        "timestamp": datetime.now(UTC).isoformat(),
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
    """WebSocket endpoint for polling progress updates."""
    interval = int(websocket.query_params.get("interval", "5"))

    user = await _authenticate_ws(websocket)
    if user is None:
        return

    origin = websocket.headers.get("origin")
    accepted = await manager.connect(
        websocket,
        session_id,
        user_id=user.get("user_id"),
        origin=origin,
        allowed_origins=settings.cors_origins,
    )
    if not accepted:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

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
                            "timestamp": datetime.now(UTC).isoformat(),
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
                except TimeoutError:
                    pass
                except json.JSONDecodeError:
                    pass

            except Exception as e:
                logger.error(f"Error polling progress: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "session_id": session_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "message": f"Error polling progress: {e!s}",
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
