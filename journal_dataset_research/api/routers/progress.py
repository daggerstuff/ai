"""
Progress API routes.

This module provides endpoints for progress tracking.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    get_optional_user,
)
from ai.journal_dataset_research.api.models.progress import (
    ProgressMetricsResponse,
    ProgressResponse,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.journal_dataset_research.api.websocket.manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}/progress", tags=["progress"])


@router.get("", response_model=ProgressResponse)
async def get_progress(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> ProgressResponse:
    """
    Get progress metrics for a session.

    Requires: sessions:read permission
    """
    try:
        progress_data = service.get_progress(session_id)

        return ProgressResponse(
            session_id=progress_data["session_id"],
            current_phase=progress_data["current_phase"],
            progress_metrics=progress_data["progress_metrics"],
            weekly_targets=progress_data["weekly_targets"],
            progress_percentage=progress_data["progress_percentage"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get progress: {str(e)}",
        )


@router.get("/metrics", response_model=ProgressMetricsResponse)
async def get_progress_metrics(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> ProgressMetricsResponse:
    """
    Get detailed progress metrics for a session.

    Requires: sessions:read permission
    """
    try:
        metrics_data = service.get_progress_metrics(session_id)

        return ProgressMetricsResponse(
            session_id=metrics_data["session_id"],
            sources_identified=metrics_data["sources_identified"],
            datasets_evaluated=metrics_data["datasets_evaluated"],
            datasets_acquired=metrics_data["datasets_acquired"],
            integration_plans_created=metrics_data["integration_plans_created"],
            last_updated=metrics_data["last_updated"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get progress metrics: {str(e)}",
        )


async def generate_progress_events(
    session_id: str,
    service: CommandHandlerService,
    request: Request,
) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events for progress updates.

    This generator yields SSE-formatted messages containing progress updates
    for the specified session. It polls the service for updates and broadcasts
    them to connected clients.
    """
    last_progress: Optional[dict] = None
    poll_interval = 2.0  # Poll every 2 seconds

    try:
        # Send initial connection message
        yield f"event: connected\ndata: {json.dumps({'session_id': session_id, 'status': 'connected'})}\n\n"

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"SSE client disconnected for session {session_id}")
                break

            try:
                # Get current progress
                progress_data = service.get_progress(session_id)

                # Only send update if progress has changed
                if progress_data != last_progress:
                    # Format as progress update message
                    message = {
                        "type": "progress_update",
                        "sessionId": session_id,
                        "data": {
                            "phase": progress_data.get("current_phase", "unknown"),
                            "progress": progress_data.get("progress_percentage", 0),
                            "metrics": progress_data.get("progress_metrics", {}),
                            "message": f"Phase: {progress_data.get('current_phase', 'unknown')}",
                        },
                    }

                    yield f"data: {json.dumps(message)}\n\n"
                    last_progress = progress_data

                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"

            except ValueError as e:
                # Session not found
                error_message = {
                    "type": "error",
                    "sessionId": session_id,
                    "error": str(e),
                }
                yield f"event: error\ndata: {json.dumps(error_message)}\n\n"
                break
            except Exception as e:
                logger.error(f"Error generating progress events: {e}")
                error_message = {
                    "type": "error",
                    "sessionId": session_id,
                    "error": f"Failed to get progress: {str(e)}",
                }
                yield f"event: error\ndata: {json.dumps(error_message)}\n\n"
                await asyncio.sleep(poll_interval)
                continue

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for session {session_id}")
    except Exception as e:
        logger.error(f"SSE stream error for session {session_id}: {e}")
        error_message = {
            "type": "error",
            "sessionId": session_id,
            "error": f"Stream error: {str(e)}",
        }
        yield f"event: error\ndata: {json.dumps(error_message)}\n\n"


@router.get("/events")
async def stream_progress_events(
    session_id: str,
    request: Request,
    token: Optional[str] = Query(None, description="JWT token for authentication"),
    current_user: Optional[dict] = Depends(get_optional_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> StreamingResponse:
    """
    Stream progress updates as Server-Sent Events (SSE).

    This endpoint provides real-time progress updates for a research session
    using Server-Sent Events. The stream will send progress updates whenever
    the session progress changes.

    Query Parameters:
    - token: Optional JWT token for authentication (can be passed via query string)

    Returns:
    - StreamingResponse with text/event-stream content type

    Example SSE messages:
    ```
    event: connected
    data: {"session_id": "abc123", "status": "connected"}

    data: {"type": "progress_update", "sessionId": "abc123", "data": {...}}

    : heartbeat
    ```
    """
    # Verify session exists
    try:
        service.get_session(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    # Authenticate if token provided in query params
    if token:
        try:
            from ai.journal_dataset_research.api.auth.jwt import get_user_from_token

            user = get_user_from_token(token)
            logger.info(f"SSE authenticated for user {user.get('user_id')}")
        except Exception as e:
            logger.warning(f"SSE authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
            )

    # Create SSE stream
    return StreamingResponse(
        generate_progress_events(session_id, service, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )
