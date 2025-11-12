"""
Progress API routes.

This module provides endpoints for progress tracking.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
)
from ai.journal_dataset_research.api.models.progress import (
    ProgressMetricsResponse,
    ProgressResponse,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.journal_dataset_research.api.websocket.manager import manager

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
