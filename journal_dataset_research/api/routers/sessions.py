"""
Session management API routes.

This module provides endpoints for managing research sessions.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.common import PaginationParams
from ai.journal_dataset_research.api.models.sessions import (
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
    SessionUpdateRequest,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions")


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SessionListResponse:
    """
    List all research sessions.

    Requires: sessions:read permission
    """
    try:
        sessions = service.list_sessions()
        total = len(sessions)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_sessions = sessions[start:end]

        # Convert to response models
        session_responses = [
            SessionResponse(
                session_id=session.session_id,
                start_date=session.start_date,
                target_sources=session.target_sources,
                search_keywords=session.search_keywords,
                weekly_targets=session.weekly_targets,
                current_phase=session.current_phase,
                progress_metrics=session.progress_metrics,
            )
            for session in paginated_sessions
        ]

        return SessionListResponse.create(
            items=session_responses,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(require_permission_dependency("sessions:create")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SessionResponse:
    """
    Create a new research session.

    Requires: sessions:create permission
    """
    try:
        session = service.create_session(
            target_sources=request.target_sources,
            search_keywords=request.search_keywords,
            weekly_targets=request.weekly_targets,
            session_id=request.session_id,
        )

        return SessionResponse(
            session_id=session.session_id,
            start_date=session.start_date,
            target_sources=session.target_sources,
            search_keywords=session.search_keywords,
            weekly_targets=session.weekly_targets,
            current_phase=session.current_phase,
            progress_metrics=session.progress_metrics,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SessionResponse:
    """
    Get session details.

    Requires: sessions:read permission
    """
    try:
        session = service.get_session(session_id)

        return SessionResponse(
            session_id=session.session_id,
            start_date=session.start_date,
            target_sources=session.target_sources,
            search_keywords=session.search_keywords,
            weekly_targets=session.weekly_targets,
            current_phase=session.current_phase,
            progress_metrics=session.progress_metrics,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}",
        )


@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: SessionUpdateRequest,
    current_user: dict = Depends(require_permission_dependency("sessions:update")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SessionResponse:
    """
    Update session configuration.

    Requires: sessions:update permission
    """
    try:
        session = service.update_session(
            session_id=session_id,
            target_sources=request.target_sources,
            search_keywords=request.search_keywords,
            weekly_targets=request.weekly_targets,
            current_phase=request.current_phase,
        )

        return SessionResponse(
            session_id=session.session_id,
            start_date=session.start_date,
            target_sources=session.target_sources,
            search_keywords=session.search_keywords,
            weekly_targets=session.weekly_targets,
            current_phase=session.current_phase,
            progress_metrics=session.progress_metrics,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}",
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: dict = Depends(require_permission_dependency("sessions:delete")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> None:
    """
    Delete a research session.

    Requires: sessions:delete permission
    """
    try:
        service.delete_session(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )
