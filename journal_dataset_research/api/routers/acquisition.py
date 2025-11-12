"""
Acquisition API routes.

This module provides endpoints for dataset acquisition operations.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.acquisition import (
    AcquisitionInitiateRequest,
    AcquisitionListResponse,
    AcquisitionResponse,
    AcquisitionUpdateRequest,
)
from ai.journal_dataset_research.api.models.common import PaginationParams
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions/{session_id}/acquisition", tags=["acquisition"])


@router.post("", response_model=AcquisitionResponse)
async def initiate_acquisition(
    session_id: str,
    request: AcquisitionInitiateRequest,
    current_user: dict = Depends(require_permission_dependency("acquisition:read")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> AcquisitionResponse:
    """
    Initiate acquisition for sources.

    Requires: acquisition:read permission
    """
    try:
        result = service.initiate_acquisition(
            session_id=session_id,
            source_ids=request.source_ids,
        )

        if not result["acquired"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sources to acquire or no acquisitions completed",
            )

        # Get the most recent acquisition
        orchestrator = service.orchestrator
        state = orchestrator.get_session_state(session_id)
        if not state.acquired_datasets:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Acquisitions completed but not found in state",
            )

        acquisition = state.acquired_datasets[-1]
        return AcquisitionResponse(
            acquisition_id=f"acq_{acquisition.source_id}",
            source_id=acquisition.source_id,
            status="completed" if acquisition.storage_path else "pending",
            download_progress=100.0 if acquisition.storage_path else 0.0,
            file_path=acquisition.storage_path,
            file_size=acquisition.file_size_mb,
            acquired_date=acquisition.acquisition_date,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate acquisition: {str(e)}",
        )


@router.get("", response_model=AcquisitionListResponse)
async def list_acquisitions(
    session_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> AcquisitionListResponse:
    """
    List acquisitions for a session.

    Requires: acquisition:read permission
    """
    try:
        acquisitions = service.get_acquisitions(session_id)
        total = len(acquisitions)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_acquisitions = acquisitions[start:end]

        # Convert to response models
        acquisition_responses = [
            AcquisitionResponse(
                acquisition_id=f"acq_{acquisition.source_id}",
                source_id=acquisition.source_id,
                status="completed" if acquisition.storage_path else "pending",
                download_progress=100.0 if acquisition.storage_path else 0.0,
                file_path=acquisition.storage_path,
                file_size=acquisition.file_size_mb,
                acquired_date=acquisition.acquisition_date,
            )
            for acquisition in paginated_acquisitions
        ]

        return AcquisitionListResponse.create(
            items=acquisition_responses,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list acquisitions: {str(e)}",
        )


@router.get("/{acquisition_id}", response_model=AcquisitionResponse)
async def get_acquisition(
    session_id: str,
    acquisition_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> AcquisitionResponse:
    """
    Get acquisition details.

    Requires: acquisition:read permission
    """
    try:
        acquisition = service.get_acquisition(session_id, acquisition_id)

        return AcquisitionResponse(
            acquisition_id=acquisition_id,
            source_id=acquisition.source_id,
            status="completed" if acquisition.storage_path else "pending",
            download_progress=100.0 if acquisition.storage_path else 0.0,
            file_path=acquisition.storage_path,
            file_size=acquisition.file_size_mb,
            acquired_date=acquisition.acquisition_date,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get acquisition: {str(e)}",
        )


@router.put("/{acquisition_id}", response_model=AcquisitionResponse)
async def update_acquisition(
    session_id: str,
    acquisition_id: str,
    request: AcquisitionUpdateRequest,
    current_user: dict = Depends(require_permission_dependency("acquisition:approve")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> AcquisitionResponse:
    """
    Update acquisition status.

    Requires: acquisition:approve permission
    """
    try:
        acquisition = service.update_acquisition(
            session_id=session_id,
            acquisition_id=acquisition_id,
            status=request.status,
        )

        return AcquisitionResponse(
            acquisition_id=acquisition_id,
            source_id=acquisition.source_id,
            status=request.status,
            download_progress=100.0 if acquisition.storage_path else 0.0,
            file_path=acquisition.storage_path,
            file_size=acquisition.file_size_mb,
            acquired_date=acquisition.acquisition_date,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update acquisition: {str(e)}",
        )
