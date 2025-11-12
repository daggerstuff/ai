"""
Discovery API routes.

This module provides endpoints for source discovery operations.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.common import PaginationParams
from ai.journal_dataset_research.api.models.discovery import (
    DiscoveryInitiateRequest,
    DiscoveryResponse,
    SourceListResponse,
    SourceResponse,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions/{session_id}/discovery", tags=["discovery"])


@router.post("", response_model=DiscoveryResponse)
async def initiate_discovery(
    session_id: str,
    request: DiscoveryInitiateRequest,
    current_user: dict = Depends(require_permission_dependency("discovery:initiate")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> DiscoveryResponse:
    """
    Initiate source discovery.

    Requires: discovery:initiate permission
    """
    try:
        result = service.initiate_discovery(
            session_id=session_id,
            keywords=request.keywords,
            sources=request.sources,
        )

        # Get sources directly from orchestrator (already in correct format)
        orchestrator = service.orchestrator
        state = orchestrator.get_session_state(session_id)
        sources = [
            SourceResponse(
                source_id=source.source_id,
                title=source.title,
                authors=source.authors,
                publication_date=source.publication_date,
                source_type=source.source_type,
                url=source.url,
                doi=source.doi,
                abstract=source.abstract,
                keywords=source.keywords,
                open_access=source.open_access,
                data_availability=source.data_availability,
                discovery_date=source.discovery_date,
                discovery_method=source.discovery_method,
            )
            for source in state.sources
        ]

        return DiscoveryResponse(
            session_id=result["session_id"],
            sources=sources,
            total_sources=result["total_sources"],
            discovery_status="completed",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate discovery: {str(e)}",
        )


@router.get("/sources", response_model=SourceListResponse)
async def list_sources(
    session_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SourceListResponse:
    """
    List discovered sources for a session.

    Requires: discovery:read permission
    """
    try:
        sources = service.get_sources(session_id)
        total = len(sources)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_sources = sources[start:end]

        # Convert to response models
        source_responses = [
            SourceResponse(
                source_id=source.source_id,
                title=source.title,
                authors=source.authors,
                publication_date=source.publication_date,
                source_type=source.source_type,
                url=source.url,
                doi=source.doi,
                abstract=source.abstract,
                keywords=source.keywords,
                open_access=source.open_access,
                data_availability=source.data_availability,
                discovery_date=source.discovery_date,
                discovery_method=source.discovery_method,
            )
            for source in paginated_sources
        ]

        return SourceListResponse.create(
            items=source_responses,
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
            detail=f"Failed to list sources: {str(e)}",
        )


@router.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(
    session_id: str,
    source_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> SourceResponse:
    """
    Get source details.

    Requires: discovery:read permission
    """
    try:
        source = service.get_source(session_id, source_id)

        return SourceResponse(
            source_id=source.source_id,
            title=source.title,
            authors=source.authors,
            publication_date=source.publication_date,
            source_type=source.source_type,
            url=source.url,
            doi=source.doi,
            abstract=source.abstract,
            keywords=source.keywords,
            open_access=source.open_access,
            data_availability=source.data_availability,
            discovery_date=source.discovery_date,
            discovery_method=source.discovery_method,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get source: {str(e)}",
        )
