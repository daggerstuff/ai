"""
Integration API routes.

This module provides endpoints for integration planning operations.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.sourcing.journal.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.sourcing.journal.api.models.common import PaginationParams
from ai.sourcing.journal.api.models.integration import (
    IntegrationInitiateRequest,
    IntegrationPlanListResponse,
    IntegrationPlanResponse,
)
from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions/{session_id}/integration", tags=["integration"])


@router.post("", response_model=IntegrationPlanResponse)
async def initiate_integration(
    session_id: str,
    request: IntegrationInitiateRequest,
    current_user: dict = Depends(require_permission_dependency("integration:read")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> IntegrationPlanResponse:
    """
    Initiate integration planning.

    Requires: integration:read permission
    """
    try:
        result = service.initiate_integration(
            session_id=session_id,
            source_ids=request.source_ids,
            target_format=request.target_format,
        )

        if not result["plans"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No datasets to integrate or no plans created",
            )

        # Get the most recent integration plan
        orchestrator = service.orchestrator
        state = orchestrator.get_session_state(session_id)
        if not state.integration_plans:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Integration plans created but not found in state",
            )

        plan = state.integration_plans[-1]
        return IntegrationPlanResponse(
            plan_id=f"plan_{plan.source_id}",
            source_id=plan.source_id,
            complexity=plan.complexity,
            target_format=plan.dataset_format,
            required_transformations=plan.required_transformations,
            estimated_effort_hours=plan.estimated_effort_hours,
            schema_mapping=plan.schema_mapping,
            created_date=plan.created_date,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate integration: {str(e)}",
        )


@router.get("", response_model=IntegrationPlanListResponse)
async def list_integration_plans(
    session_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> IntegrationPlanListResponse:
    """
    List integration plans for a session.

    Requires: integration:read permission
    """
    try:
        plans = service.get_integration_plans(session_id)
        total = len(plans)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_plans = plans[start:end]

        # Convert to response models
        plan_responses = [
            IntegrationPlanResponse(
                plan_id=f"plan_{plan.source_id}",
                source_id=plan.source_id,
                complexity=plan.complexity,
                target_format=plan.dataset_format,
                required_transformations=plan.required_transformations,
                estimated_effort_hours=plan.estimated_effort_hours,
                schema_mapping=plan.schema_mapping,
                created_date=plan.created_date,
            )
            for plan in paginated_plans
        ]

        return IntegrationPlanListResponse.create(
            items=plan_responses,
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
            detail=f"Failed to list integration plans: {str(e)}",
        )


@router.get("/{plan_id}", response_model=IntegrationPlanResponse)
async def get_integration_plan(
    session_id: str,
    plan_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> IntegrationPlanResponse:
    """
    Get integration plan details.

    Requires: integration:read permission
    """
    try:
        plan = service.get_integration_plan(session_id, plan_id)

        return IntegrationPlanResponse(
            plan_id=plan_id,
            source_id=plan.source_id,
            complexity=plan.complexity,
            target_format=plan.dataset_format,
            required_transformations=plan.required_transformations,
            estimated_effort_hours=plan.estimated_effort_hours,
            schema_mapping=plan.schema_mapping,
            created_date=plan.created_date,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get integration plan: {str(e)}",
        )
