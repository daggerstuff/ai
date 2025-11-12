"""
Evaluation API routes.

This module provides endpoints for dataset evaluation operations.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_command_handler_service,
    get_current_user,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.common import PaginationParams
from ai.journal_dataset_research.api.models.evaluation import (
    EvaluationInitiateRequest,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationUpdateRequest,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

router = APIRouter(prefix="/sessions/{session_id}/evaluation", tags=["evaluation"])


@router.post("", response_model=EvaluationResponse)
async def initiate_evaluation(
    session_id: str,
    request: EvaluationInitiateRequest,
    current_user: dict = Depends(require_permission_dependency("evaluation:read")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> EvaluationResponse:
    """
    Initiate evaluation for sources.

    Requires: evaluation:read permission
    """
    try:
        result = service.initiate_evaluation(
            session_id=session_id,
            source_ids=request.source_ids,
        )

        if not result["evaluations"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sources to evaluate or no evaluations created",
            )

        # Get the first evaluation from orchestrator
        orchestrator = service.orchestrator
        state = orchestrator.get_session_state(session_id)
        if not state.evaluations:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluations created but not found in state",
            )

        # Return the most recent evaluation
        evaluation = state.evaluations[-1]
        return EvaluationResponse(
            evaluation_id=f"eval_{evaluation.source_id}",
            source_id=evaluation.source_id,
            therapeutic_relevance=evaluation.therapeutic_relevance,
            data_structure_quality=evaluation.data_structure_quality,
            training_integration=evaluation.training_integration,
            ethical_accessibility=evaluation.ethical_accessibility,
            overall_score=evaluation.overall_score,
            priority_tier=evaluation.priority_tier,
            evaluation_date=evaluation.evaluation_date,
            evaluator=evaluation.evaluator,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate evaluation: {str(e)}",
        )


@router.get("", response_model=EvaluationListResponse)
async def list_evaluations(
    session_id: str,
    pagination: PaginationParams = Depends(),
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> EvaluationListResponse:
    """
    List evaluations for a session.

    Requires: evaluation:read permission
    """
    try:
        evaluations = service.get_evaluations(session_id)
        total = len(evaluations)

        # Paginate
        start = (pagination.page - 1) * pagination.page_size
        end = start + pagination.page_size
        paginated_evaluations = evaluations[start:end]

        # Convert to response models
        evaluation_responses = [
            EvaluationResponse(
                evaluation_id=f"eval_{evaluation.source_id}",
                source_id=evaluation.source_id,
                therapeutic_relevance=evaluation.therapeutic_relevance,
                data_structure_quality=evaluation.data_structure_quality,
                training_integration=evaluation.training_integration,
                ethical_accessibility=evaluation.ethical_accessibility,
                overall_score=evaluation.overall_score,
                priority_tier=evaluation.priority_tier,
                evaluation_date=evaluation.evaluation_date,
                evaluator=evaluation.evaluator,
            )
            for evaluation in paginated_evaluations
        ]

        return EvaluationListResponse.create(
            items=evaluation_responses,
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
            detail=f"Failed to list evaluations: {str(e)}",
        )


@router.get("/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    session_id: str,
    evaluation_id: str,
    current_user: Optional[dict] = Depends(get_current_user),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> EvaluationResponse:
    """
    Get evaluation details.

    Requires: evaluation:read permission
    """
    try:
        evaluation = service.get_evaluation(session_id, evaluation_id)

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            source_id=evaluation.source_id,
            therapeutic_relevance=evaluation.therapeutic_relevance,
            data_structure_quality=evaluation.data_structure_quality,
            training_integration=evaluation.training_integration,
            ethical_accessibility=evaluation.ethical_accessibility,
            overall_score=evaluation.overall_score,
            priority_tier=evaluation.priority_tier,
            evaluation_date=evaluation.evaluation_date,
            evaluator=evaluation.evaluator,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation: {str(e)}",
        )


@router.put("/{evaluation_id}", response_model=EvaluationResponse)
async def update_evaluation(
    session_id: str,
    evaluation_id: str,
    request: EvaluationUpdateRequest,
    current_user: dict = Depends(require_permission_dependency("evaluation:update")),
    service: CommandHandlerService = Depends(get_command_handler_service),
) -> EvaluationResponse:
    """
    Update evaluation scores.

    Requires: evaluation:update permission
    """
    try:
        evaluation = service.update_evaluation(
            session_id=session_id,
            evaluation_id=evaluation_id,
            therapeutic_relevance=request.therapeutic_relevance,
            data_structure_quality=request.data_structure_quality,
            training_integration=request.training_integration,
            ethical_accessibility=request.ethical_accessibility,
            priority_tier=request.priority_tier,
        )

        return EvaluationResponse(
            evaluation_id=evaluation_id,
            source_id=evaluation.source_id,
            therapeutic_relevance=evaluation.therapeutic_relevance,
            data_structure_quality=evaluation.data_structure_quality,
            training_integration=evaluation.training_integration,
            ethical_accessibility=evaluation.ethical_accessibility,
            overall_score=evaluation.overall_score,
            priority_tier=evaluation.priority_tier,
            evaluation_date=evaluation.evaluation_date,
            evaluator=evaluation.evaluator,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update evaluation: {str(e)}",
        )
