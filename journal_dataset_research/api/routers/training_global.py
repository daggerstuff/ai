"""
Global Training Pipeline API Routes

Provides global endpoints for training pipeline status (not session-scoped).
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from ai.journal_dataset_research.api.dependencies import (
    get_training_pipeline_service,
    require_permission_dependency,
)
from ai.journal_dataset_research.api.models.responses import ErrorResponse
from ai.journal_dataset_research.api.services.training_pipeline_service import (
    TrainingPipelineService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["training"])


@router.get(
    "/pipeline-status",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Training pipeline status"},
    },
)
async def get_pipeline_status(
    current_user: dict = Depends(require_permission_dependency("training:read")),
    training_service: TrainingPipelineService = Depends(get_training_pipeline_service),
) -> Dict[str, Any]:
    """
    Get overall training pipeline status.

    Returns:
        Current status of the training pipeline orchestrator
    """
    try:
        status_result = await training_service.get_pipeline_status()
        return status_result

    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pipeline status: {str(e)}",
        )

