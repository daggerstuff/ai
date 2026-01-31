"""
Training Pipeline Integration API Routes

Provides endpoints for integrating journal research datasets into the training pipeline.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ai.sourcing.journal.api.dependencies import (
    get_command_handler_service,
    get_training_pipeline_service,
    require_permission_dependency,
)
from ai.sourcing.journal.api.models.responses import ErrorResponse
from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.api.services.training_pipeline_service import (
    TrainingPipelineService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}/training", tags=["training"])


@router.post(
    "/integrate/{source_id}",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Dataset successfully integrated into training pipeline"},
        404: {"model": ErrorResponse, "description": "Dataset or session not found"},
        500: {"model": ErrorResponse, "description": "Integration failed"},
    },
)
async def integrate_dataset(
    session_id: str,
    source_id: str,
    auto_integrate: bool = True,
    current_user: dict = Depends(require_permission_dependency("training:write")),
    service: CommandHandlerService = Depends(get_command_handler_service),
    training_service: TrainingPipelineService = Depends(get_training_pipeline_service),
) -> Dict[str, Any]:
    """
    Integrate an acquired dataset into the training pipeline.

    This endpoint:
    1. Retrieves the acquired dataset and evaluation from the session
    2. Gets the integration plan
    3. Registers the dataset with the training pipeline orchestrator
    4. Returns integration status

    Args:
        session_id: Research session ID
        source_id: Dataset source ID to integrate
        auto_integrate: Whether to automatically trigger integration
        current_user: Authenticated user
        service: Command handler service
        training_service: Training pipeline service

    Returns:
        Integration result with status and details
    """
    try:
        # Get session and state
        orchestrator = service.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Find the acquired dataset
        acquired_dataset = None
        evaluation = None
        integration_plan = None

        if "acquired_datasets" in state:
            for dataset in state["acquired_datasets"]:
                if dataset.get("source_id") == source_id:
                    acquired_dataset = dataset
                    break

        if not acquired_dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Acquired dataset {source_id} not found in session {session_id}",
            )

        # Get evaluation if available
        if "evaluations" in state:
            for eval_data in state["evaluations"]:
                if eval_data.get("source_id") == source_id:
                    evaluation = eval_data
                    break

        # Get integration plan if available
        if "integration_plans" in state:
            for plan_data in state["integration_plans"]:
                if plan_data.get("source_id") == source_id:
                    integration_plan = plan_data
                    break

        # Integrate with training pipeline
        result = await training_service.integrate_dataset(
            session_id=session_id,
            source_id=source_id,
            acquired_dataset=acquired_dataset,
            evaluation=evaluation,
            integration_plan=integration_plan,
            auto_integrate=auto_integrate,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error integrating dataset {source_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to integrate dataset: {str(e)}",
        )


@router.get(
    "/status",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Training pipeline status for session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_training_status(
    session_id: str,
    current_user: dict = Depends(require_permission_dependency("training:read")),
    service: CommandHandlerService = Depends(get_command_handler_service),
    training_service: TrainingPipelineService = Depends(get_training_pipeline_service),
) -> Dict[str, Any]:
    """
    Get training pipeline integration status for a research session.

    Returns:
        Status of all datasets in the session and their training pipeline integration
    """
    try:
        # Get session and state
        orchestrator = service.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Get training status for all datasets in session
        status_result = await training_service.get_session_status(session_id, state)

        return status_result

    except Exception as e:
        logger.error(f"Error getting training status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}",
        )


@router.post(
    "/integrate-all",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "All datasets integrated"},
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def integrate_all_datasets(
    session_id: str,
    auto_integrate: bool = True,
    current_user: dict = Depends(require_permission_dependency("training:write")),
    service: CommandHandlerService = Depends(get_command_handler_service),
    training_service: TrainingPipelineService = Depends(get_training_pipeline_service),
) -> Dict[str, Any]:
    """
    Integrate all acquired datasets from a session into the training pipeline.

    Returns:
        Results for all dataset integrations
    """
    try:
        # Get session and state
        orchestrator = service.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Get all acquired datasets
        acquired_datasets = state.get("acquired_datasets", [])

        if not acquired_datasets:
            return {
                "success": True,
                "message": "No acquired datasets to integrate",
                "integrated": 0,
                "failed": 0,
                "results": [],
            }

        # Integrate all datasets
        results = []
        integrated_count = 0
        failed_count = 0

        for dataset in acquired_datasets:
            source_id = dataset.get("source_id")
            if not source_id:
                continue

            try:
                # Get evaluation and integration plan
                evaluation = None
                integration_plan = None

                if "evaluations" in state:
                    for eval_data in state["evaluations"]:
                        if eval_data.get("source_id") == source_id:
                            evaluation = eval_data
                            break

                if "integration_plans" in state:
                    for plan_data in state["integration_plans"]:
                        if plan_data.get("source_id") == source_id:
                            integration_plan = plan_data
                            break

                # Integrate
                result = await training_service.integrate_dataset(
                    session_id=session_id,
                    source_id=source_id,
                    acquired_dataset=dataset,
                    evaluation=evaluation,
                    integration_plan=integration_plan,
                    auto_integrate=auto_integrate,
                )

                results.append({
                    "source_id": source_id,
                    "success": result.get("success", False),
                    "result": result,
                })

                if result.get("success"):
                    integrated_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error integrating dataset {source_id}: {e}", exc_info=True)
                results.append({
                    "source_id": source_id,
                    "success": False,
                    "error": str(e),
                })
                failed_count += 1

        return {
            "success": True,
            "integrated": integrated_count,
            "failed": failed_count,
            "total": len(acquired_datasets),
            "results": results,
        }

    except Exception as e:
        logger.error(f"Error integrating all datasets for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to integrate datasets: {str(e)}",
        )


@router.get(
    "/pipeline/status",
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

