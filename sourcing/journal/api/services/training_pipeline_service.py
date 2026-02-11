"""
Training Pipeline Service

Service layer for connecting journal research system to training pipeline orchestrator.
"""

import logging
from typing import Any, Dict, Optional

from ai.sourcing.journal.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    IntegrationPlan,
)

logger = logging.getLogger(__name__)

# Optional import for PipelineOrchestrator
try:
    from ai.pipelines.orchestrator.orchestration.pipeline_orchestrator import (
        PipelineOrchestrator,
    )
    from ai.sourcing.journal.models.dataset_models import (
        AcquiredDataset as AcquiredDatasetModel,
        DatasetEvaluation as DatasetEvaluationModel,
        IntegrationPlan as IntegrationPlanModel,
    )
    PIPELINE_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    PIPELINE_ORCHESTRATOR_AVAILABLE = False
    PipelineOrchestrator = None  # type: ignore
    logger.warning("PipelineOrchestrator not available (optional dependency)")


class TrainingPipelineService:
    """
    Service for integrating journal research datasets into training pipeline.

    This service acts as a bridge between the journal research API and the
    training pipeline orchestrator, similar to MCPPipelineBridge but for web API.
    """

    def __init__(self):
        """Initialize the training pipeline service."""
        self.pipeline_orchestrator: Optional[PipelineOrchestrator] = None
        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Initialize pipeline orchestrator if available."""
        if not PIPELINE_ORCHESTRATOR_AVAILABLE:
            logger.warning("PipelineOrchestrator not available, training integration disabled")
            return

        try:
            from ai.pipelines.orchestrator.orchestration.pipeline_orchestrator import (
                PipelineConfig,
            )
            from pathlib import Path

            config = PipelineConfig(
                output_directory=Path("data/processed"),
                quality_threshold=0.7,
            )
            self.pipeline_orchestrator = PipelineOrchestrator(config=config)
            logger.info("Training pipeline orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline orchestrator: {e}", exc_info=True)
            self.pipeline_orchestrator = None

    async def integrate_dataset(
        self,
        session_id: str,
        source_id: str,
        acquired_dataset: Dict[str, Any],
        evaluation: Optional[Dict[str, Any]] = None,
        integration_plan: Optional[Dict[str, Any]] = None,
        auto_integrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Integrate a dataset into the training pipeline.

        Args:
            session_id: Research session ID
            source_id: Dataset source ID
            acquired_dataset: Acquired dataset dictionary
            evaluation: Optional evaluation dictionary
            integration_plan: Optional integration plan dictionary
            auto_integrate: Whether to automatically trigger integration

        Returns:
            Integration result dictionary
        """
        if not self.pipeline_orchestrator:
            return {
                "success": False,
                "error": "Training pipeline orchestrator not available",
                "source_id": source_id,
            }

        if not auto_integrate:
            return {
                "success": False,
                "error": "Auto-integration disabled",
                "source_id": source_id,
            }

        try:
            # Convert dictionaries to model objects
            acquired_dataset_obj = self._dict_to_acquired_dataset(acquired_dataset)
            evaluation_obj = self._dict_to_evaluation(evaluation) if evaluation else None
            integration_plan_obj = (
                self._dict_to_integration_plan(integration_plan) if integration_plan else None
            )

            # Register with training pipeline
            # Extract evaluation details safely
            evaluation_score = None
            evaluation_details = None
            if evaluation_obj:
                if hasattr(evaluation_obj, 'overall_score'):
                    evaluation_score = evaluation_obj.overall_score
                elif isinstance(evaluation_obj, dict):
                    evaluation_score = evaluation_obj.get('overall_score')

                if hasattr(evaluation_obj, '__dict__'):
                    evaluation_details = evaluation_obj.__dict__
                elif isinstance(evaluation_obj, dict):
                    evaluation_details = evaluation_obj

            result = self.pipeline_orchestrator.register_journal_research_dataset(
                dataset=acquired_dataset_obj,
                integration_plan=integration_plan_obj,
                evaluation_score=evaluation_score,
                evaluation_details=evaluation_details,
            )

            logger.info(
                f"Dataset {source_id} from session {session_id} integrated: {result.get('success', False)}"
            )

            return {
                "success": result.get("success", False),
                "source_id": source_id,
                "session_id": session_id,
                "result": result,
            }

        except Exception as e:
            logger.error(
                f"Error integrating dataset {source_id} from session {session_id}: {e}",
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
                "source_id": source_id,
                "session_id": session_id,
            }

    async def get_session_status(
        self, session_id: str, session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get training pipeline status for all datasets in a session.

        Args:
            session_id: Research session ID
            session_state: Session state dictionary

        Returns:
            Status dictionary with integration status for each dataset
        """
        if not self.pipeline_orchestrator:
            return {
                "session_id": session_id,
                "pipeline_available": False,
                "datasets": [],
            }

        acquired_datasets = session_state.get("acquired_datasets", [])
        statuses = []

        for dataset in acquired_datasets:
            source_id = dataset.get("source_id")
            if not source_id:
                continue

            # Check integration status
            integration_status = self.pipeline_orchestrator.get_journal_research_integration_status(
                source_id
            )

            statuses.append({
                "source_id": source_id,
                "integrated": integration_status is not None,
                "integration_status": integration_status,
            })

        return {
            "session_id": session_id,
            "pipeline_available": True,
            "total_datasets": len(acquired_datasets),
            "integrated_datasets": sum(1 for s in statuses if s["integrated"]),
            "datasets": statuses,
        }

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall training pipeline status.

        Returns:
            Pipeline status dictionary
        """
        if not self.pipeline_orchestrator:
            return {
                "available": False,
                "message": "Training pipeline orchestrator not available",
            }

        try:
            # Get metrics
            metrics = self.pipeline_orchestrator.metrics
            journal_datasets = self.pipeline_orchestrator.journal_research_datasets

            return {
                "available": True,
                "total_datasets": metrics.total_datasets,
                "completed_datasets": metrics.completed_datasets,
                "failed_datasets": metrics.failed_datasets,
                "total_conversations": metrics.total_conversations,
                "accepted_conversations": metrics.accepted_conversations,
                "rejected_conversations": metrics.rejected_conversations,
                "journal_research_datasets": len(journal_datasets),
                "current_stage": metrics.current_stage.value if metrics.current_stage else None,
            }
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}", exc_info=True)
            return {
                "available": True,
                "error": str(e),
            }

    def _dict_to_acquired_dataset(self, data: Dict[str, Any]) -> Any:
        """Convert dictionary to AcquiredDataset model."""
        if not PIPELINE_ORCHESTRATOR_AVAILABLE:
            # Create a simple object that can be used
            from types import SimpleNamespace
            return SimpleNamespace(**data)

        try:
            # Handle datetime strings if present
            if "acquisition_date" in data and isinstance(data["acquisition_date"], str):
                from datetime import datetime
                data["acquisition_date"] = datetime.fromisoformat(data["acquisition_date"])

            return AcquiredDatasetModel(**data)
        except Exception as e:
            logger.warning(f"Failed to convert to AcquiredDataset model: {e}, using dict")
            from types import SimpleNamespace
            return SimpleNamespace(**data)

    def _dict_to_evaluation(self, data: Dict[str, Any]) -> Any:
        """Convert dictionary to DatasetEvaluation model."""
        if not PIPELINE_ORCHESTRATOR_AVAILABLE:
            from types import SimpleNamespace
            return SimpleNamespace(**data)

        try:
            # Handle datetime strings if present
            if "evaluation_date" in data and isinstance(data["evaluation_date"], str):
                from datetime import datetime
                data["evaluation_date"] = datetime.fromisoformat(data["evaluation_date"])

            return DatasetEvaluationModel(**data)
        except Exception as e:
            logger.warning(f"Failed to convert to DatasetEvaluation model: {e}, using dict")
            from types import SimpleNamespace
            return SimpleNamespace(**data)

    def _dict_to_integration_plan(self, data: Dict[str, Any]) -> Any:
        """Convert dictionary to IntegrationPlan model."""
        if not PIPELINE_ORCHESTRATOR_AVAILABLE:
            from types import SimpleNamespace
            return SimpleNamespace(**data)

        try:
            # Handle datetime strings if present
            if "created_date" in data and isinstance(data["created_date"], str):
                from datetime import datetime
                data["created_date"] = datetime.fromisoformat(data["created_date"])

            return IntegrationPlanModel(**data)
        except Exception as e:
            logger.warning(f"Failed to convert to IntegrationPlan model: {e}, using dict")
            from types import SimpleNamespace
            return SimpleNamespace(**data)

