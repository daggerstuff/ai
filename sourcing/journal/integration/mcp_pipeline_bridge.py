"""
MCP Pipeline Bridge

Bridge between MCP server tools and training pipeline orchestrator.
Enables automatic integration of acquired datasets into the training pipeline.
"""

import logging
from typing import Any, Callable, Dict, Optional

from ai.sourcing.journal.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    IntegrationPlan,
)

logger = logging.getLogger(__name__)


class MCPPipelineBridge:
    """
    Bridge between MCP server tools and training pipeline orchestrator.
    
    This bridge:
    1. Receives notifications when datasets are acquired via MCP tools
    2. Triggers automatic integration into training pipeline
    3. Provides progress updates back to MCP resources
    4. Handles error recovery and retry logic
    """

    def __init__(
        self,
        pipeline_orchestrator: Optional[Any] = None,
        integration_service: Optional[Any] = None,
        auto_integrate: bool = True,
    ):
        """
        Initialize the MCP pipeline bridge.
        
        Args:
            pipeline_orchestrator: PipelineOrchestrator instance (optional, lazy-loaded)
            integration_service: PipelineIntegrationService instance (optional, lazy-loaded)
            auto_integrate: Whether to automatically integrate acquired datasets
        """
        self.pipeline_orchestrator = pipeline_orchestrator
        self.integration_service = integration_service
        self.auto_integrate = auto_integrate
        
        # Track integration operations
        self.integration_operations: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks for progress updates
        self.progress_callbacks: list[Callable[[str, Dict[str, Any]], None]] = []
        
        logger.info(
            f"Initialized MCPPipelineBridge (auto_integrate={auto_integrate})"
        )

    def register_pipeline_orchestrator(self, orchestrator: Any) -> None:
        """
        Register pipeline orchestrator instance.
        
        Args:
            orchestrator: PipelineOrchestrator instance
        """
        self.pipeline_orchestrator = orchestrator
        logger.info("Pipeline orchestrator registered")

    def register_integration_service(self, service: Any) -> None:
        """
        Register integration service instance.
        
        Args:
            service: PipelineIntegrationService instance
        """
        self.integration_service = service
        logger.info("Integration service registered")

    def on_dataset_acquired(
        self,
        dataset: AcquiredDataset,
        evaluation: Optional[DatasetEvaluation] = None,
        integration_plan: Optional[IntegrationPlan] = None,
    ) -> Dict[str, Any]:
        """
        Handle notification when a dataset is acquired via MCP tool.
        
        This method:
        1. Creates integration plan if not provided
        2. Triggers automatic integration if enabled
        3. Registers dataset with pipeline orchestrator
        4. Returns integration status
        
        Args:
            dataset: Acquired dataset
            evaluation: Optional evaluation scores for quality filtering
            integration_plan: Optional pre-created integration plan
            
        Returns:
            Dictionary with integration status and results
        """
        logger.info(f"Handling acquired dataset: {dataset.source_id}")
        
        # Initialize operation tracking
        self.integration_operations[dataset.source_id] = {
            "status": "pending",
            "dataset": dataset,
            "evaluation": evaluation,
            "integration_plan": integration_plan,
            "pipeline_registered": False,
            "error": None,
        }
        
        if not self.auto_integrate:
            logger.info(f"Auto-integration disabled, skipping integration for {dataset.source_id}")
            return {
                "source_id": dataset.source_id,
                "status": "pending",
                "auto_integrated": False,
                "message": "Auto-integration disabled",
            }
        
        try:
            # Ensure we have required components
            if not self.pipeline_orchestrator:
                logger.warning(
                    "Pipeline orchestrator not registered, "
                    "cannot integrate dataset automatically"
                )
                return {
                    "source_id": dataset.source_id,
                    "status": "pending",
                    "auto_integrated": False,
                    "message": "Pipeline orchestrator not available",
                }
            
            # Create integration plan if not provided
            if not integration_plan:
                if not self.integration_service:
                    logger.warning(
                        "Integration service not registered, "
                        "cannot create integration plan"
                    )
                    return {
                        "source_id": dataset.source_id,
                        "status": "pending",
                        "auto_integrated": False,
                        "message": "Integration service not available",
                    }
                
                logger.info(f"Creating integration plan for {dataset.source_id}")
                integration_plan = self.integration_service.create_integration_plan(
                    dataset=dataset,
                    target_format="chatml",
                )
                self.integration_operations[dataset.source_id]["integration_plan"] = (
                    integration_plan
                )
            
            # Get evaluation score if available
            evaluation_score = None
            if evaluation:
                evaluation_score = evaluation.overall_score
            
            # Register with pipeline orchestrator
            logger.info(f"Registering dataset {dataset.source_id} with pipeline orchestrator")
            integration_result = self.pipeline_orchestrator.register_journal_research_dataset(
                dataset=dataset,
                integration_plan=integration_plan,
                evaluation_score=evaluation_score,
            )
            
            # Update operation tracking
            self.integration_operations[dataset.source_id].update({
                "status": "completed" if integration_result.get("success") else "failed",
                "pipeline_registered": integration_result.get("success", False),
                "integration_result": integration_result,
                "error": integration_result.get("error"),
            })
            
            # Notify progress callbacks
            self._notify_progress(dataset.source_id, {
                "status": "completed" if integration_result.get("success") else "failed",
                "integration_result": integration_result,
            })
            
            logger.info(
                f"Dataset {dataset.source_id} integration "
                f"{'completed' if integration_result.get('success') else 'failed'}"
            )
            
            return {
                "source_id": dataset.source_id,
                "status": "completed" if integration_result.get("success") else "failed",
                "auto_integrated": True,
                "integration_result": integration_result,
            }
            
        except Exception as e:
            logger.error(
                f"Error integrating dataset {dataset.source_id}: {e}",
                exc_info=True,
            )
            self.integration_operations[dataset.source_id].update({
                "status": "failed",
                "error": str(e),
            })
            
            # Notify progress callbacks
            self._notify_progress(dataset.source_id, {
                "status": "failed",
                "error": str(e),
            })
            
            return {
                "source_id": dataset.source_id,
                "status": "failed",
                "auto_integrated": False,
                "error": str(e),
            }

    def get_integration_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get integration status for a dataset.
        
        Args:
            source_id: Source ID of the dataset
            
        Returns:
            Integration status dictionary or None if not found
        """
        return self.integration_operations.get(source_id)

    def add_progress_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Add callback for progress updates.
        
        Args:
            callback: Function that receives (source_id, progress_data)
        """
        self.progress_callbacks.append(callback)
        logger.debug("Progress callback added")

    def _notify_progress(self, source_id: str, progress_data: Dict[str, Any]) -> None:
        """
        Notify all progress callbacks.
        
        Args:
            source_id: Source ID
            progress_data: Progress data dictionary
        """
        for callback in self.progress_callbacks:
            try:
                callback(source_id, progress_data)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

    def list_integrated_datasets(self) -> list[str]:
        """
        List all integrated dataset source IDs.
        
        Returns:
            List of source IDs
        """
        return [
            source_id
            for source_id, op in self.integration_operations.items()
            if op.get("status") == "completed"
        ]


