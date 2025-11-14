"""
Batch Converter

Batch conversion workflow for multiple journal datasets.
Converts multiple acquired datasets to training format in batch operations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.integration.pipeline_integration_service import (
    PipelineIntegrationService,
)
from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    IntegrationPlan,
)

logger = logging.getLogger(__name__)


class BatchConverter:
    """
    Batch converter for multiple journal research datasets.
    
    Provides:
    1. Batch conversion of multiple datasets
    2. Progress tracking for batch operations
    3. Error recovery and retry logic
    4. Summary reporting
    """

    def __init__(
        self,
        integration_service: Optional[PipelineIntegrationService] = None,
        output_directory: Path = Path("data/processed/journal_research/batch"),
        max_concurrent: int = 3,
    ):
        """
        Initialize batch converter.
        
        Args:
            integration_service: PipelineIntegrationService instance (creates new if None)
            output_directory: Directory for batch conversion outputs
            max_concurrent: Maximum concurrent conversions
        """
        self.integration_service = (
            integration_service or PipelineIntegrationService()
        )
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        
        logger.info(
            f"Initialized BatchConverter: output={output_directory}, "
            f"max_concurrent={max_concurrent}"
        )

    def convert_batch(
        self,
        datasets: List[AcquiredDataset],
        integration_plans: Dict[str, IntegrationPlan],
        existing_dataset_path: Optional[Path] = None,
        target_format: str = "chatml",
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Convert multiple datasets in batch.
        
        Args:
            datasets: List of acquired datasets to convert
            integration_plans: Dictionary mapping source_id to IntegrationPlan
            existing_dataset_path: Optional path to existing dataset for merging
            target_format: Target format ("chatml" or "conversation_record")
            progress_callback: Optional callback for progress updates (source_id, progress_data)
            
        Returns:
            Dictionary with batch conversion results
        """
        logger.info(f"Starting batch conversion of {len(datasets)} datasets")
        
        results = {
            "total": len(datasets),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "conversions": {},
            "summary": {},
        }
        
        existing_path_str = str(existing_dataset_path) if existing_dataset_path else None
        
        for i, dataset in enumerate(datasets):
            source_id = dataset.source_id
            logger.info(f"Converting dataset {i+1}/{len(datasets)}: {source_id}")
            
            # Get integration plan
            integration_plan = integration_plans.get(source_id)
            if not integration_plan:
                logger.warning(
                    f"No integration plan found for {source_id}, skipping"
                )
                results["skipped"] += 1
                results["conversions"][source_id] = {
                    "status": "skipped",
                    "error": "No integration plan found",
                }
                continue
            
            # Determine output path
            output_filename = f"{source_id}_integrated.jsonl"
            output_path = self.output_directory / output_filename
            
            try:
                # Convert dataset
                conversion_result = self.integration_service.integrate_dataset(
                    dataset=dataset,
                    integration_plan=integration_plan,
                    existing_dataset_path=existing_path_str,
                    output_path=str(output_path),
                    target_format=target_format,
                    validate=True,
                    merge=existing_path_str is not None,
                    quality_check=True,
                )
                
                if conversion_result.get("success"):
                    results["successful"] += 1
                    results["conversions"][source_id] = {
                        "status": "success",
                        "output_path": str(output_path),
                        "conversion": conversion_result.get("conversion"),
                        "validation": conversion_result.get("validation"),
                        "quality_check": conversion_result.get("quality_check"),
                    }
                    logger.info(f"Successfully converted {source_id}")
                else:
                    results["failed"] += 1
                    error_msg = conversion_result.get("error", "Unknown error")
                    results["conversions"][source_id] = {
                        "status": "failed",
                        "error": error_msg,
                        "conversion": conversion_result.get("conversion"),
                    }
                    logger.error(f"Failed to convert {source_id}: {error_msg}")
                
                # Notify progress callback
                if progress_callback:
                    try:
                        progress_callback(source_id, {
                            "status": "success" if conversion_result.get("success") else "failed",
                            "progress": (i + 1) / len(datasets),
                            "current": i + 1,
                            "total": len(datasets),
                        })
                    except Exception as e:
                        logger.warning(f"Error in progress callback: {e}")
                
            except Exception as e:
                logger.error(
                    f"Error converting dataset {source_id}: {e}",
                    exc_info=True,
                )
                results["failed"] += 1
                results["conversions"][source_id] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        # Generate summary
        results["summary"] = {
            "total_datasets": results["total"],
            "successful": results["successful"],
            "failed": results["failed"],
            "skipped": results["skipped"],
            "success_rate": (
                results["successful"] / results["total"]
                if results["total"] > 0
                else 0.0
            ),
        }
        
        logger.info(
            f"Batch conversion complete: {results['successful']} successful, "
            f"{results['failed']} failed, {results['skipped']} skipped"
        )
        
        return results

    def convert_with_retry(
        self,
        dataset: AcquiredDataset,
        integration_plan: IntegrationPlan,
        max_retries: int = 3,
        existing_dataset_path: Optional[Path] = None,
        target_format: str = "chatml",
    ) -> Dict[str, Any]:
        """
        Convert a single dataset with retry logic.
        
        Args:
            dataset: Acquired dataset to convert
            integration_plan: Integration plan for the dataset
            max_retries: Maximum number of retry attempts
            existing_dataset_path: Optional path to existing dataset for merging
            target_format: Target format
            
        Returns:
            Conversion result dictionary
        """
        output_filename = f"{dataset.source_id}_integrated.jsonl"
        output_path = self.output_directory / output_filename
        existing_path_str = str(existing_dataset_path) if existing_dataset_path else None
        
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Conversion attempt {attempt + 1}/{max_retries} for {dataset.source_id}"
                )
                
                result = self.integration_service.integrate_dataset(
                    dataset=dataset,
                    integration_plan=integration_plan,
                    existing_dataset_path=existing_path_str,
                    output_path=str(output_path),
                    target_format=target_format,
                    validate=True,
                    merge=existing_path_str is not None,
                    quality_check=True,
                )
                
                if result.get("success"):
                    logger.info(
                        f"Successfully converted {dataset.source_id} on attempt {attempt + 1}"
                    )
                    return result
                else:
                    last_error = result.get("error", "Unknown error")
                    logger.warning(
                        f"Conversion failed for {dataset.source_id} on attempt {attempt + 1}: {last_error}"
                    )
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Exception during conversion attempt {attempt + 1} for {dataset.source_id}: {e}"
                )
        
        # All retries failed
        logger.error(
            f"Failed to convert {dataset.source_id} after {max_retries} attempts"
        )
        return {
            "success": False,
            "source_id": dataset.source_id,
            "error": f"Failed after {max_retries} attempts: {last_error}",
            "attempts": max_retries,
        }


