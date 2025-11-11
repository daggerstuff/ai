"""
Pipeline Integration Service

Main service that orchestrates the complete integration workflow:
1. Convert dataset to training pipeline format
2. Validate against schema
3. Merge with existing data
4. Perform quality checks

This service provides a high-level API for integrating journal datasets
into the training pipeline.
"""

import logging
from typing import Optional

from ai.journal_dataset_research.integration.integration_planning_engine import (
    IntegrationPlanningEngine,
)
from ai.journal_dataset_research.integration.pipeline_integrator import (
    DatasetMerger,
    PipelineFormatConverter,
    PipelineSchemaValidator,
    QualityChecker,
)
from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    IntegrationPlan,
)

logger = logging.getLogger(__name__)


class PipelineIntegrationService:
    """Main service for integrating datasets into the training pipeline."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        pipeline_schema: Optional[dict] = None,
    ):
        """Initialize the pipeline integration service."""
        self.format_converter = PipelineFormatConverter(pipeline_schema=pipeline_schema)
        self.schema_validator = PipelineSchemaValidator(pipeline_schema=pipeline_schema)
        self.dataset_merger = DatasetMerger(similarity_threshold=similarity_threshold)
        self.quality_checker = QualityChecker()
        self.integration_planner = IntegrationPlanningEngine(pipeline_schema=pipeline_schema)
        logger.info("Initialized Pipeline Integration Service")

    def integrate_dataset(
        self,
        dataset: AcquiredDataset,
        integration_plan: IntegrationPlan,
        existing_dataset_path: Optional[str] = None,
        output_path: str = "integrated_dataset.jsonl",
        target_format: str = "chatml",
        validate: bool = True,
        merge: bool = True,
        quality_check: bool = True,
    ) -> dict:
        """
        Complete integration workflow for a dataset.

        Args:
            dataset: The acquired dataset to integrate
            integration_plan: The integration plan for the dataset
            existing_dataset_path: Path to existing dataset for merging (optional)
            output_path: Path to save integrated dataset
            target_format: Target format ("chatml" or "conversation_record")
            validate: Whether to validate the converted dataset
            merge: Whether to merge with existing dataset
            quality_check: Whether to perform quality checks

        Returns:
            Dictionary with integration results and statistics
        """
        logger.info(f"Starting integration workflow for dataset: {dataset.source_id}")

        results = {
            "source_id": dataset.source_id,
            "target_format": target_format,
            "conversion": None,
            "validation": None,
            "merge": None,
            "quality_check": None,
            "success": False,
        }

        try:
            # Step 1: Convert dataset to training pipeline format
            logger.info("Step 1: Converting dataset to training pipeline format")
            conversion_result = self.format_converter.convert_dataset(
                dataset=dataset,
                integration_plan=integration_plan,
                output_path=output_path,
                target_format=target_format,
            )
            results["conversion"] = {
                "success": conversion_result.success,
                "records_converted": conversion_result.records_converted,
                "records_failed": conversion_result.records_failed,
                "errors": conversion_result.errors,
                "warnings": conversion_result.warnings,
            }

            if not conversion_result.success:
                logger.error("Conversion failed, aborting integration")
                return results

            # Step 2: Validate converted dataset
            if validate:
                logger.info("Step 2: Validating converted dataset")
                validation_result = self.schema_validator.validate_dataset(
                    dataset_path=output_path,
                    target_format=target_format,
                )
                results["validation"] = {
                    "valid": validation_result.valid,
                    "records_validated": validation_result.records_validated,
                    "records_passed": validation_result.records_passed,
                    "records_failed": validation_result.records_failed,
                    "errors": validation_result.errors,
                }

                if not validation_result.valid:
                    logger.warning(
                        f"Validation failed: {validation_result.records_failed} records failed"
                    )

            # Step 3: Merge with existing dataset
            merged_path = output_path
            if merge and existing_dataset_path:
                logger.info("Step 3: Merging with existing dataset")
                merge_result = self.dataset_merger.merge_datasets(
                    new_dataset_path=output_path,
                    existing_dataset_path=existing_dataset_path,
                    output_path=f"{output_path}.merged",
                    target_format=target_format,
                )
                results["merge"] = {
                    "success": merge_result.success,
                    "records_merged": merge_result.records_merged,
                    "duplicates_removed": merge_result.duplicates_removed,
                    "conflicts_resolved": merge_result.conflicts_resolved,
                    "errors": merge_result.errors,
                    "warnings": merge_result.warnings,
                }
                merged_path = f"{output_path}.merged"

                if not merge_result.success:
                    logger.warning("Merge had errors, but continuing with integration")

            # Step 4: Quality check
            if quality_check:
                logger.info("Step 4: Performing quality checks")
                quality_result = self.quality_checker.check_quality(
                    dataset_path=merged_path,
                    target_format=target_format,
                )
                results["quality_check"] = {
                    "passed": quality_result.passed,
                    "records_checked": quality_result.records_checked,
                    "records_passed": quality_result.records_passed,
                    "records_failed": quality_result.records_failed,
                    "pii_detected": quality_result.pii_detected,
                    "structure_issues": quality_result.structure_issues,
                    "completeness_issues": quality_result.completeness_issues,
                    "quality_score": quality_result.quality_score,
                    "errors": quality_result.errors,
                }

                if not quality_result.passed:
                    logger.warning(
                        f"Quality check failed: {quality_result.records_failed} records failed, "
                        f"quality score: {quality_result.quality_score:.2f}"
                    )

            # Determine overall success
            results["success"] = (
                conversion_result.success
                and (not validate or results["validation"]["valid"])
                and (not merge or not existing_dataset_path or results["merge"]["success"])
                and (not quality_check or results["quality_check"]["passed"])
            )

            results["output_path"] = merged_path
            logger.info(
                f"Integration workflow complete: success={results['success']}, "
                f"output={merged_path}"
            )

            return results

        except Exception as e:
            logger.error(f"Integration workflow failed: {e}", exc_info=True)
            results["error"] = str(e)
            return results

    def create_integration_plan(
        self, dataset: AcquiredDataset, target_format: str = "chatml"
    ) -> IntegrationPlan:
        """
        Create an integration plan for a dataset.

        Args:
            dataset: The acquired dataset
            target_format: Target format ("chatml" or "conversation_record")

        Returns:
            IntegrationPlan with analysis and transformation details
        """
        logger.info(f"Creating integration plan for dataset: {dataset.source_id}")
        return self.integration_planner.create_integration_plan(
            dataset=dataset, target_format=target_format
        )

