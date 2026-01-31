#!/usr/bin/env python3
"""
Provenance Orchestrator Wrapper

Wrapper class to add provenance tracking to existing pipeline orchestrators.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .provenance_integration import (
    create_provenance_from_pipeline_stage,
    track_pipeline_stage,
    update_storage_info,
)
from .provenance_service import ProvenanceService

logger = logging.getLogger(__name__)


class ProvenanceOrchestratorWrapper:
    """
    Wrapper to add provenance tracking to pipeline orchestrators.

    Usage:
        wrapper = ProvenanceOrchestratorWrapper(provenance_service)
        await wrapper.initialize_provenance(dataset_id, ...)
        await wrapper.track_stage(...)
        await wrapper.finalize_provenance(...)
    """

    def __init__(
        self,
        provenance_service: Optional[ProvenanceService] = None,
        enabled: bool = True,
    ):
        """
        Initialize wrapper.

        Args:
            provenance_service: ProvenanceService instance (creates if None and enabled=True)
            enabled: Whether provenance tracking is enabled
        """
        self.enabled = enabled
        self.provenance_service = provenance_service
        self._service_created = False

        if enabled and not provenance_service:
            self.provenance_service = ProvenanceService()
            self._service_created = True

    async def initialize(self) -> None:
        """Initialize provenance service connection."""
        if not self.enabled:
            return

        if self._service_created and self.provenance_service:
            await self.provenance_service.connect()
            await self.provenance_service.ensure_schema()

    async def cleanup(self) -> None:
        """Cleanup provenance service connection."""
        if self._service_created and self.provenance_service:
            await self.provenance_service.disconnect()

    async def initialize_dataset_provenance(
        self,
        dataset_id: str,
        dataset_name: str,
        source_info: Dict[str, Any],
        license_info: Dict[str, Any],
        pipeline_version: str = "1.0.0",
    ) -> None:
        """
        Initialize provenance record at start of dataset processing.

        Args:
            dataset_id: Dataset identifier
            dataset_name: Dataset name
            source_info: Dictionary with source information
            license_info: Dictionary with license information
            pipeline_version: Pipeline version
        """
        if not self.enabled or not self.provenance_service:
            return

        try:
            provenance = create_provenance_from_pipeline_stage(
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                source_id=source_info.get("source_id", "unknown"),
                source_name=source_info.get("source_name", "Unknown Source"),
                source_type=source_info.get("source_type", "unknown"),
                acquisition_method=source_info.get("acquisition_method", "unknown"),
                license_type=license_info.get("license_type", "unclear"),
                allowed_uses=license_info.get("allowed_uses", []),
                prohibited_uses=license_info.get("prohibited_uses", []),
                pipeline_version=pipeline_version,
            )

            await self.provenance_service.create_provenance(
                provenance,
                changed_by="pipeline",
            )

            logger.info(
                f"Initialized provenance for {dataset_id}",
                dataset_id=dataset_id,
            )

        except Exception as e:
            logger.warning(
                f"Failed to initialize provenance: {str(e)}",
                error=str(e),
                dataset_id=dataset_id,
            )

    async def track_processing_stage(
        self,
        dataset_id: str,
        stage_name: str,
        transformation_type: str,
        stage_order: int,
        input_count: Optional[int] = None,
        output_count: Optional[int] = None,
        records_filtered: Optional[int] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        """
        Track a processing stage.

        Args:
            dataset_id: Dataset identifier
            stage_name: Name of the stage
            transformation_type: Type of transformation
            stage_order: Order in pipeline
            input_count: Records before stage
            output_count: Records after stage
            records_filtered: Records filtered out
            started_at: Stage start time
            completed_at: Stage completion time
        """
        if not self.enabled or not self.provenance_service:
            return

        try:
            await track_pipeline_stage(
                self.provenance_service,
                dataset_id,
                stage_name,
                transformation_type,
                stage_order,
                input_count=input_count,
                output_count=output_count,
                records_filtered=records_filtered,
                started_at=started_at or datetime.now(timezone.utc),
                completed_at=completed_at,
            )
        except Exception as e:
            logger.warning(
                f"Failed to track stage: {str(e)}",
                error=str(e),
                dataset_id=dataset_id,
                stage_name=stage_name,
            )

    async def update_storage_information(
        self,
        dataset_id: str,
        storage_path: str,
        checksum: str,
        file_size_bytes: Optional[int] = None,
        storage_bucket: Optional[str] = None,
    ) -> None:
        """
        Update storage information.

        Args:
            dataset_id: Dataset identifier
            storage_path: Path to stored dataset
            checksum: File checksum
            file_size_bytes: File size in bytes
            storage_bucket: S3 bucket if applicable
        """
        if not self.enabled or not self.provenance_service:
            return

        try:
            await update_storage_info(
                self.provenance_service,
                dataset_id,
                storage_path,
                checksum,
                file_size_bytes=file_size_bytes,
                storage_bucket=storage_bucket,
            )
        except Exception as e:
            logger.warning(
                f"Failed to update storage info: {str(e)}",
                error=str(e),
                dataset_id=dataset_id,
            )

    async def finalize_provenance(
        self,
        dataset_id: str,
        record_count: Optional[int] = None,
        quality_tier: Optional[str] = None,
    ) -> None:
        """
        Finalize provenance record after processing complete.

        Args:
            dataset_id: Dataset identifier
            record_count: Final record count
            quality_tier: Quality tier assignment
        """
        if not self.enabled or not self.provenance_service:
            return

        try:
            provenance = await self.provenance_service.get_provenance(dataset_id)
            if not provenance:
                logger.warning(f"No provenance found for {dataset_id} to finalize")
                return

            # Update metadata
            if record_count:
                provenance.metadata.record_count = record_count
            if quality_tier:
                from ..schemas.provenance_schema import QualityTier

                provenance.metadata.quality_tier = QualityTier(quality_tier.lower())

            # Update timestamps
            provenance.timestamps.processed_at = datetime.now(timezone.utc)
            provenance.timestamps.validated_at = datetime.now(timezone.utc)
            provenance.timestamps.updated_at = datetime.now(timezone.utc)

            await self.provenance_service.update_provenance(
                provenance,
                changed_by="pipeline",
                change_reason="Finalized after processing complete",
            )

            logger.info(
                f"Finalized provenance for {dataset_id}",
                dataset_id=dataset_id,
            )

        except Exception as e:
            logger.warning(
                f"Failed to finalize provenance: {str(e)}",
                error=str(e),
                dataset_id=dataset_id,
            )
