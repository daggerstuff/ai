#!/usr/bin/env python3
"""
Provenance Pipeline Integration

Integration layer for adding provenance tracking to dataset pipeline orchestrators.
This module provides decorators and context managers for automatic provenance tracking.
"""

import functools
import logging
from typing import Callable, Optional

from .provenance_integration import (
    create_provenance_from_pipeline_stage,
    track_pipeline_stage,
    update_storage_info,
)
from .provenance_service import ProvenanceService

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Context manager for tracking provenance during pipeline execution.
    """

    def __init__(
        self,
        dataset_id: str,
        dataset_name: str,
        provenance_service: Optional[ProvenanceService] = None,
    ):
        """
        Initialize provenance tracker.

        Args:
            dataset_id: Dataset identifier
            dataset_name: Dataset name
            provenance_service: ProvenanceService instance (creates if not provided)
        """
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.provenance_service = provenance_service
        self._service_created = False

    async def __aenter__(self):
        """Enter context manager."""
        if not self.provenance_service:
            self.provenance_service = ProvenanceService()
            await self.provenance_service.connect()
            await self.provenance_service.ensure_schema()
            self._service_created = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self._service_created and self.provenance_service:
            await self.provenance_service.disconnect()

    async def track_stage(
        self,
        stage_name: str,
        transformation_type: str,
        stage_order: int,
        **kwargs,
    ) -> None:
        """
        Track a processing stage.

        Args:
            stage_name: Name of the stage
            transformation_type: Type of transformation
            stage_order: Order in pipeline
            **kwargs: Additional stage parameters
        """
        await track_pipeline_stage(
            self.provenance_service,
            self.dataset_id,
            stage_name,
            transformation_type,
            stage_order,
            **kwargs,
        )

    async def update_storage(
        self,
        storage_path: str,
        checksum: str,
        **kwargs,
    ) -> None:
        """Update storage information."""
        await update_storage_info(
            self.provenance_service,
            self.dataset_id,
            storage_path,
            checksum,
            **kwargs,
        )


def with_provenance_tracking(
    dataset_id_field: str = "dataset_id",
    dataset_name_field: str = "dataset_name",
):
    """
    Decorator to automatically track provenance for pipeline functions.

    Args:
        dataset_id_field: Field name in kwargs containing dataset_id
        dataset_name_field: Field name in kwargs containing dataset_name

    Example:
        @with_provenance_tracking()
        async def process_dataset(dataset_id: str, dataset_name: str, ...):
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dataset info from kwargs
            dataset_id = kwargs.get(dataset_id_field)
            dataset_name = kwargs.get(dataset_name_field)

            if not dataset_id:
                logger.warning(
                    f"No {dataset_id_field} found, skipping provenance tracking"
                )
                return await func(*args, **kwargs)

            # Create provenance tracker
            async with ProvenanceTracker(
                dataset_id=dataset_id or "unknown",
                dataset_name=dataset_name or "Unknown Dataset",
            ) as tracker:
                # Track function start
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.error(
                        f"Pipeline function failed: {str(e)}",
                        dataset_id=dataset_id,
                    )
                    raise

        return wrapper

    return decorator


async def initialize_dataset_provenance(
    dataset_id: str,
    dataset_name: str,
    source_id: str,
    source_name: str,
    source_type: str,
    acquisition_method: str,
    license_type: str,
    allowed_uses: list[str],
    prohibited_uses: list[str],
    provenance_service: Optional[ProvenanceService] = None,
) -> ProvenanceRecord:
    """
    Initialize provenance record at start of dataset processing.

    Args:
        dataset_id: Dataset identifier
        dataset_name: Dataset name
        source_id: Source identifier
        source_name: Source name
        source_type: Source type
        acquisition_method: Acquisition method
        license_type: License type
        allowed_uses: Allowed uses
        prohibited_uses: Prohibited uses
        provenance_service: ProvenanceService instance

    Returns:
        ProvenanceRecord
    """
    # Create provenance record
    provenance = create_provenance_from_pipeline_stage(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        source_id=source_id,
        source_name=source_name,
        source_type=source_type,
        acquisition_method=acquisition_method,
        license_type=license_type,
        allowed_uses=allowed_uses,
        prohibited_uses=prohibited_uses,
    )

    # Store if service provided
    if provenance_service:
        await provenance_service.create_provenance(
            provenance,
            changed_by="pipeline",
        )

    return provenance
