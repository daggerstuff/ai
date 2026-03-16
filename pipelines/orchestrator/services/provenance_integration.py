#!/usr/bin/env python3
"""
Provenance Integration Helpers

Helper functions for integrating provenance tracking into the dataset pipeline.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..schemas.provenance_schema import (
    AcquisitionMethod,
    FileFormat,
    LicenseInfo,
    LicenseType,
    LicenseVerificationStatus,
    ProcessingLineage,
    ProcessingStage,
    ProvenanceRecord,
    SourceInfo,
    SourceType,
    StorageInfo,
    StorageType,
    Timestamps,
    TransformationType,
)

logger = logging.getLogger(__name__)


def create_provenance_from_pipeline_stage(
    dataset_id: str,
    dataset_name: str,
    source_id: str,
    source_name: str,
    source_type: str,
    acquisition_method: str,
    license_type: str,
    allowed_uses: list[str],
    prohibited_uses: list[str],
    pipeline_version: str = "1.0.0",
    processing_stages: Optional[list[Dict[str, Any]]] = None,
    storage_path: Optional[str] = None,
    storage_bucket: Optional[str] = None,
    checksum: Optional[str] = None,
    record_count: Optional[int] = None,
    quality_tier: Optional[str] = None,
) -> ProvenanceRecord:
    """
    Create a ProvenanceRecord from pipeline processing information.

    Args:
        dataset_id: Unique dataset identifier
        dataset_name: Human-readable dataset name
        source_id: Source identifier from data source matrix
        source_name: Source name
        source_type: Type of source (journal, repository, etc.)
        acquisition_method: How data was acquired
        license_type: License type
        allowed_uses: List of allowed uses
        prohibited_uses: List of prohibited uses
        pipeline_version: Version of processing pipeline
        processing_stages: List of processing stage dictionaries
        storage_path: Storage location path
        storage_bucket: S3 bucket name
        checksum: File checksum
        record_count: Number of records
        quality_tier: Quality tier assignment

    Returns:
        ProvenanceRecord
    """
    # Create source info
    source = SourceInfo(
        source_id=source_id,
        source_name=source_name,
        source_type=SourceType(source_type.lower()),
        acquisition_method=AcquisitionMethod(acquisition_method.lower()),
        acquisition_date=datetime.now(timezone.utc),
    )

    # Create license info
    license = LicenseInfo(
        license_type=LicenseType(license_type.lower()),
        allowed_uses=allowed_uses,
        prohibited_uses=prohibited_uses,
        license_verification_status=LicenseVerificationStatus.PENDING,
    )

    # Create timestamps
    timestamps = Timestamps(created_at=datetime.now(timezone.utc))

    # Create processing stages
    stages = []
    if processing_stages:
        for idx, stage_data in enumerate(processing_stages):
            stage = ProcessingStage(
                stage_name=stage_data.get("stage_name", f"stage_{idx}"),
                stage_order=stage_data.get("stage_order", idx),
                started_at=datetime.fromisoformat(stage_data.get("started_at"))
                if isinstance(stage_data.get("started_at"), str)
                else datetime.now(timezone.utc),
                transformation_type=TransformationType(
                    stage_data.get("transformation_type", "processing")
                ),
                completed_at=datetime.fromisoformat(stage_data.get("completed_at"))
                if isinstance(stage_data.get("completed_at"), str)
                else None,
                transformation_details=stage_data.get("transformation_details", {}),
                input_record_count=stage_data.get("input_record_count"),
                output_record_count=stage_data.get("output_record_count"),
                records_filtered=stage_data.get("records_filtered"),
            )
            stages.append(stage)

    # Create processing lineage
    lineage = ProcessingLineage(
        pipeline_version=pipeline_version,
        processing_stages=stages,
    )

    # Create storage info
    storage_type = StorageType.S3 if storage_bucket else StorageType.LOCAL
    storage = StorageInfo(
        storage_type=storage_type,
        storage_path=storage_path or f"datasets/{dataset_id}",
        file_format=FileFormat.JSONL,
        checksum=checksum or "",
        storage_bucket=storage_bucket,
    )

    # Create provenance record
    from ..schemas.provenance_schema import create_provenance_record

    provenance = create_provenance_record(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        source=source,
        license=license,
        processing_lineage=lineage,
        storage=storage,
    )

    # Add metadata
    if record_count:
        provenance.metadata.record_count = record_count
    if quality_tier:
        from ..schemas.provenance_schema import QualityTier

        provenance.metadata.quality_tier = QualityTier(quality_tier.lower())

    return provenance


async def track_pipeline_stage(
    provenance_service,
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
    Add a processing stage to existing provenance record.

    Args:
        provenance_service: ProvenanceService instance
        dataset_id: Dataset identifier
        stage_name: Name of the processing stage
        transformation_type: Type of transformation
        stage_order: Order of stage in pipeline
        input_count: Records before stage
        output_count: Records after stage
        records_filtered: Records filtered out
        started_at: Stage start time
        completed_at: Stage completion time
    """
    try:
        # Get existing provenance
        provenance = await provenance_service.get_provenance(dataset_id)
        if not provenance:
            logger.warning(
                f"No provenance record found for {dataset_id}, cannot track stage"
            )
            return

        # Create new stage
        stage = ProcessingStage(
            stage_name=stage_name,
            stage_order=stage_order,
            started_at=started_at or datetime.now(timezone.utc),
            transformation_type=TransformationType(transformation_type.lower()),
            completed_at=completed_at,
            input_record_count=input_count,
            output_record_count=output_count,
            records_filtered=records_filtered,
        )

        # Add to lineage
        provenance.processing_lineage.processing_stages.append(stage)

        # Update processed_at if stage completed
        if completed_at:
            provenance.timestamps.processed_at = completed_at
            provenance.timestamps.updated_at = datetime.now(timezone.utc)

        # Update provenance record
        await provenance_service.update_provenance(
            provenance,
            changed_by="pipeline",
            change_reason=f"Added processing stage: {stage_name}",
        )

        logger.info(
            f"Tracked pipeline stage: {stage_name} for {dataset_id}",
            dataset_id=dataset_id,
            stage_name=stage_name,
        )

    except Exception as e:
        logger.error(
            f"Failed to track pipeline stage: {str(e)}",
            error=str(e),
            dataset_id=dataset_id,
            stage_name=stage_name,
        )


async def update_storage_info(
    provenance_service,
    dataset_id: str,
    storage_path: str,
    checksum: str,
    file_size_bytes: Optional[int] = None,
    storage_bucket: Optional[str] = None,
) -> None:
    """
    Update storage information in provenance record.

    Args:
        provenance_service: ProvenanceService instance
        dataset_id: Dataset identifier
        storage_path: Path to stored dataset
        checksum: File checksum
        file_size_bytes: File size in bytes
        storage_bucket: S3 bucket if applicable
    """
    try:
        provenance = await provenance_service.get_provenance(dataset_id)
        if not provenance:
            logger.warning(
                f"No provenance record found for {dataset_id}, cannot update storage"
            )
            return

        # Update storage info
        provenance.storage.storage_path = storage_path
        provenance.storage.checksum = checksum
        if file_size_bytes:
            provenance.storage.file_size_bytes = file_size_bytes
        if storage_bucket:
            provenance.storage.storage_bucket = storage_bucket
            provenance.storage.storage_type = StorageType.S3

        provenance.timestamps.updated_at = datetime.now(timezone.utc)

        await provenance_service.update_provenance(
            provenance,
            changed_by="pipeline",
            change_reason="Updated storage information",
        )

        logger.info(
            f"Updated storage info for {dataset_id}",
            dataset_id=dataset_id,
            storage_path=storage_path,
        )

    except Exception as e:
        logger.error(
            f"Failed to update storage info: {str(e)}",
            error=str(e),
            dataset_id=dataset_id,
        )
