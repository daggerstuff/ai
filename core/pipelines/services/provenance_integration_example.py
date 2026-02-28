#!/usr/bin/env python3
"""
Example: Provenance Integration in Pipeline

This example shows how to integrate provenance tracking into the dataset pipeline.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from .provenance_orchestrator_wrapper import ProvenanceOrchestratorWrapper
from .provenance_service import ProvenanceService


async def example_pipeline_with_provenance():
    """
    Example pipeline execution with provenance tracking.

    This demonstrates the complete workflow:
    1. Initialize provenance at start
    2. Track processing stages
    3. Update storage information
    4. Finalize provenance
    """
    # Initialize provenance wrapper
    provenance_wrapper = ProvenanceOrchestratorWrapper()
    await provenance_wrapper.initialize()

    dataset_id = "priority_1_FINAL"
    dataset_name = "Priority 1 Therapeutic Conversations"
    dataset_path = Path("datasets/consolidated/priority_1_FINAL.jsonl")

    try:
        # 1. Initialize provenance at start of processing
        await provenance_wrapper.initialize_dataset_provenance(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_info={
                "source_id": "gdrive_wendy",
                "source_name": "GDrive Wendy Priority Datasets",
                "source_type": "training_material",
                "acquisition_method": "direct_download",
            },
            license_info={
                "license_type": "contracted",
                "allowed_uses": ["training", "evaluation"],
                "prohibited_uses": ["redistribution"],
            },
            pipeline_version="1.0.0",
        )

        # 2. Track ingestion stage
        await provenance_wrapper.track_processing_stage(
            dataset_id=dataset_id,
            stage_name="ingestion",
            transformation_type="ingestion",
            stage_order=0,
            input_count=0,
            output_count=462000,  # Example count
            started_at=datetime.now(timezone.utc),
        )

        # 3. Track normalization stage
        await provenance_wrapper.track_processing_stage(
            dataset_id=dataset_id,
            stage_name="normalization",
            transformation_type="normalization",
            stage_order=1,
            input_count=462000,
            output_count=462000,
            started_at=datetime.now(timezone.utc),
        )

        # 4. Track deduplication stage
        await provenance_wrapper.track_processing_stage(
            dataset_id=dataset_id,
            stage_name="deduplication",
            transformation_type="deduplication",
            stage_order=2,
            input_count=462000,
            output_count=425000,
            records_filtered=37000,
            started_at=datetime.now(timezone.utc),
        )

        # 5. Track quality scoring stage
        await provenance_wrapper.track_processing_stage(
            dataset_id=dataset_id,
            stage_name="quality_scoring",
            transformation_type="quality_scoring",
            stage_order=3,
            input_count=425000,
            output_count=425000,
            started_at=datetime.now(timezone.utc),
        )

        # 6. Calculate checksum and update storage info
        if dataset_path.exists():
            checksum = ProvenanceService.calculate_checksum(dataset_path)
            file_size = dataset_path.stat().st_size

            await provenance_wrapper.update_storage_information(
                dataset_id=dataset_id,
                storage_path=str(dataset_path),
                checksum=checksum,
                file_size_bytes=file_size,
            )

        # 7. Finalize provenance after processing complete
        await provenance_wrapper.finalize_provenance(
            dataset_id=dataset_id,
            record_count=425000,
            quality_tier="priority",
        )

        print(f"✅ Provenance tracking complete for {dataset_id}")

    finally:
        await provenance_wrapper.cleanup()


if __name__ == "__main__":
    asyncio.run(example_pipeline_with_provenance())
