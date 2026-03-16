# Provenance Service

Service layer for managing dataset provenance metadata in the Dataset Expansion project.

## Overview

The Provenance Service provides complete provenance tracking for all datasets, including:
- Source information and acquisition details
- License and usage rights tracking
- Complete processing lineage
- Storage location and integrity verification
- Audit trail for compliance

## Components

### Core Service

- **`provenance_service.py`**: Main service class with CRUD operations
- **`provenance_integration.py`**: Helper functions for pipeline integration
- **`provenance_pipeline_integration.py`**: Decorators and context managers

### Schema

- **`../schemas/provenance_schema.py`**: Python dataclasses for provenance data models
- **`../../docs/governance/provenance_schema.json`**: JSON schema definition
- **`../../db/provenance_schema.sql`**: Database schema migration

## Usage

### Basic Usage

```python
from ai.pipelines.orchestrator.services.provenance_service import ProvenanceService

# Initialize service
service = ProvenanceService()
await service.connect()
await service.ensure_schema()

# Create provenance record
from ai.pipelines.orchestrator.schemas.provenance_schema import (
    SourceInfo, LicenseInfo, ProcessingLineage, StorageInfo,
    SourceType, AcquisitionMethod, LicenseType, LicenseVerificationStatus
)

source = SourceInfo(
    source_id="pubmed_001",
    source_name="PubMed Mental Health Abstracts",
    source_type=SourceType.JOURNAL,
    acquisition_method=AcquisitionMethod.API,
    acquisition_date=datetime.now(timezone.utc),
)

license = LicenseInfo(
    license_type=LicenseType.PERMISSIVE,
    allowed_uses=["training", "evaluation"],
    prohibited_uses=["redistribution"],
    license_verification_status=LicenseVerificationStatus.VERIFIED,
)

# ... create lineage and storage ...

provenance = create_provenance_record(
    dataset_id="pubmed_v1",
    dataset_name="PubMed Mental Health v1",
    source=source,
    license=license,
    processing_lineage=lineage,
    storage=storage,
)

# Store
await service.create_provenance(provenance, changed_by="pipeline")
```

### Pipeline Integration

```python
from ai.pipelines.orchestrator.services.provenance_pipeline_integration import (
    ProvenanceTracker,
    initialize_dataset_provenance,
)

# Initialize at start of processing
provenance = await initialize_dataset_provenance(
    dataset_id="my_dataset",
    dataset_name="My Dataset",
    source_id="source_123",
    source_name="Data Source",
    source_type="journal",
    acquisition_method="api",
    license_type="permissive",
    allowed_uses=["training"],
    prohibited_uses=["redistribution"],
)

# Track processing stages
async with ProvenanceTracker(dataset_id, dataset_name) as tracker:
    await tracker.track_stage(
        stage_name="ingestion",
        transformation_type="ingestion",
        stage_order=0,
        input_count=0,
        output_count=1000,
    )
    
    await tracker.track_stage(
        stage_name="quality_scoring",
        transformation_type="quality_scoring",
        stage_order=1,
        input_count=1000,
        output_count=950,
        records_filtered=50,
    )
```

### CLI Usage

```bash
# Initialize database schema
python -m ai.pipelines.orchestrator.cli.provenance_cli init-schema

# Create provenance record from JSON file
python -m ai.pipelines.orchestrator.cli.provenance_cli create \
    --dataset-id my_dataset \
    --file provenance.json

# Get provenance record
python -m ai.pipelines.orchestrator.cli.provenance_cli get --dataset-id my_dataset

# List provenance records
python -m ai.pipelines.orchestrator.cli.provenance_cli list \
    --license-type permissive \
    --quality-tier priority \
    --limit 20

# View audit log
python -m ai.pipelines.orchestrator.cli.provenance_cli audit-log \
    --dataset-id my_dataset \
    --limit 50
```

## Database Schema

The service uses PostgreSQL (Supabase) with two main tables:

- **`dataset_provenance`**: Main provenance records
- **`provenance_audit_log`**: Immutable audit trail

See `db/provenance_schema.sql` for complete schema definition.

## S3 Storage

Provenance documents are automatically stored to S3 at:
```
s3://{bucket}/provenance/{dataset_id}/v{timestamp}/provenance.json
```

Configure via environment variables:
- `S3_BUCKET`: Bucket name (default: "pixelated-datasets")
- `S3_REGION`: AWS region (default: "us-east-1")

## Configuration

Environment variables:
- `DATABASE_URL` or `SUPABASE_DB_URL`: PostgreSQL connection string
- `S3_BUCKET`: S3 bucket for file storage
- `S3_REGION`: AWS region

## Integration Points

To integrate provenance tracking into existing pipeline:

1. **At Dataset Ingestion**: Create initial provenance record
2. **During Processing**: Track each processing stage
3. **After Storage**: Update storage information with checksum
4. **On Completion**: Update final timestamps and metadata

See `provenance_pipeline_integration.py` for decorators and context managers.

## Related Documentation

- Schema Definition: `docs/governance/provenance_schema.json`
- Storage Plan: `docs/governance/provenance_storage_plan.md`
- Audit Report Example: `docs/governance/audit_report_example.json`
- Jira Issue: KAN-9
