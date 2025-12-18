#!/usr/bin/env python3
"""
Provenance Metadata Schema Implementation

This module provides Python dataclasses and validation for the provenance
metadata schema as defined in governance/provenance_schema.json.

Related Documentation:
- Schema Definition: governance/provenance_schema.json
- Storage Plan: governance/provenance_storage_plan.md
- Audit Report Example: governance/audit_report_example.json
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of data sources."""

    JOURNAL = "journal"
    REPOSITORY = "repository"
    CLINICAL_TRIAL = "clinical_trial"
    TRAINING_MATERIAL = "training_material"
    SOCIAL_MEDIA = "social_media"
    SYNTHETIC = "synthetic"
    INTERNAL = "internal"


class AcquisitionMethod(Enum):
    """Methods for acquiring data."""

    API = "api"
    WEB_SCRAPING = "web_scraping"
    DIRECT_DOWNLOAD = "direct_download"
    MANUAL_COLLECTION = "manual_collection"
    CONTRACTED = "contracted"
    GENERATED = "generated"


class LicenseType(Enum):
    """Types of licenses."""

    PERMISSIVE = "permissive"
    CONTRACTED = "contracted"
    PROPRIETARY = "proprietary"
    UNCLEAR = "unclear"
    CC_BY = "cc_by"
    CC_BY_NC = "cc_by_nc"
    CC_BY_NC_ND = "cc_by_nc_nd"
    CUSTOM = "custom"
    NONE = "none"


class LicenseVerificationStatus(Enum):
    """License verification status."""

    VERIFIED = "verified"
    PENDING = "pending"
    UNVERIFIED = "unverified"
    UNCLEAR = "unclear"


class StorageType(Enum):
    """Storage types."""

    S3 = "s3"
    LOCAL = "local"
    DATABASE = "database"
    REMOTE = "remote"


class FileFormat(Enum):
    """File formats."""

    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    DATABASE = "database"


class CompressionType(Enum):
    """Compression types."""

    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    ZSTD = "zstd"


class TransformationType(Enum):
    """Types of transformations."""

    INGESTION = "ingestion"
    NORMALIZATION = "normalization"
    DEDUPLICATION = "deduplication"
    QUALITY_SCORING = "quality_scoring"
    FILTERING = "filtering"
    ANONYMIZATION = "anonymization"
    REDACTION = "redaction"
    VALIDATION = "validation"
    CURATION = "curation"
    COMPILATION = "compilation"


class DeduplicationMethod(Enum):
    """Deduplication methods."""

    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    NONE = "none"


class AnonymizationLevel(Enum):
    """Anonymization levels."""

    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    FULL = "full"


class ComplianceStatus(Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"
    UNKNOWN = "unknown"


class QualityTier(Enum):
    """Quality tiers."""

    PRIORITY = "priority"
    PROFESSIONAL = "professional"
    COT = "cot"
    REDDIT = "reddit"
    SYNTHETIC = "synthetic"
    ARCHIVE = "archive"


@dataclass
class SourceInfo:
    """Source information for the dataset."""

    source_id: str
    source_name: str
    source_type: SourceType
    acquisition_method: AcquisitionMethod
    acquisition_date: datetime
    source_url: Optional[str] = None
    doi: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    acquisition_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["source_type"] = self.source_type.value
        data["acquisition_method"] = self.acquisition_method.value
        # Convert datetimes to ISO format strings
        if self.acquisition_date:
            data["acquisition_date"] = self.acquisition_date.isoformat()
        if self.publication_date:
            data["publication_date"] = self.publication_date.isoformat()
        return data


@dataclass
class LicenseInfo:
    """License and usage rights information."""

    license_type: LicenseType
    allowed_uses: List[str]
    prohibited_uses: List[str]
    license_verification_status: LicenseVerificationStatus
    license_text: Optional[str] = None
    attribution_required: bool = False
    attribution_text: Optional[str] = None
    license_verified_by: Optional[str] = None
    license_verified_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["license_type"] = self.license_type.value
        data["license_verification_status"] = self.license_verification_status.value
        if self.license_verified_date:
            data["license_verified_date"] = self.license_verified_date.isoformat()
        return data


@dataclass
class Timestamps:
    """Critical timestamps throughout the dataset lifecycle."""

    created_at: datetime
    acquired_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        return {
            key: value.isoformat() if isinstance(value, datetime) else value
            for key, value in asdict(self).items()
            if value
        }


@dataclass
class ProcessingStage:
    """A single processing stage in the lineage."""

    stage_name: str
    stage_order: int
    started_at: datetime
    transformation_type: TransformationType
    completed_at: Optional[datetime] = None
    transformation_details: Dict[str, Any] = field(default_factory=dict)
    input_record_count: Optional[int] = None
    output_record_count: Optional[int] = None
    records_filtered: Optional[int] = None
    quality_metrics_before: Dict[str, Any] = field(default_factory=dict)
    quality_metrics_after: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["transformation_type"] = self.transformation_type.value
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


@dataclass
class ParentDataset:
    """Reference to a parent dataset if this is a merged/compiled dataset."""

    dataset_id: str
    dataset_name: str
    contribution_ratio: float = 0.0


@dataclass
class QualityScores:
    """Quality scores for the dataset."""

    empathy_score: Optional[float] = None
    fidelity_score: Optional[float] = None
    harmfulness_score: Optional[float] = None
    domain_relevance_score: Optional[float] = None
    overall_quality_score: Optional[float] = None


@dataclass
class DeduplicationInfo:
    """Deduplication information."""

    method: DeduplicationMethod
    duplicates_removed: Optional[int] = None
    duplication_rate: Optional[float] = None
    threshold_used: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["method"] = self.method.value
        return data


@dataclass
class AnonymizationInfo:
    """Anonymization information."""

    anonymization_level: AnonymizationLevel
    pii_removed_count: Optional[int] = None
    phi_removed_count: Optional[int] = None
    anonymization_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["anonymization_level"] = self.anonymization_level.value
        return data


@dataclass
class ProcessingLineage:
    """Complete processing lineage showing transformations applied."""

    pipeline_version: str
    processing_stages: List[ProcessingStage]
    parent_datasets: List[ParentDataset] = field(default_factory=list)
    quality_scores: Optional[QualityScores] = None
    deduplication_info: Optional[DeduplicationInfo] = None
    anonymization_info: Optional[AnonymizationInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = {
            "pipeline_version": self.pipeline_version,
            "processing_stages": [stage.to_dict() for stage in self.processing_stages],
            "parent_datasets": [asdict(pd) for pd in self.parent_datasets],
        }
        if self.quality_scores:
            data["quality_scores"] = asdict(self.quality_scores)
        if self.deduplication_info:
            data["deduplication_info"] = self.deduplication_info.to_dict()
        if self.anonymization_info:
            data["anonymization_info"] = self.anonymization_info.to_dict()
        return data


@dataclass
class StorageInfo:
    """Storage location and access information."""

    storage_type: StorageType
    storage_path: str
    file_format: FileFormat
    checksum: str
    storage_bucket: Optional[str] = None
    storage_region: Optional[str] = None
    file_size_bytes: Optional[int] = None
    compression: CompressionType = CompressionType.NONE
    access_restrictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["storage_type"] = self.storage_type.value
        data["file_format"] = self.file_format.value
        data["compression"] = self.compression.value
        return data


@dataclass
class AuditFinding:
    """An audit finding."""

    finding_type: str
    severity: str
    description: str
    recommendation: str
    status: str
    reported_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        if self.reported_at:
            data["reported_at"] = self.reported_at.isoformat()
        return data


@dataclass
class AuditInfo:
    """Audit trail information."""

    last_audited_at: Optional[datetime] = None
    audited_by: Optional[str] = None
    audit_findings: List[AuditFinding] = field(default_factory=list)
    compliance_status: ComplianceStatus = ComplianceStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        data["compliance_status"] = self.compliance_status.value
        if self.last_audited_at:
            data["last_audited_at"] = self.last_audited_at.isoformat()
        data["audit_findings"] = [f.to_dict() for f in self.audit_findings]
        return data


@dataclass
class DatasetMetadata:
    """Additional metadata."""

    record_count: Optional[int] = None
    total_tokens: Optional[int] = None
    quality_tier: Optional[QualityTier] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        if self.quality_tier:
            data["quality_tier"] = self.quality_tier.value
        return data


@dataclass
class ProvenanceRecord:
    """Complete provenance metadata record."""

    provenance_id: str
    dataset_id: str
    dataset_name: str
    source: SourceInfo
    license: LicenseInfo
    timestamps: Timestamps
    processing_lineage: ProcessingLineage
    storage: StorageInfo
    audit: AuditInfo = field(default_factory=AuditInfo)
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)

    def __post_init__(self):
        """Set updated_at if not provided."""
        if not self.timestamps.updated_at:
            self.timestamps.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provenance_id": self.provenance_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "source": self.source.to_dict(),
            "license": self.license.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "processing_lineage": self.processing_lineage.to_dict(),
            "storage": self.storage.to_dict(),
            "audit": self.audit.to_dict(),
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Create ProvenanceRecord from dictionary."""
        # Parse source
        source_data = data["source"]
        source = SourceInfo(
            source_id=source_data["source_id"],
            source_name=source_data["source_name"],
            source_type=SourceType(source_data["source_type"]),
            acquisition_method=AcquisitionMethod(source_data["acquisition_method"]),
            acquisition_date=datetime.fromisoformat(source_data["acquisition_date"]),
            source_url=source_data.get("source_url"),
            doi=source_data.get("doi"),
            authors=source_data.get("authors", []),
            publication_date=datetime.fromisoformat(source_data["publication_date"])
            if source_data.get("publication_date")
            else None,
            acquisition_metadata=source_data.get("acquisition_metadata", {}),
        )

        # Parse license
        license_data = data["license"]
        license = LicenseInfo(
            license_type=LicenseType(license_data["license_type"]),
            allowed_uses=license_data["allowed_uses"],
            prohibited_uses=license_data["prohibited_uses"],
            license_verification_status=LicenseVerificationStatus(
                license_data["license_verification_status"]
            ),
            license_text=license_data.get("license_text"),
            attribution_required=license_data.get("attribution_required", False),
            attribution_text=license_data.get("attribution_text"),
            license_verified_by=license_data.get("license_verified_by"),
            license_verified_date=datetime.fromisoformat(
                license_data["license_verified_date"]
            )
            if license_data.get("license_verified_date")
            else None,
        )

        # Parse timestamps
        ts_data = data["timestamps"]
        timestamps = Timestamps(
            created_at=datetime.fromisoformat(ts_data["created_at"]),
            acquired_at=datetime.fromisoformat(ts_data["acquired_at"])
            if ts_data.get("acquired_at")
            else None,
            processed_at=datetime.fromisoformat(ts_data["processed_at"])
            if ts_data.get("processed_at")
            else None,
            validated_at=datetime.fromisoformat(ts_data["validated_at"])
            if ts_data.get("validated_at")
            else None,
            published_at=datetime.fromisoformat(ts_data["published_at"])
            if ts_data.get("published_at")
            else None,
            updated_at=datetime.fromisoformat(ts_data["updated_at"])
            if ts_data.get("updated_at")
            else None,
        )

        # Parse processing lineage
        lineage_data = data["processing_lineage"]
        stages = []
        for stage_data in lineage_data["processing_stages"]:
            stages.append(
                ProcessingStage(
                    stage_name=stage_data["stage_name"],
                    stage_order=stage_data["stage_order"],
                    started_at=datetime.fromisoformat(stage_data["started_at"]),
                    transformation_type=TransformationType(
                        stage_data["transformation_type"]
                    ),
                    completed_at=datetime.fromisoformat(stage_data["completed_at"])
                    if stage_data.get("completed_at")
                    else None,
                    transformation_details=stage_data.get("transformation_details", {}),
                    input_record_count=stage_data.get("input_record_count"),
                    output_record_count=stage_data.get("output_record_count"),
                    records_filtered=stage_data.get("records_filtered"),
                    quality_metrics_before=stage_data.get("quality_metrics_before", {}),
                    quality_metrics_after=stage_data.get("quality_metrics_after", {}),
                    errors=stage_data.get("errors", []),
                    warnings=stage_data.get("warnings", []),
                )
            )

        processing_lineage = ProcessingLineage(
            pipeline_version=lineage_data["pipeline_version"],
            processing_stages=stages,
            parent_datasets=[
                ParentDataset(**pd) for pd in lineage_data.get("parent_datasets", [])
            ],
        )

        # Parse storage
        storage_data = data["storage"]
        storage = StorageInfo(
            storage_type=StorageType(storage_data["storage_type"]),
            storage_path=storage_data["storage_path"],
            file_format=FileFormat(storage_data["file_format"]),
            checksum=storage_data["checksum"],
            storage_bucket=storage_data.get("storage_bucket"),
            storage_region=storage_data.get("storage_region"),
            file_size_bytes=storage_data.get("file_size_bytes"),
            compression=CompressionType(storage_data.get("compression", "none")),
            access_restrictions=storage_data.get("access_restrictions", []),
        )

        return cls(
            provenance_id=data["provenance_id"],
            dataset_id=data["dataset_id"],
            dataset_name=data["dataset_name"],
            source=source,
            license=license,
            timestamps=timestamps,
            processing_lineage=processing_lineage,
            storage=storage,
            audit=AuditInfo(**data.get("audit", {})),
            metadata=DatasetMetadata(**data.get("metadata", {})),
        )

    def validate(self) -> Dict[str, Any]:
        """Validate the provenance record against schema requirements."""
        errors = []
        warnings = []

        # Required fields check
        if not self.provenance_id:
            errors.append("provenance_id is required")
        if not self.dataset_id:
            errors.append("dataset_id is required")
        if not self.dataset_name:
            errors.append("dataset_name is required")

        # License verification check
        if (
            self.license.license_verification_status
            == LicenseVerificationStatus.PENDING
        ):
            warnings.append(
                "License verification status is 'pending' - should be verified "
                "before production use"
            )

        # Processing stages order check
        stage_orders = [
            s.stage_order for s in self.processing_lineage.processing_stages
        ]
        if sorted(stage_orders) != list(range(len(stage_orders))):
            errors.append("Processing stage orders must be sequential starting from 0")

        # Checksum format check (should be hex)
        if self.storage.checksum and not all(
            c in "0123456789abcdefABCDEF" for c in self.storage.checksum
        ):
            warnings.append("Checksum format may be invalid (expected hex string)")

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


def create_provenance_record(
    dataset_id: str,
    dataset_name: str,
    source: SourceInfo,
    license: LicenseInfo,
    processing_lineage: ProcessingLineage,
    storage: StorageInfo,
) -> ProvenanceRecord:
    """Factory function to create a new provenance record with default values."""
    return ProvenanceRecord(
        provenance_id=str(uuid4()),
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        source=source,
        license=license,
        timestamps=Timestamps(created_at=datetime.now(timezone.utc)),
        processing_lineage=processing_lineage,
        storage=storage,
    )


if __name__ == "__main__":
    # Example usage
    source = SourceInfo(
        source_id="pubmed_001",
        source_name="PubMed Mental Health Abstracts",
        source_type=SourceType.JOURNAL,
        acquisition_method=AcquisitionMethod.API,
        acquisition_date=datetime.now(timezone.utc),
        source_url="https://pubmed.ncbi.nlm.nih.gov/",
    )

    license = LicenseInfo(
        license_type=LicenseType.PERMISSIVE,
        allowed_uses=["training", "evaluation", "research"],
        prohibited_uses=["redistribution"],
        license_verification_status=LicenseVerificationStatus.VERIFIED,
        license_verified_by="Data Governance Team",
        license_verified_date=datetime.now(timezone.utc),
    )

    processing_lineage = ProcessingLineage(
        pipeline_version="1.0.0",
        processing_stages=[
            ProcessingStage(
                stage_name="ingestion",
                stage_order=0,
                started_at=datetime.now(timezone.utc),
                transformation_type=TransformationType.INGESTION,
                completed_at=datetime.now(timezone.utc),
                input_record_count=0,
                output_record_count=1000,
            )
        ],
    )

    storage = StorageInfo(
        storage_type=StorageType.S3,
        storage_path="s3://bucket/datasets/pubmed_v1.jsonl",
        file_format=FileFormat.JSONL,
        checksum="abc123def456",
    )

    record = create_provenance_record(
        dataset_id="pubmed_mental_health_v1",
        dataset_name="PubMed Mental Health Abstracts v1",
        source=source,
        license=license,
        processing_lineage=processing_lineage,
        storage=storage,
    )

    # Validate
    validation = record.validate()
    print(f"Validation: {validation}")

    # Export to JSON
    print("\nProvenance Record JSON:")
    print(record.to_json())
