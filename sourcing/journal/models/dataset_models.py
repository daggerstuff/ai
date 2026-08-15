"""
Data models for the journal dataset research system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ResearchProgress:
    """Tracks progress metrics for the research session."""

    sources_identified: int = 0
    datasets_evaluated: int = 0
    datasets_acquired: int = 0
    access_established: int = 0
    integration_plans_created: int = 0
    last_updated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, int]:
        return {
            "sources_identified": self.sources_identified,
            "datasets_evaluated": self.datasets_evaluated,
            "datasets_acquired": self.datasets_acquired,
            "access_established": self.access_established,
            "integration_plans_created": self.integration_plans_created,
        }


@dataclass
class WeeklyReport:
    """Weekly progress report."""

    week_number: int
    start_date: datetime
    end_date: datetime
    sources_identified: int = 0
    datasets_evaluated: int = 0
    datasets_acquired: int = 0
    access_established: int = 0
    integration_plans_created: int = 0
    key_findings: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    next_week_priorities: List[str] = field(default_factory=list)


@dataclass
class DatasetSource:
    """Represents a potential dataset source (paper, repo, etc)."""

    source_id: str
    title: str
    url: str
    source_type: str = "journal"
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    data_availability: str = "unknown"
    metadata: Dict = field(default_factory=dict)
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    open_access: bool = False
    discovery_date: Optional[datetime] = None
    discovery_method: Optional[str] = None

    VALID_SOURCE_TYPES = [
        "journal",
        "repository",
        "clinical_trial",
        "training_material",
    ]
    VALID_AVAILABILITY = ["available", "upon_request", "restricted", "unknown"]

    def validate(self) -> List[str]:
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if not self.title:
            errors.append("title is required")
        if not self.url:
            errors.append("url is required")
        if self.source_type not in self.VALID_SOURCE_TYPES:
            errors.append(f"source_type must be one of {self.VALID_SOURCE_TYPES}")
        if self.data_availability not in self.VALID_AVAILABILITY:
            errors.append(f"data_availability must be one of {self.VALID_AVAILABILITY}")
        return errors


@dataclass
class DatasetEvaluation:
    """Evaluation metrics for a dataset."""

    source_id: str
    therapeutic_relevance: int  # 1-10
    data_structure_quality: int  # 1-10
    training_integration: int  # 1-10
    ethical_accessibility: int  # 1-10
    priority_tier: str = "medium"
    evaluator_notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    therapeutic_relevance_notes: str = ""
    data_structure_notes: str = ""
    integration_notes: str = ""
    ethical_notes: str = ""
    overall_score: float = 0.0
    evaluation_date: datetime = field(default_factory=datetime.now)
    evaluator: str = "system"
    competitive_advantages: List[str] = field(default_factory=list)
    compliance_checked: bool = False
    compliance_status: str = ""
    compliance_score: float = 0.0
    license_compatible: bool = False
    privacy_compliant: bool = False
    hipaa_compliant: bool = False

    VALID_TIERS = ["high", "medium", "low"]

    def validate(self) -> List[str]:
        errors = []
        if not self.source_id:
            errors.append("source_id is required")

        for score_name, score in [
            ("therapeutic_relevance", self.therapeutic_relevance),
            ("data_structure_quality", self.data_structure_quality),
            ("training_integration", self.training_integration),
            ("ethical_accessibility", self.ethical_accessibility),
        ]:
            if not 1 <= score <= 10:
                errors.append(f"{score_name} must be between 1 and 10")

        if self.priority_tier not in self.VALID_TIERS:
            errors.append(f"priority_tier must be one of {self.VALID_TIERS}")
        return errors


@dataclass
class AccessRequest:
    """Request for dataset access."""

    source_id: str
    access_method: str
    status: str = "pending"
    notes: str = ""
    request_date: datetime = field(default_factory=datetime.now)
    access_url: Optional[str] = None
    credentials_required: bool = False
    institutional_affiliation_required: bool = False
    estimated_access_date: Optional[datetime] = None

    VALID_METHODS = ["direct", "api", "request_form", "collaboration", "registration"]
    VALID_STATUSES = ["pending", "approved", "denied", "downloaded", "error"]

    def validate(self) -> List[str]:
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if self.access_method not in self.VALID_METHODS:
            errors.append(f"access_method must be one of {self.VALID_METHODS}")
        if self.status not in self.VALID_STATUSES:
            errors.append(f"status must be one of {self.VALID_STATUSES}")
        return errors


@dataclass
class AcquiredDataset:
    """Successfully acquired dataset."""

    source_id: str
    storage_path: str
    acquisition_date: datetime = field(default_factory=datetime.now)
    file_metadata: Dict = field(default_factory=dict)
    file_format: str = ""
    file_size_mb: float = 0.0
    license: str = ""
    usage_restrictions: List[str] = field(default_factory=list)
    attribution_required: bool = False
    checksum: str = ""
    encrypted: bool = False
    compliance_status: str = ""
    compliance_score: float = 0.0
    hipaa_compliant: bool = False
    privacy_assessed: bool = False

    def validate(self) -> List[str]:
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if not self.storage_path:
            errors.append("storage_path is required")
        return errors


@dataclass
class TransformationSpec:
    """Specification for data transformation."""

    transformation_type: str
    input_format: str
    output_format: str
    transformation_logic: str

    VALID_TYPES = ["format_conversion", "field_mapping", "cleaning", "validation"]

    def validate(self) -> List[str]:
        errors = []
        if self.transformation_type not in self.VALID_TYPES:
            errors.append(f"transformation_type must be one of {self.VALID_TYPES}")
        return errors


@dataclass
class IntegrationPlan:
    """Plan for integrating dataset into training."""

    source_id: str
    dataset_format: str
    complexity: str
    transformations: List[TransformationSpec] = field(default_factory=list)
    estimated_records: int = 0
    schema_mapping: Dict = field(default_factory=dict)
    required_transformations: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    estimated_effort_hours: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    integration_priority: int = 0
    created_date: Optional[datetime] = None

    VALID_COMPLEXITIES = ["low", "medium", "high"]

    def validate(self) -> List[str]:
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if self.complexity not in self.VALID_COMPLEXITIES:
            errors.append(f"complexity must be one of {self.VALID_COMPLEXITIES}")
        return errors


@dataclass
class ResearchLog:
    """Log entry for research activities."""

    activity_type: str
    description: str
    outcome: str
    timestamp: datetime = field(default_factory=datetime.now)
    source_id: Optional[str] = None
    duration_minutes: int = 0

    ALLOWED_ACTIVITY_TYPES = [
        "search",
        "eval",
        "evaluation",
        "acquisition",
        "submit_request",
        "download",
        "integration",
        "integration_plan",
        "session_start",
        "phase_transition",
        "manual_intervention",
        "error",
        "system",
    ]

    def validate(self) -> List[str]:
        errors = []
        if self.activity_type not in self.ALLOWED_ACTIVITY_TYPES:
            errors.append(f"activity_type must be one of {self.ALLOWED_ACTIVITY_TYPES}")
        return errors


@dataclass
class ResearchSession:
    """Top-level research session state."""

    session_id: str
    current_phase: str = "discovery"
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    progress_metrics: Dict[str, int] = field(default_factory=dict)
    weekly_targets: Dict[str, int] = field(default_factory=dict)
    logs: List[ResearchLog] = field(default_factory=list)
    start_date: Optional[datetime] = None
    target_sources: List[str] = field(default_factory=list)
    search_keywords: Dict = field(default_factory=dict)

    VALID_PHASES = ["discovery", "evaluation", "acquisition", "integration"]

    def validate(self) -> List[str]:
        errors = []
        if not self.session_id:
            errors.append("session_id is required")
        if self.current_phase not in self.VALID_PHASES:
            errors.append(f"current_phase must be one of {self.VALID_PHASES}")
        return errors
