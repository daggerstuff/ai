"""
Data models for journal dataset research system.

This module defines all data structures used throughout the research system,
including dataset sources, evaluations, access requests, and research sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class DatasetSource:
    """Represents a discovered dataset source from academic repositories."""

    source_id: str
    title: str
    authors: List[str]
    publication_date: datetime
    source_type: str  # journal, repository, clinical_trial, training_material
    url: str
    doi: Optional[str] = None
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    open_access: bool = False
    data_availability: str = "unknown"  # available, upon_request, restricted, unknown
    discovery_date: datetime = field(default_factory=datetime.now)
    discovery_method: str = ""  # pubmed_search, doaj_manual, repository_api, citation

    def validate(self) -> List[str]:
        """Validate the dataset source data and return list of errors."""
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if not self.title:
            errors.append("title is required")
        if not self.url:
            errors.append("url is required")
        if self.source_type not in [
            "journal",
            "repository",
            "clinical_trial",
            "training_material",
        ]:
            errors.append(
                f"source_type must be one of: journal, repository, clinical_trial, training_material"
            )
        if self.data_availability not in [
            "available",
            "upon_request",
            "restricted",
            "unknown",
        ]:
            errors.append(
                f"data_availability must be one of: available, upon_request, restricted, unknown"
            )
        return errors


@dataclass
class DatasetEvaluation:
    """Represents the evaluation of a dataset across multiple dimensions."""

    source_id: str
    therapeutic_relevance: int  # 1-10
    data_structure_quality: int  # 1-10
    training_integration: int  # 1-10
    ethical_accessibility: int  # 1-10
    therapeutic_relevance_notes: str = ""
    data_structure_notes: str = ""
    integration_notes: str = ""
    ethical_notes: str = ""
    overall_score: float = 0.0
    priority_tier: str = "low"  # high, medium, low
    evaluation_date: datetime = field(default_factory=datetime.now)
    evaluator: str = ""
    competitive_advantages: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the evaluation data and return list of errors."""
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if not (1 <= self.therapeutic_relevance <= 10):
            errors.append("therapeutic_relevance must be between 1 and 10")
        if not (1 <= self.data_structure_quality <= 10):
            errors.append("data_structure_quality must be between 1 and 10")
        if not (1 <= self.training_integration <= 10):
            errors.append("training_integration must be between 1 and 10")
        if not (1 <= self.ethical_accessibility <= 10):
            errors.append("ethical_accessibility must be between 1 and 10")
        if self.priority_tier not in ["high", "medium", "low"]:
            errors.append("priority_tier must be one of: high, medium, low")
        return errors


@dataclass
class AccessRequest:
    """Represents a request to access a dataset."""

    source_id: str
    access_method: str  # direct, api, request_form, collaboration, registration
    request_date: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, approved, denied, downloaded, error
    access_url: str = ""
    credentials_required: bool = False
    institutional_affiliation_required: bool = False
    estimated_access_date: Optional[datetime] = None
    notes: str = ""

    def validate(self) -> List[str]:
        """Validate the access request data and return list of errors."""
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if self.access_method not in [
            "direct",
            "api",
            "request_form",
            "collaboration",
            "registration",
        ]:
            errors.append(
                "access_method must be one of: direct, api, request_form, collaboration, registration"
            )
        if self.status not in [
            "pending",
            "approved",
            "denied",
            "downloaded",
            "error",
        ]:
            errors.append(
                "status must be one of: pending, approved, denied, downloaded, error"
            )
        return errors


@dataclass
class AcquiredDataset:
    """Represents an acquired dataset with storage information."""

    source_id: str
    acquisition_date: datetime = field(default_factory=datetime.now)
    storage_path: str = ""
    file_format: str = ""
    file_size_mb: float = 0.0
    license: str = ""
    usage_restrictions: List[str] = field(default_factory=list)
    attribution_required: bool = False
    checksum: str = ""

    def validate(self) -> List[str]:
        """Validate the acquired dataset data and return list of errors."""
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if not self.storage_path:
            errors.append("storage_path is required")
        return errors


@dataclass
class TransformationSpec:
    """Specifies a data transformation for pipeline integration."""

    transformation_type: str  # format_conversion, field_mapping, cleaning, validation
    input_format: str
    output_format: str
    transformation_logic: str
    validation_rules: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the transformation spec and return list of errors."""
        errors = []
        if self.transformation_type not in [
            "format_conversion",
            "field_mapping",
            "cleaning",
            "validation",
        ]:
            errors.append(
                "transformation_type must be one of: format_conversion, field_mapping, cleaning, validation"
            )
        return errors


@dataclass
class IntegrationPlan:
    """Represents a plan for integrating a dataset into the training pipeline."""

    source_id: str
    dataset_format: str  # csv, json, xml, parquet, custom
    schema_mapping: Dict[str, str] = field(
        default_factory=dict
    )  # dataset_field -> pipeline_field
    required_transformations: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    complexity: str = "medium"  # low, medium, high
    estimated_effort_hours: int = 0
    dependencies: List[str] = field(default_factory=list)
    integration_priority: int = 0
    created_date: datetime = field(default_factory=datetime.now)

    def validate(self) -> List[str]:
        """Validate the integration plan and return list of errors."""
        errors = []
        if not self.source_id:
            errors.append("source_id is required")
        if self.complexity not in ["low", "medium", "high"]:
            errors.append("complexity must be one of: low, medium, high")
        return errors


@dataclass
class ResearchSession:
    """Represents a research session with targets and progress tracking."""

    session_id: str
    start_date: datetime = field(default_factory=datetime.now)
    target_sources: List[str] = field(default_factory=list)
    search_keywords: Dict[str, List[str]] = field(default_factory=dict)
    weekly_targets: Dict[str, int] = field(default_factory=dict)
    current_phase: str = "discovery"  # discovery, evaluation, acquisition, integration
    progress_metrics: Dict[str, int] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate the research session and return list of errors."""
        errors = []
        if not self.session_id:
            errors.append("session_id is required")
        if self.current_phase not in [
            "discovery",
            "evaluation",
            "acquisition",
            "integration",
        ]:
            errors.append(
                "current_phase must be one of: discovery, evaluation, acquisition, integration"
            )
        return errors


@dataclass
class ResearchProgress:
    """Tracks overall research progress metrics."""

    sources_identified: int = 0
    datasets_evaluated: int = 0
    access_established: int = 0
    datasets_acquired: int = 0
    integration_plans_created: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchLog:
    """Represents a log entry for research activities."""

    timestamp: datetime = field(default_factory=datetime.now)
    activity_type: str = ""  # search, evaluation, access_request, download, integration
    source_id: Optional[str] = None
    description: str = ""
    outcome: str = ""
    duration_minutes: int = 0

    def validate(self) -> List[str]:
        """Validate the research log and return list of errors."""
        errors = []
        if self.activity_type not in [
            "search",
            "evaluation",
            "access_request",
            "download",
            "integration",
        ]:
            errors.append(
                "activity_type must be one of: search, evaluation, access_request, download, integration"
            )
        return errors


@dataclass
class WeeklyReport:
    """Represents a weekly progress report."""

    week_number: int
    start_date: datetime
    end_date: datetime
    sources_identified: int = 0
    datasets_evaluated: int = 0
    access_established: int = 0
    datasets_acquired: int = 0
    integration_plans_created: int = 0
    key_findings: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    next_week_priorities: List[str] = field(default_factory=list)

