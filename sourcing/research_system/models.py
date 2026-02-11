"""
Core data models for the journal dataset research system.

This module defines all data structures used throughout the research workflow,
including dataset sources, evaluations, access requests, and integration plans.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DatasetSource:
    """Represents a potential therapeutic dataset source from academic literature."""
    
    source_id: str
    title: str
    authors: list[str]
    publication_date: datetime
    source_type: str  # journal, repository, clinical_trial, training_material
    url: str
    doi: Optional[str]
    abstract: str
    keywords: list[str]
    open_access: bool
    data_availability: str  # available, upon_request, restricted, unknown
    discovery_date: datetime
    discovery_method: str  # pubmed_search, doaj_manual, repository_api, citation
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the dataset source data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.source_id:
            errors.append("source_id is required")
        
        if not self.title:
            errors.append("title is required")
        
        if not self.authors:
            errors.append("authors list cannot be empty")
        
        valid_source_types = ["journal", "repository", "clinical_trial", "training_material"]
        if self.source_type not in valid_source_types:
            errors.append(f"source_type must be one of {valid_source_types}")
        
        if not self.url:
            errors.append("url is required")
        
        valid_availability = ["available", "upon_request", "restricted", "unknown"]
        if self.data_availability not in valid_availability:
            errors.append(f"data_availability must be one of {valid_availability}")
        
        valid_discovery_methods = ["pubmed_search", "doaj_manual", "repository_api", "citation"]
        if self.discovery_method not in valid_discovery_methods:
            errors.append(f"discovery_method must be one of {valid_discovery_methods}")
        
        return len(errors) == 0, errors


@dataclass
class DatasetEvaluation:
    """Quality assessment and scoring for a dataset source."""
    
    source_id: str
    therapeutic_relevance: int  # 1-10
    therapeutic_relevance_notes: str
    data_structure_quality: int  # 1-10
    data_structure_notes: str
    training_integration: int  # 1-10
    integration_notes: str
    ethical_accessibility: int  # 1-10
    ethical_notes: str
    overall_score: float
    priority_tier: str  # high, medium, low
    evaluation_date: datetime
    evaluator: str
    competitive_advantages: list[str] = field(default_factory=list)
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the evaluation data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.source_id:
            errors.append("source_id is required")
        
        # Validate score ranges
        score_fields = [
            ("therapeutic_relevance", self.therapeutic_relevance),
            ("data_structure_quality", self.data_structure_quality),
            ("training_integration", self.training_integration),
            ("ethical_accessibility", self.ethical_accessibility),
        ]
        
        for field_name, score in score_fields:
            if not 1 <= score <= 10:
                errors.append(f"{field_name} must be between 1 and 10")
        
        if not 0 <= self.overall_score <= 10:
            errors.append("overall_score must be between 0 and 10")
        
        valid_tiers = ["high", "medium", "low"]
        if self.priority_tier not in valid_tiers:
            errors.append(f"priority_tier must be one of {valid_tiers}")
        
        if not self.evaluator:
            errors.append("evaluator is required")
        
        return len(errors) == 0, errors
    
    def calculate_overall_score(self) -> float:
        """
        Calculate the weighted overall score based on evaluation dimensions.
        
        Returns:
            Weighted overall score (0-10)
        """
        weights = {
            'therapeutic_relevance': 0.35,
            'data_structure_quality': 0.25,
            'training_integration': 0.20,
            'ethical_accessibility': 0.20
        }
        
        overall = (
            self.therapeutic_relevance * weights['therapeutic_relevance'] +
            self.data_structure_quality * weights['data_structure_quality'] +
            self.training_integration * weights['training_integration'] +
            self.ethical_accessibility * weights['ethical_accessibility']
        )
        
        return round(overall, 2)


@dataclass
class AccessRequest:
    """Tracks dataset access requests and their status."""
    
    source_id: str
    access_method: str  # direct, api, request_form, collaboration, registration
    request_date: datetime
    status: str  # pending, approved, denied, downloaded, error
    access_url: str
    credentials_required: bool
    institutional_affiliation_required: bool
    estimated_access_date: Optional[datetime]
    notes: str = ""
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the access request data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.source_id:
            errors.append("source_id is required")
        
        valid_methods = ["direct", "api", "request_form", "collaboration", "registration"]
        if self.access_method not in valid_methods:
            errors.append(f"access_method must be one of {valid_methods}")
        
        valid_statuses = ["pending", "approved", "denied", "downloaded", "error"]
        if self.status not in valid_statuses:
            errors.append(f"status must be one of {valid_statuses}")
        
        if not self.access_url:
            errors.append("access_url is required")
        
        return len(errors) == 0, errors


@dataclass
class AcquiredDataset:
    """Represents a successfully acquired dataset with storage metadata."""
    
    source_id: str
    acquisition_date: datetime
    storage_path: str
    file_format: str
    file_size_mb: float
    license: str
    usage_restrictions: list[str]
    attribution_required: bool
    checksum: str
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the acquired dataset data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.source_id:
            errors.append("source_id is required")
        
        if not self.storage_path:
            errors.append("storage_path is required")
        
        if not self.file_format:
            errors.append("file_format is required")
        
        if self.file_size_mb <= 0:
            errors.append("file_size_mb must be positive")
        
        if not self.license:
            errors.append("license is required")
        
        if not self.checksum:
            errors.append("checksum is required")
        
        return len(errors) == 0, errors


@dataclass
class TransformationSpec:
    """Specification for data transformation operations."""
    
    transformation_type: str  # format_conversion, field_mapping, cleaning, validation
    input_format: str
    output_format: str
    transformation_logic: str
    validation_rules: list[str]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the transformation specification.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        valid_types = ["format_conversion", "field_mapping", "cleaning", "validation"]
        if self.transformation_type not in valid_types:
            errors.append(f"transformation_type must be one of {valid_types}")
        
        if not self.input_format:
            errors.append("input_format is required")
        
        if not self.output_format:
            errors.append("output_format is required")
        
        if not self.transformation_logic:
            errors.append("transformation_logic is required")
        
        return len(errors) == 0, errors


@dataclass
class IntegrationPlan:
    """Plan for integrating a dataset into the training pipeline."""
    
    source_id: str
    dataset_format: str  # csv, json, xml, parquet, custom
    schema_mapping: dict[str, str]  # dataset_field -> pipeline_field
    required_transformations: list[str]
    preprocessing_steps: list[str]
    complexity: str  # low, medium, high
    estimated_effort_hours: int
    dependencies: list[str]
    integration_priority: int
    created_date: datetime
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the integration plan.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.source_id:
            errors.append("source_id is required")
        
        valid_formats = ["csv", "json", "xml", "parquet", "custom"]
        if self.dataset_format not in valid_formats:
            errors.append(f"dataset_format must be one of {valid_formats}")
        
        valid_complexity = ["low", "medium", "high"]
        if self.complexity not in valid_complexity:
            errors.append(f"complexity must be one of {valid_complexity}")
        
        if self.estimated_effort_hours < 0:
            errors.append("estimated_effort_hours must be non-negative")
        
        if self.integration_priority < 1:
            errors.append("integration_priority must be at least 1")
        
        return len(errors) == 0, errors


@dataclass
class ResearchProgress:
    """Tracks progress metrics for the research workflow."""
    
    sources_identified: int
    datasets_evaluated: int
    access_established: int
    datasets_acquired: int
    integration_plans_created: int
    last_updated: datetime
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the research progress data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # All counts must be non-negative
        counts = [
            ("sources_identified", self.sources_identified),
            ("datasets_evaluated", self.datasets_evaluated),
            ("access_established", self.access_established),
            ("datasets_acquired", self.datasets_acquired),
            ("integration_plans_created", self.integration_plans_created),
        ]
        
        for field_name, count in counts:
            if count < 0:
                errors.append(f"{field_name} must be non-negative")
        
        return len(errors) == 0, errors


@dataclass
class ResearchSession:
    """Represents a research session with targets and current state."""
    
    session_id: str
    start_date: datetime
    target_sources: list[str]
    search_keywords: dict[str, list[str]]
    weekly_targets: dict[str, int]
    current_phase: str  # discovery, evaluation, acquisition, integration
    progress_metrics: dict[str, int]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the research session data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.session_id:
            errors.append("session_id is required")
        
        if not self.target_sources:
            errors.append("target_sources cannot be empty")
        
        valid_phases = ["discovery", "evaluation", "acquisition", "integration"]
        if self.current_phase not in valid_phases:
            errors.append(f"current_phase must be one of {valid_phases}")
        
        return len(errors) == 0, errors


@dataclass
class ResearchLog:
    """Log entry for research activities."""
    
    timestamp: datetime
    activity_type: str  # search, evaluation, access_request, download, integration
    source_id: Optional[str]
    description: str
    outcome: str
    duration_minutes: int
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the research log entry.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        valid_types = ["search", "evaluation", "access_request", "download", "integration"]
        if self.activity_type not in valid_types:
            errors.append(f"activity_type must be one of {valid_types}")
        
        if not self.description:
            errors.append("description is required")
        
        if not self.outcome:
            errors.append("outcome is required")
        
        if self.duration_minutes < 0:
            errors.append("duration_minutes must be non-negative")
        
        return len(errors) == 0, errors


@dataclass
class WeeklyReport:
    """Weekly progress report with metrics and findings."""
    
    week_number: int
    start_date: datetime
    end_date: datetime
    sources_identified: int
    datasets_evaluated: int
    access_established: int
    datasets_acquired: int
    integration_plans_created: int
    key_findings: list[str]
    challenges: list[str]
    next_week_priorities: list[str]
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the weekly report data.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if self.week_number < 1:
            errors.append("week_number must be at least 1")
        
        if self.end_date <= self.start_date:
            errors.append("end_date must be after start_date")
        
        # All counts must be non-negative
        counts = [
            ("sources_identified", self.sources_identified),
            ("datasets_evaluated", self.datasets_evaluated),
            ("access_established", self.access_established),
            ("datasets_acquired", self.datasets_acquired),
            ("integration_plans_created", self.integration_plans_created),
        ]
        
        for field_name, count in counts:
            if count < 0:
                errors.append(f"{field_name} must be non-negative")
        
        return len(errors) == 0, errors
