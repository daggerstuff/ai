"""
Shared type definitions for the Research Orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

from ai.journal_dataset_research.acquisition.acquisition_manager import DownloadProgress
from ai.journal_dataset_research.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchSession,
    ResearchProgress,
)


class DiscoveryServiceProtocol(Protocol):
    """Protocol for dataset discovery services."""

    def discover_sources(self, session: ResearchSession) -> List[DatasetSource]:
        """Discover dataset sources for the given research session."""
        ...


class EvaluationServiceProtocol(Protocol):
    """Protocol for dataset evaluation services."""

    def evaluate_dataset(
        self, source: DatasetSource, evaluator: str = "system"
    ) -> DatasetEvaluation:
        """Evaluate a dataset source and return evaluation results."""
        ...


class AcquisitionServiceProtocol(Protocol):
    """Protocol for dataset acquisition services."""

    def submit_access_request(
        self,
        source: DatasetSource,
        access_method: Optional[str] = None,
        notes: str = "",
    ) -> AccessRequest:
        """Submit an access request for a dataset source."""
        ...

    def download_dataset(
        self,
        source: DatasetSource,
        access_request: Optional[AccessRequest] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> AcquiredDataset:
        """Download a dataset using the provided access request."""
        ...


class IntegrationServiceProtocol(Protocol):
    """Protocol for integration planning services."""

    def create_integration_plan(
        self, dataset: AcquiredDataset, target_format: str = "chatml"
    ) -> IntegrationPlan:
        """Create an integration plan for the acquired dataset."""
        ...

    def validate_integration_feasibility(self, plan: IntegrationPlan) -> bool:
        """Validate whether the integration plan is feasible."""
        ...


@dataclass
class SessionState:
    """Maintains state accumulated during the research workflow."""

    sources: List[DatasetSource] = field(default_factory=list)
    evaluations: List[DatasetEvaluation] = field(default_factory=list)
    access_requests: List[AccessRequest] = field(default_factory=list)
    acquired_datasets: List[AcquiredDataset] = field(default_factory=list)
    integration_plans: List[IntegrationPlan] = field(default_factory=list)
    integration_feasibility: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """Historical snapshot of research progress metrics."""

    timestamp: datetime
    progress: ResearchProgress
    metrics: Dict[str, int]


@dataclass
class OrchestratorConfig:
    """Configuration options for the research orchestrator."""

    max_retries: int = 3
    retry_delay_seconds: float = 0.0
    progress_history_limit: int = 100
    parallel_evaluation: bool = False
    parallel_integration_planning: bool = False
    max_workers: int = 4
    session_storage_path: Optional[Path] = None
    visualization_max_points: int = 100
    fallback_on_failure: bool = True

