"""
Shared type definitions for the Research Orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ai.sourcing.journal.acquisition.acquisition_manager import DownloadProgress
from ai.sourcing.journal.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchProgress,
    ResearchSession,
)


class DiscoveryServiceProtocol(Protocol):
    """Protocol for dataset discovery services."""

    def discover_sources(self, session: ResearchSession) -> list[DatasetSource]:
        """Discover dataset sources for the given research session."""
        ...


class EvaluationServiceProtocol(Protocol):
    """Protocol for dataset evaluation services."""

    def evaluate_dataset(self, source: DatasetSource, evaluator: str = "system") -> DatasetEvaluation:
        """Evaluate a dataset source and return evaluation results."""
        ...


class AcquisitionServiceProtocol(Protocol):
    """Protocol for dataset acquisition services."""

    def submit_access_request(
        self,
        source: DatasetSource,
        access_method: str | None = None,
        notes: str = "",
    ) -> AccessRequest:
        """Submit an access request for a dataset source."""
        ...

    def download_dataset(
        self,
        source: DatasetSource,
        access_request: AccessRequest | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> AcquiredDataset:
        """Download a dataset using the provided access request."""
        ...


class IntegrationServiceProtocol(Protocol):
    """Protocol for integration planning services."""

    def create_integration_plan(self, dataset: AcquiredDataset, target_format: str = "chatml") -> IntegrationPlan:
        """Create an integration plan for the acquired dataset."""
        ...

    def validate_integration_feasibility(self, plan: IntegrationPlan) -> bool:
        """Validate whether the integration plan is feasible."""
        ...


@dataclass
class SessionState:
    """Maintains state accumulated during the research workflow."""

    sources: list[DatasetSource] = field(default_factory=list)
    evaluations: list[DatasetEvaluation] = field(default_factory=list)
    access_requests: list[AccessRequest] = field(default_factory=list)
    acquired_datasets: list[AcquiredDataset] = field(default_factory=list)
    integration_plans: list[IntegrationPlan] = field(default_factory=list)
    integration_feasibility: dict[str, bool] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """Historical snapshot of research progress metrics."""

    timestamp: datetime
    progress: ResearchProgress
    metrics: dict[str, int]


@dataclass
class OrchestratorConfig:
    """Configuration options for the research orchestrator."""

    max_retries: int = 3
    retry_delay_seconds: float = 0.0
    progress_history_limit: int = 100
    parallel_evaluation: bool = False
    parallel_integration_planning: bool = False
    max_workers: int = 4
    session_storage_path: Path | None = None
    visualization_max_points: int = 100
    fallback_on_failure: bool = True
