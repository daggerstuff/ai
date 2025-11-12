"""
Research Orchestrator

Coordinates the journal dataset research workflow across discovery, evaluation,
acquisition, and integration planning phases. Provides research session
management, progress tracking, reporting, and robust error recovery with retry
logic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

from ai.journal_dataset_research.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchLog,
    ResearchProgress,
    ResearchSession,
)
from ai.journal_dataset_research.orchestrator.mixins import (
    ProgressReportingMixin,
    RetryMixin,
    WorkflowMixin,
)
from ai.journal_dataset_research.orchestrator.types import (
    AcquisitionServiceProtocol,
    DiscoveryServiceProtocol,
    EvaluationServiceProtocol,
    IntegrationServiceProtocol,
    OrchestratorConfig,
    ProgressSnapshot,
    SessionState,
)

logger = logging.getLogger(__name__)

class ResearchOrchestrator(WorkflowMixin, ProgressReportingMixin, RetryMixin):
    """Coordinates the journal dataset research workflow."""

    PHASE_ORDER: Sequence[str] = ("discovery", "evaluation", "acquisition", "integration")

    def __init__(
        self,
        discovery_service: Optional[DiscoveryServiceProtocol] = None,
        evaluation_engine: Optional[EvaluationServiceProtocol] = None,
        acquisition_manager: Optional[AcquisitionServiceProtocol] = None,
        integration_engine: Optional[IntegrationServiceProtocol] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self.discovery_service = discovery_service
        self.evaluation_engine = evaluation_engine
        self.acquisition_manager = acquisition_manager
        self.integration_engine = integration_engine
        self.config = config or OrchestratorConfig()

        self.sessions: Dict[str, ResearchSession] = {}
        self.session_states: Dict[str, SessionState] = {}
        self.progress_states: Dict[str, ResearchProgress] = {}
        self.progress_history: Dict[str, List[ProgressSnapshot]] = {}
        self.activity_logs: Dict[str, List[ResearchLog]] = {}
        self.error_log: Dict[str, List[Dict[str, Any]]] = {}

        self._session_storage_path = self._resolve_session_storage_path(
            self.config.session_storage_path
        )
        self._lock = threading.Lock()

    def start_research_session(
        self,
        target_sources: List[str],
        search_keywords: Dict[str, List[str]],
        weekly_targets: Optional[Dict[str, int]] = None,
        session_id: Optional[str] = None,
    ) -> ResearchSession:
        """
        Initialize a new research session with provided targets and keywords.

        Returns the created `ResearchSession` instance.
        """
        session_identifier = session_id or f"session_{uuid4().hex[:8]}"
        session = ResearchSession(session_id=session_identifier)
        session.start_date = datetime.now()
        session.target_sources = target_sources
        session.search_keywords = search_keywords
        session.weekly_targets = weekly_targets or {}
        session.current_phase = self.PHASE_ORDER[0]
        session.progress_metrics = self._initial_progress_metrics()

        errors = session.validate()
        if errors:
            raise ValueError(f"Invalid research session: {', '.join(errors)}")

        with self._lock:
            self.sessions[session_identifier] = session
            self.session_states[session_identifier] = SessionState()
            self.progress_states[session_identifier] = ResearchProgress()
            self.progress_history[session_identifier] = []
            self.activity_logs[session_identifier] = []
            self.error_log[session_identifier] = []

        self.log_activity(
            session_identifier,
            activity_type="session_start",
            description="Research session initialized",
            outcome="started",
        )

        return session

    def advance_phase(self, session_id: str) -> str:
        """
        Advance the research session to the next phase in the workflow sequence.
        """
        session = self._get_session(session_id)
        current_index = self.PHASE_ORDER.index(session.current_phase)

        if current_index < len(self.PHASE_ORDER) - 1:
            session.current_phase = self.PHASE_ORDER[current_index + 1]
            session.progress_metrics["phase_transitions"] = (
                session.progress_metrics.get("phase_transitions", 0) + 1
            )
            self.log_activity(
                session_id,
                activity_type="phase_transition",
                description="Phase advanced",
                outcome=session.current_phase,
            )
        return session.current_phase

    def update_progress(self, session_id: str, metrics: Dict[str, int]) -> None:
        """Update progress metrics for a session and record a snapshot."""
        if not metrics:
            return

        self._ensure_session_tracking(session_id)
        session = self._get_session(session_id)
        progress = self.progress_states[session_id]
        now = datetime.now()

        with self._lock:
            for key, value in metrics.items():
                current_value = session.progress_metrics.get(key, 0)
                if value >= current_value:
                    new_value = value
                else:
                    new_value = current_value + value
                session.progress_metrics[key] = new_value
                if hasattr(progress, key):
                    setattr(progress, key, new_value)

            progress.last_updated = now
            snapshot = ProgressSnapshot(
                timestamp=now,
                progress=ResearchProgress(**asdict(progress)),
                metrics=dict(session.progress_metrics),
            )
            self.progress_history[session_id].append(snapshot)
            if len(self.progress_history[session_id]) > self.config.progress_history_limit:
                self.progress_history[session_id].pop(0)

    def run_session(
        self,
        session_id: str,
        evaluator: str = "system",
        auto_acquire: bool = True,
        target_format: str = "chatml",
    ) -> SessionState:
        """Execute the research workflow for the given session."""
        session = self._get_session(session_id)
        state = self.session_states[session_id]

        self._run_discovery_phase(session_id, session, state)
        self._run_evaluation_phase(session_id, state, evaluator)
        self._run_acquisition_phase(session_id, state, auto_acquire)
        self._run_integration_phase(session_id, state, target_format)

        return state

    def _run_discovery_phase(
        self, session_id: str, session: ResearchSession, state: SessionState
    ) -> None:
        if not self.discovery_service or state.sources:
            return

        self.log_activity(
            session_id, "search", "Starting source discovery", "in_progress"
        )
        try:
            sources = self._execute_with_retries(
                session_id,
                "discovery",
                self.discovery_service.discover_sources,
                session,
            )
        except Exception as exc:
            if self.config.fallback_on_failure:
                logger.warning(
                    f"Discovery failed for session {session_id}, continuing with empty sources",
                    exc_info=exc,
                )
                self.log_activity(
                    session_id,
                    "search",
                    "Source discovery failed, using fallback (empty sources)",
                    outcome=f"Fallback: {str(exc)}",
                )
                sources = []
            else:
                self.log_activity(
                    session_id,
                    "search",
                    "Source discovery failed",
                    outcome=str(exc),
                )
                raise

        state.sources = sources
        self.update_progress(session_id, {"sources_identified": len(sources)})
        self.log_activity(
            session_id,
            "search",
            "Source discovery completed",
            f"{len(sources)} sources",
        )
        self.advance_phase(session_id)

    def _run_evaluation_phase(
        self, session_id: str, state: SessionState, evaluator: str
    ) -> None:
        if (
            not self.evaluation_engine
            or not state.sources
            or state.evaluations
        ):
            return

        self.log_activity(
            session_id, "evaluation", "Evaluating dataset sources", "in_progress"
        )
        evaluations = self._evaluate_sources(session_id, state.sources, evaluator)
        state.evaluations.extend(evaluations)
        if evaluations:
            self.update_progress(
                session_id, {"datasets_evaluated": len(state.evaluations)}
            )
        self.log_activity(
            session_id,
            "evaluation",
            "Evaluation complete",
            f"{len(evaluations)} datasets evaluated",
        )
        self.advance_phase(session_id)

    def _run_acquisition_phase(
        self, session_id: str, state: SessionState, auto_acquire: bool
    ) -> None:
        if (
            not auto_acquire
            or not self.acquisition_manager
            or not state.sources
            or state.acquired_datasets
        ):
            return

        self.log_activity(
            session_id, "acquisition", "Starting dataset acquisition", "in_progress"
        )
        self._acquire_datasets(session_id, state)

        if state.access_requests:
            self.update_progress(
                session_id, {"access_established": len(state.access_requests)}
            )
        if state.acquired_datasets:
            self.update_progress(
                session_id, {"datasets_acquired": len(state.acquired_datasets)}
            )

        outcome = (
            f"{len(state.acquired_datasets)} datasets acquired"
            if state.acquired_datasets
            else "Dataset acquisition completed with no downloads"
        )
        self.log_activity(
            session_id,
            "acquisition",
            "Dataset acquisition complete",
            outcome,
        )
        self.advance_phase(session_id)

    def _run_integration_phase(
        self, session_id: str, state: SessionState, target_format: str
    ) -> None:
        if (
            not self.integration_engine
            or not state.acquired_datasets
            or state.integration_plans
        ):
            return

        self.log_activity(
            session_id,
            "integration",
            "Generating integration plans",
            "in_progress",
        )
        plans = self._create_integration_plans(session_id, state, target_format)
        state.integration_plans.extend(plans)
        if plans:
            self.update_progress(
                session_id, {"integration_plans_created": len(state.integration_plans)}
            )
        self.log_activity(
            session_id,
            "integration",
            "Integration planning complete",
            f"{len(plans)} plans created",
        )

    def get_session_state(self, session_id: str) -> SessionState:
        """Return accumulated state for the session."""
        self._ensure_session_tracking(session_id)
        return self.session_states[session_id]

    def get_progress(self, session_id: str) -> ResearchProgress:
        """Return current progress metrics for the session."""
        self._ensure_session_tracking(session_id)
        return self.progress_states[session_id]

    def get_progress_history(self, session_id: str) -> List[ProgressSnapshot]:
        """Return recorded progress history for the session."""
        self._ensure_session_tracking(session_id)
        return list(self.progress_history.get(session_id, []))

    def get_activity_log(self, session_id: str) -> List[ResearchLog]:
        """Retrieve recorded activity log entries for a session."""
        self._ensure_session_tracking(session_id)
        return list(self.activity_logs.get(session_id, []))

    def get_error_log(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve error log entries for a session."""
        self._ensure_session_tracking(session_id)
        return list(self.error_log.get(session_id, []))

    def save_session_state(
        self,
        session_id: str,
        directory: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Persist the current session, workflow state, progress metrics, and logs to disk.

        Returns the path to the persisted session bundle.
        """
        self._ensure_session_tracking(session_id)
        bundle = self._build_session_bundle(session_id)
        storage_dir = self._resolve_session_storage_path(
            directory or self._session_storage_path
        )
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / f"{session_id}.json"
        file_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        return file_path

    def load_session_state(
        self,
        session_id: str,
        directory: Optional[Union[str, Path]] = None,
    ) -> ResearchSession:
        """
        Load a previously persisted session bundle and restore orchestrator state.

        Returns the restored `ResearchSession`.
        """
        storage_dir = self._resolve_session_storage_path(
            directory or self._session_storage_path
        )
        file_path = storage_dir / f"{session_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Persisted session not found at {file_path}")

        bundle = json.loads(file_path.read_text(encoding="utf-8"))
        bundle = self._restore_datetimes(bundle)

        (
            session,
            state,
            progress,
            history,
            activity_logs,
            error_entries,
        ) = self._restore_session_components(bundle)

        with self._lock:
            self.sessions[session_id] = session
            self.session_states[session_id] = state
            self.progress_states[session_id] = progress
            self.progress_history[session_id] = history[
                -self.config.progress_history_limit :
            ]
            self.activity_logs[session_id] = activity_logs
            self.error_log[session_id] = error_entries

        return session

    def generate_progress_visualization_data(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produce chart-ready progress data with weekly target coverage.

        Returns a dictionary containing time-series metrics, current progress,
        and target completion percentages.
        """
        self._ensure_session_tracking(session_id)
        session = self._get_session(session_id)
        progress = self.progress_states[session_id]
        max_points = (
            limit if limit is not None else self.config.visualization_max_points
        )
        series = self._build_visualization_series(session_id, session, max_points)
        targets = self._calculate_target_progress(session)

        return {
            "session_id": session.session_id,
            "current_phase": session.current_phase,
            "series": series,
            "targets": targets,
            "current_metrics": dict(session.progress_metrics),
            "last_updated": progress.last_updated.isoformat()
            if progress.last_updated
            else None,
        }

    def log_activity(
        self,
        session_id: str,
        activity_type: str,
        description: str,
        outcome: str,
        source_id: Optional[str] = None,
        duration_minutes: int = 0,
    ) -> None:
        """Record a research activity entry for the session."""
        log_entry = ResearchLog(
            timestamp=datetime.now(),
            activity_type=activity_type,
            source_id=source_id,
            description=description,
            outcome=outcome,
            duration_minutes=duration_minutes,
        )

        errors = log_entry.validate()
        if errors:
            raise ValueError(f"Invalid research log entry: {', '.join(errors)}")

        with self._lock:
            self.activity_logs.setdefault(session_id, []).append(log_entry)

    def record_manual_intervention(
        self,
        session_id: str,
        description: str,
        outcome: str,
        source_id: Optional[str] = None,
    ) -> None:
        """Record a manual intervention action for auditability."""
        self.log_activity(
            session_id=session_id,
            activity_type="manual_intervention",
            description=description,
            outcome=outcome,
            source_id=source_id,
        )

    def _evaluate_sources(
        self, session_id: str, sources: List[DatasetSource], evaluator: str
    ) -> List[DatasetEvaluation]:
        """Evaluate dataset sources, optionally using parallel execution."""
        if not self.evaluation_engine or not sources:
            return []

        if self.config.parallel_evaluation and len(sources) > 1:
            return self._evaluate_sources_parallel(session_id, sources, evaluator)

        evaluations: List[DatasetEvaluation] = []
        for source in sources:
            evaluation = self._evaluate_single_source(session_id, source, evaluator)
            if evaluation:
                evaluations.append(evaluation)
        return evaluations

    def _evaluate_sources_parallel(
        self, session_id: str, sources: List[DatasetSource], evaluator: str
    ) -> List[DatasetEvaluation]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        evaluations: List[DatasetEvaluation] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_single_source, session_id, source, evaluator
                ): source
                for source in sources
            }
            for future in as_completed(futures):
                evaluation = future.result()
                if evaluation:
                    evaluations.append(evaluation)
        return evaluations

    def _evaluate_single_source(
        self, session_id: str, source: DatasetSource, evaluator: str
    ) -> Optional[DatasetEvaluation]:
        engine = self.evaluation_engine
        if engine is None:
            return None
        try:
            return self._execute_with_retries(
                session_id,
                f"evaluate:{source.source_id}",
                engine.evaluate_dataset,
                source,
                evaluator,
            )
        except Exception as exc:  # pragma: no cover - retry logic tested separately
            self._log_error(
                session_id, f"evaluate:{source.source_id}", exc, self.config.max_retries
            )
            return None

    def _acquire_datasets(self, session_id: str, state: SessionState) -> None:
        """Handle access requests and dataset downloads."""
        if not self.acquisition_manager or not state.sources:
            return

        for source in state.sources:
            try:
                access_request = self._execute_with_retries(
                    session_id,
                    f"access_request:{source.source_id}",
                    self.acquisition_manager.submit_access_request,
                    source,
                )
                state.access_requests.append(access_request)
            except Exception as exc:
                self._log_error(session_id, f"access_request:{source.source_id}", exc, self.config.max_retries)
                continue

            try:
                acquired_dataset = self._execute_with_retries(
                    session_id,
                    f"download:{source.source_id}",
                    self.acquisition_manager.download_dataset,
                    source,
                    access_request,
                )
                state.acquired_datasets.append(acquired_dataset)
            except Exception as exc:
                self._log_error(session_id, f"download:{source.source_id}", exc, self.config.max_retries)
                self.log_activity(
                    session_id,
                    "acquisition",
                    f"Download failed for {source.source_id}",
                    outcome=str(exc),
                    source_id=source.source_id,
                )

    def _create_integration_plans(
        self, session_id: str, state: SessionState, target_format: str
    ) -> List[IntegrationPlan]:
        """Create integration plans for acquired datasets."""
        if not self.integration_engine or not state.acquired_datasets:
            return []

        datasets = state.acquired_datasets
        if self.config.parallel_integration_planning and len(datasets) > 1:
            return self._create_plans_parallel(session_id, state, datasets, target_format)

        plans: List[IntegrationPlan] = []
        for dataset in datasets:
            plan = self._create_single_integration_plan(
                session_id, state, dataset, target_format
            )
            if plan:
                plans.append(plan)
        return plans

    def _create_plans_parallel(
        self,
        session_id: str,
        state: SessionState,
        datasets: List[AcquiredDataset],
        target_format: str,
    ) -> List[IntegrationPlan]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        plans: List[IntegrationPlan] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._create_single_integration_plan,
                    session_id,
                    state,
                    dataset,
                    target_format,
                ): dataset
                for dataset in datasets
            }
            for future in as_completed(futures):
                plan = future.result()
                if plan:
                    plans.append(plan)
        return plans

    def _create_single_integration_plan(
        self,
        session_id: str,
        state: SessionState,
        dataset: AcquiredDataset,
        target_format: str,
    ) -> Optional[IntegrationPlan]:
        engine = self.integration_engine
        if engine is None:
            return None
        try:
            plan = self._execute_with_retries(
                session_id,
                f"integration_plan:{dataset.source_id}",
                engine.create_integration_plan,
                dataset,
                target_format,
            )
        except Exception as exc:  # pragma: no cover - retry logic tested separately
            self._log_error(
                session_id,
                f"integration_plan:{dataset.source_id}",
                exc,
                self.config.max_retries,
            )
            return None

        feasible = True
        if hasattr(engine, "validate_integration_feasibility"):
            try:
                feasible = engine.validate_integration_feasibility(plan)
            except Exception as exc:  # pragma: no cover - retry logic tested separately
                feasible = False
                self._log_error(
                    session_id,
                    f"integration_validation:{dataset.source_id}",
                    exc,
                    self.config.max_retries,
                )
        state.integration_feasibility[dataset.source_id] = feasible
        return plan

    def _execute_with_retries(
        self,
        session_id: str,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Execute an operation with retry logic, exponential backoff, and error logging."""
        attempts = 0
        last_exception: Optional[Exception] = None

        while attempts < self.config.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - exception details tested separately
                attempts += 1
                last_exception = exc
                self._log_error(session_id, operation, exc, attempts)
                if attempts >= self.config.max_retries:
                    break

                # Exponential backoff: delay = base_delay * (2 ^ (attempt - 1))
                if self.config.retry_delay_seconds > 0:
                    delay = self.config.retry_delay_seconds * (2 ** (attempts - 1))
                    logger.debug(
                        f"Retrying {operation} after {delay:.2f}s (attempt {attempts}/{self.config.max_retries})"
                    )
                    time.sleep(delay)

        assert last_exception is not None
        raise last_exception

    def _log_error(
        self,
        session_id: str,
        operation: str,
        error: Exception,
        attempt: int,
    ) -> None:
        """Record an error entry for diagnostics and retry tracking."""
        entry = {
            "timestamp": datetime.now(),
            "operation": operation,
            "attempt": str(attempt),
            "message": str(error),
        }
        with self._lock:
            self.error_log.setdefault(session_id, []).append(entry)
        logger.warning(
            "Research orchestrator error",
            extra={"session_id": session_id, "operation": operation, "attempt": attempt, "error": error},
        )

    def _build_visualization_series(
        self,
        session_id: str,
        session: ResearchSession,
        max_points: int,
    ) -> List[Dict[str, Any]]:
        history = self.progress_history.get(session_id, [])
        if max_points <= 0:
            max_points = len(history) or 1

        series = [
            {
                "timestamp": snapshot.timestamp.isoformat(),
                "metrics": dict(snapshot.metrics),
            }
            for snapshot in history[-max_points:]
        ]

        if not series:
            series.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": dict(session.progress_metrics),
                }
            )

        return series

    def _calculate_target_progress(
        self, session: ResearchSession
    ) -> Dict[str, Dict[str, Any]]:
        targets: Dict[str, Dict[str, Any]] = {}
        for key, target in session.weekly_targets.items():
            achieved = session.progress_metrics.get(key, 0)
            percent_complete = (achieved / target * 100.0) if target else 0.0
            targets[key] = {
                "achieved": achieved,
                "target": target,
                "percent_complete": round(min(percent_complete, 100.0), 2),
                "remaining": max(0, target - achieved),
            }
        return targets

    def _build_session_bundle(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions[session_id]
        state = self.session_states[session_id]
        progress = self.progress_states[session_id]
        history = self.progress_history.get(session_id, [])
        activity_logs = self.activity_logs.get(session_id, [])
        error_entries = self.error_log.get(session_id, [])

        bundle = {
            "session": self._convert_datetimes(asdict(session)),
            "state": self._convert_datetimes(asdict(state)),
            "progress": self._convert_datetimes(asdict(progress)),
            "progress_history": [
                self._convert_datetimes(asdict(snapshot)) for snapshot in history
            ],
            "activity_logs": [
                self._convert_datetimes(asdict(entry)) for entry in activity_logs
            ],
            "error_log": self._convert_datetimes(error_entries),
        }
        return bundle

    def _restore_session_components(
        self, bundle: Dict[str, Any]
    ) -> Tuple[
        ResearchSession,
        SessionState,
        ResearchProgress,
        List[ProgressSnapshot],
        List[ResearchLog],
        List[Dict[str, Any]],
    ]:
        session_data = dict(bundle.get("session", {}))
        session = ResearchSession(**session_data)
        session.validate()

        state_data = bundle.get("state", {})
        state = SessionState(
            sources=[
                DatasetSource(**item) for item in state_data.get("sources", [])
            ],
            evaluations=[
                DatasetEvaluation(**item)
                for item in state_data.get("evaluations", [])
            ],
            access_requests=[
                AccessRequest(**item)
                for item in state_data.get("access_requests", [])
            ],
            acquired_datasets=[
                AcquiredDataset(**item)
                for item in state_data.get("acquired_datasets", [])
            ],
            integration_plans=[
                IntegrationPlan(**item)
                for item in state_data.get("integration_plans", [])
            ],
            integration_feasibility=state_data.get("integration_feasibility", {}),
        )

        progress = ResearchProgress(**bundle.get("progress", {}))

        history: List[ProgressSnapshot] = []
        for entry in bundle.get("progress_history", []):
            history.append(
                ProgressSnapshot(
                    timestamp=entry["timestamp"],
                    progress=ResearchProgress(**entry.get("progress", {})),
                    metrics=dict(entry.get("metrics", {})),
                )
            )

        activity_logs = [
            ResearchLog(**log_entry) for log_entry in bundle.get("activity_logs", [])
        ]

        error_entries: List[Dict[str, Any]] = []
        for error_entry in bundle.get("error_log", []):
            error_entries.append(
                {
                    "timestamp": error_entry.get("timestamp"),
                    "operation": error_entry.get("operation", ""),
                    "attempt": str(error_entry.get("attempt", "")),
                    "message": error_entry.get("message", ""),
                }
            )

        return session, state, progress, history, activity_logs, error_entries

    @staticmethod
    def _resolve_session_storage_path(
        path: Optional[Union[str, Path]]
    ) -> Path:
        if path is None:
            return Path.cwd() / "ai" / "journal_dataset_research" / "sessions"
        return Path(path).expanduser()

    @staticmethod
    def _convert_datetimes(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, list):
            return [ResearchOrchestrator._convert_datetimes(item) for item in value]
        if isinstance(value, dict):
            return {
                key: ResearchOrchestrator._convert_datetimes(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _restore_datetimes(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value
        if isinstance(value, list):
            return [ResearchOrchestrator._restore_datetimes(item) for item in value]
        if isinstance(value, dict):
            return {
                key: ResearchOrchestrator._restore_datetimes(item)
                for key, item in value.items()
            }
        return value

    def _normalize_report_window(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> Tuple[datetime, datetime]:
        if start_date is None or end_date is None:
            resolved_end = datetime.now()
            resolved_start = resolved_end - timedelta(days=7)
        else:
            resolved_start, resolved_end = start_date, end_date
        return resolved_start, resolved_end

    def _collect_progress_history(
        self, session_id: str, start_date: datetime, end_date: datetime
    ) -> List[ProgressSnapshot]:
        history = self.progress_history.get(session_id, [])
        return [
            snapshot
            for snapshot in history
            if start_date <= snapshot.timestamp <= end_date
        ]

    def _resolve_progress_bounds(
        self,
        session_id: str,
        history: List[ProgressSnapshot],
        end_date: datetime,
    ) -> Tuple[ResearchProgress, ResearchProgress]:
        if history:
            return history[0].progress, history[-1].progress

        current = ResearchProgress(**asdict(self.progress_states[session_id]))
        return current, current

    @staticmethod
    def _compute_progress_deltas(
        start_progress: ResearchProgress, end_progress: ResearchProgress
    ) -> Dict[str, int]:
        deltas = {
            "sources_identified": end_progress.sources_identified - start_progress.sources_identified,
            "datasets_evaluated": end_progress.datasets_evaluated - start_progress.datasets_evaluated,
            "access_established": end_progress.access_established - start_progress.access_established,
            "datasets_acquired": end_progress.datasets_acquired - start_progress.datasets_acquired,
            "integration_plans_created": end_progress.integration_plans_created
            - start_progress.integration_plans_created,
        }
        return {key: max(0, value) for key, value in deltas.items()}

    @staticmethod
    def _build_key_findings(deltas: Dict[str, int]) -> List[str]:
        findings: List[str] = []
        if deltas["sources_identified"]:
            findings.append(f"Identified {deltas['sources_identified']} new dataset sources")
        if deltas["datasets_evaluated"]:
            findings.append(f"Completed evaluations for {deltas['datasets_evaluated']} datasets")
        if deltas["integration_plans_created"]:
            findings.append(f"Produced {deltas['integration_plans_created']} integration plans")

        if not findings:
            findings.append("Maintained research infrastructure with no new datasets processed")
        return findings

    def _build_challenges(
        self, session_id: str, start_date: datetime, end_date: datetime
    ) -> List[str]:
        errors = [
            entry
            for entry in self.error_log.get(session_id, [])
            if start_date <= entry["timestamp"] <= end_date
        ]
        if not errors:
            return []
        return [f"Encountered {len(errors)} errors requiring manual review"]

    def _build_next_week_priorities(
        self, session: ResearchSession, deltas: Dict[str, int]
    ) -> List[str]:
        priorities: List[str] = []
        if session.weekly_targets:
            for key, target in session.weekly_targets.items():
                achieved = session.progress_metrics.get(key, 0)
                if achieved < target:
                    remaining = target - achieved
                    priorities.append(
                        f"Complete remaining {remaining} {key.replace('_', ' ')} to meet weekly target"
                    )

        if not priorities:
            if deltas["datasets_acquired"]:
                priorities.append("Plan integration workflows for newly acquired datasets")
            else:
                priorities.append("Review high-priority datasets and prepare next acquisition wave")

        return priorities

    def _ensure_session_tracking(self, session_id: str) -> None:
        """
        Ensure tracking structures exist for a session.

        Creates default state, progress, history, and logs if they haven't been
        initialized yet. Raises KeyError if the session itself does not exist.
        """
        self._get_session(session_id)
        with self._lock:
            if session_id not in self.session_states:
                self.session_states[session_id] = SessionState()
            if session_id not in self.progress_states:
                self.progress_states[session_id] = ResearchProgress()
            if session_id not in self.progress_history:
                self.progress_history[session_id] = []
            if session_id not in self.activity_logs:
                self.activity_logs[session_id] = []
            if session_id not in self.error_log:
                self.error_log[session_id] = []

    def _get_session(self, session_id: str) -> ResearchSession:
        """Retrieve a session by ID, raising if it does not exist."""
        if session_id not in self.sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self.sessions[session_id]

    @staticmethod
    def _initial_progress_metrics() -> Dict[str, int]:
        """Create initial progress metrics dictionary."""
        return {
            "sources_identified": 0,
            "datasets_evaluated": 0,
            "access_established": 0,
            "datasets_acquired": 0,
            "integration_plans_created": 0,
        }

