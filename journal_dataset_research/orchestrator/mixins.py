"""
Utility mixins used by the Research Orchestrator.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchProgress,
    WeeklyReport,
)
from ai.journal_dataset_research.orchestrator.types import (
    ProgressSnapshot,
    SessionState,
)

logger = logging.getLogger(__name__)


class RetryMixin:
    """Provides retry handling with error logging."""

    config: Any
    error_log: Dict[str, List[Dict[str, Any]]]
    _lock: Any

    def _execute_with_retries(
        self,
        session_id: str,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
        attempts = 0
        last_exception: Optional[Exception] = None

        while attempts < self.config.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - logging path
                attempts += 1
                last_exception = exc
                self._log_error(session_id, operation, exc, attempts)
                if attempts >= self.config.max_retries:
                    break

        assert last_exception is not None
        raise last_exception

    def _log_error(
        self,
        session_id: str,
        operation: str,
        error: Exception,
        attempt: int,
    ) -> None:
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
            extra={
                "session_id": session_id,
                "operation": operation,
                "attempt": attempt,
                "error": error,
            },
        )


class ProgressReportingMixin(RetryMixin):
    """Implements progress and weekly reporting helpers."""

    sessions: Dict[str, Any]
    progress_states: Dict[str, ResearchProgress]
    progress_history: Dict[str, List[ProgressSnapshot]]
    activity_logs: Dict[str, List[Any]]

    def generate_progress_report(self, session_id: str) -> str:
        session = self.sessions[session_id]
        progress = self.progress_states[session_id]
        activity_log = self.activity_logs.get(session_id, [])
        latest_activity = activity_log[-1] if activity_log else None

        report_lines = [
            "# Research Progress Report",
            "",
            f"**Session ID**: {session.session_id}",
            f"**Current Phase**: {session.current_phase.title()}",
            f"**Session Start**: {session.start_date.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Last Updated**: {progress.last_updated.strftime('%Y-%m-%d %H:%M:%S') if progress.last_updated else 'N/A'}",
            "",
            "## Progress Metrics",
        ]

        for key in sorted(session.progress_metrics.keys()):
            value = session.progress_metrics[key]
            report_lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        if session.weekly_targets:
            report_lines.extend(["", "## Weekly Targets"])
            for key, value in session.weekly_targets.items():
                achieved = session.progress_metrics.get(key, 0)
                report_lines.append(f"- {key.replace('_', ' ').title()}: {achieved}/{value}")

        if latest_activity:
            report_lines.extend(
                [
                    "",
                    "## Latest Activity",
                    f"- **Type**: {latest_activity.activity_type}",
                    f"- **Description**: {latest_activity.description}",
                    f"- **Outcome**: {latest_activity.outcome}",
                    f"- **Timestamp**: {latest_activity.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                ]
            )

        if self.error_log.get(session_id):
            report_lines.extend(["", "## Recent Errors"])
            for error_entry in self.error_log[session_id][-5:]:
                report_lines.append(
                    f"- {error_entry['timestamp']}: {error_entry['operation']} - {error_entry['message']}"
                )

        return "\n".join(report_lines)

    def generate_weekly_report(
        self,
        session_id: str,
        week_number: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> WeeklyReport:
        start_date, end_date = self._normalize_report_window(start_date, end_date)
        session = self.sessions[session_id]

        history = self._collect_progress_history(session_id, start_date, end_date)
        start_progress, end_progress = self._resolve_progress_bounds(
            session_id, history, end_date
        )
        deltas = self._compute_progress_deltas(start_progress, end_progress)

        key_findings = self._build_key_findings(deltas)
        challenges = self._build_challenges(session_id, start_date, end_date)
        next_week_priorities = self._build_next_week_priorities(session, deltas)

        return WeeklyReport(
            week_number=week_number or end_date.isocalendar().week,
            start_date=start_date,
            end_date=end_date,
            sources_identified=deltas["sources_identified"],
            datasets_evaluated=deltas["datasets_evaluated"],
            access_established=deltas["access_established"],
            datasets_acquired=deltas["datasets_acquired"],
            integration_plans_created=deltas["integration_plans_created"],
            key_findings=key_findings,
            challenges=challenges,
            next_week_priorities=next_week_priorities,
        )

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
        self, session: Any, deltas: Dict[str, int]
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


class WorkflowMixin(RetryMixin):
    """Implements workflow execution helpers."""

    discovery_service: Any
    evaluation_engine: Any
    acquisition_manager: Any
    integration_engine: Any

    # Methods provided by classes using this mixin (e.g., ResearchOrchestrator)
    def log_activity(
        self,
        session_id: str,
        activity_type: str,
        description: str,
        outcome: str,
        source_id: Optional[str] = None,
        duration_minutes: int = 0,
    ) -> None:
        """Record a research activity entry. Implemented by the orchestrator."""
        raise NotImplementedError("log_activity must be implemented by the orchestrator")

    def update_progress(self, session_id: str, metrics: Dict[str, int]) -> None:
        """Update progress metrics for a session. Implemented by the orchestrator."""
        raise NotImplementedError("update_progress must be implemented by the orchestrator")

    def advance_phase(self, session_id: str) -> str:
        """Advance the research session to the next phase. Implemented by the orchestrator."""
        raise NotImplementedError("advance_phase must be implemented by the orchestrator")

    def _run_discovery_phase(
        self,
        session_id: str,
        session: Any,
        state: SessionState,
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
                session_id, {"datasets_evaluated": len(evaluations)}
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
                session_id, {"integration_plans_created": len(plans)}
            )
        self.log_activity(
            session_id,
            "integration",
            "Integration planning complete",
            f"{len(plans)} plans created",
        )

    def _evaluate_sources(
        self, session_id: str, sources: List[DatasetSource], evaluator: str
    ) -> List[DatasetEvaluation]:
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
        except Exception as exc:  # pragma: no cover - logging path
            self._log_error(
                session_id, f"evaluate:{source.source_id}", exc, self.config.max_retries
            )
            return None

    def _acquire_datasets(self, session_id: str, state: SessionState) -> None:
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
                self._log_error(
                    session_id,
                    f"access_request:{source.source_id}",
                    exc,
                    self.config.max_retries,
                )
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
                self._log_error(
                    session_id,
                    f"download:{source.source_id}",
                    exc,
                    self.config.max_retries,
                )
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
        except Exception as exc:
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
            except Exception as exc:
                feasible = False
                self._log_error(
                    session_id,
                    f"integration_validation:{dataset.source_id}",
                    exc,
                    self.config.max_retries,
                )
        state.integration_feasibility[dataset.source_id] = feasible
        return plan


