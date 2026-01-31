from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.auth.authorization import AuthorizationHandler
from ai.sourcing.journal.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchLog,
    ResearchProgress,
    ResearchSession,
)
from ai.sourcing.journal.orchestrator.types import ProgressSnapshot, SessionState


class AllowAllAuthorization(AuthorizationHandler):
    """Authorization handler that grants all requests (test only)."""

    async def authorize(
        self,
        user: Dict[str, Any],
        resource: str,
        action: str,
    ) -> bool:
        return True

    async def require_authorization(
        self,
        user: Dict[str, Any],
        resource: str,
        action: str,
    ) -> None:
        return None


class FakeOrchestrator:
    """Lightweight orchestrator stub for MCP integration and E2E tests."""

    def __init__(
        self,
        session: ResearchSession,
        sources: List[DatasetSource],
        evaluations: List[DatasetEvaluation],
        acquisitions: List[AcquiredDataset],
        integration_plans: List[IntegrationPlan],
        report: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session = session
        self.state = SessionState(
            sources=deepcopy(sources),
            evaluations=deepcopy(evaluations),
            access_requests=[],
            acquired_datasets=deepcopy(acquisitions),
            integration_plans=deepcopy(integration_plans),
        )
        self.progress = ResearchProgress(
            sources_identified=len(self.state.sources),
            datasets_evaluated=len(self.state.evaluations),
            access_established=len(
                [d for d in self.state.acquired_datasets if d.storage_path]
            ),
            datasets_acquired=len(self.state.acquired_datasets),
            integration_plans_created=len(self.state.integration_plans),
            last_updated=datetime.utcnow(),
        )
        self.history: List[ProgressSnapshot] = [
            ProgressSnapshot(
                timestamp=self.progress.last_updated or datetime.utcnow(),
                progress=self.progress,
                metrics={
                    "sources_identified": self.progress.sources_identified,
                    "datasets_evaluated": self.progress.datasets_evaluated,
                    "datasets_acquired": self.progress.datasets_acquired,
                },
            ),
        ]
        self.activity_log: List[ResearchLog] = [
            ResearchLog(
                timestamp=datetime.utcnow(),
                activity_type="session_start",
                description="Session initialized for testing",
                outcome="success",
            )
        ]
        self.error_log: List[Dict[str, str]] = []
        self.report = report or {
            "report_id": f"report_{session.session_id}",
            "session_id": session.session_id,
            "report_type": "summary_report",
            "format": "json",
            "generated_date": datetime.utcnow().isoformat(),
            "content": {"summary": "Test report"},
        }

    def _validate_session_id(self, session_id: str) -> None:
        if session_id != self.session.session_id:
            raise KeyError(session_id)

    def get_progress(self, session_id: str) -> ResearchProgress:
        self._validate_session_id(session_id)
        return self.progress

    def get_progress_history(self, session_id: str) -> List[ProgressSnapshot]:
        self._validate_session_id(session_id)
        return list(self.history)

    def get_session_state(self, session_id: str) -> SessionState:
        self._validate_session_id(session_id)
        return self.state

    def get_activity_log(self, session_id: str) -> List[ResearchLog]:
        self._validate_session_id(session_id)
        return list(self.activity_log)

    def get_error_log(self, session_id: str) -> List[Dict[str, str]]:
        self._validate_session_id(session_id)
        return list(self.error_log)

    def generate_progress_report(self, session_id: str) -> Dict[str, Any]:
        self._validate_session_id(session_id)
        return deepcopy(self.report)

    def update_sources(self, sources: List[DatasetSource]) -> None:
        self.state.sources = deepcopy(sources)
        self._update_progress_counts()

    def update_evaluations(self, evaluations: List[DatasetEvaluation]) -> None:
        self.state.evaluations = deepcopy(evaluations)
        self._update_progress_counts()

    def update_acquisitions(self, acquisitions: List[AcquiredDataset]) -> None:
        self.state.acquired_datasets = deepcopy(acquisitions)
        self._update_progress_counts()

    def update_integration_plans(self, plans: List[IntegrationPlan]) -> None:
        self.state.integration_plans = deepcopy(plans)
        self._update_progress_counts()

    def _update_progress_counts(self) -> None:
        self.progress.sources_identified = len(self.state.sources)
        self.progress.datasets_evaluated = len(self.state.evaluations)
        self.progress.datasets_acquired = len(self.state.acquired_datasets)
        self.progress.access_established = len(
            [d for d in self.state.acquired_datasets if d.storage_path]
        )
        self.progress.integration_plans_created = len(self.state.integration_plans)
        self.progress.last_updated = datetime.utcnow()
        self.history.append(
            ProgressSnapshot(
                timestamp=self.progress.last_updated,
                progress=self.progress,
                metrics={
                    "sources_identified": self.progress.sources_identified,
                    "datasets_evaluated": self.progress.datasets_evaluated,
                    "datasets_acquired": self.progress.datasets_acquired,
                },
            )
        )


class FakeCommandHandlerService:
    """CommandHandlerService stub wired to the fake orchestrator."""

    def __init__(
        self,
        session: ResearchSession,
        sources: List[DatasetSource],
        evaluations: List[DatasetEvaluation],
        acquisitions: List[AcquiredDataset],
        integration_plans: List[IntegrationPlan],
        report: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session = deepcopy(session)
        self.sources = [deepcopy(source) for source in sources]
        self.evaluations = [deepcopy(evaluation) for evaluation in evaluations]
        self.acquisitions = [deepcopy(acquisition) for acquisition in acquisitions]
        self.integration_plans = [
            deepcopy(plan) for plan in integration_plans
        ]
        self.report = deepcopy(report) if report else None
        self.orchestrator = FakeOrchestrator(
            self.session,
            self.sources,
            self.evaluations,
            self.acquisitions,
            self.integration_plans,
            report=self.report,
        )
        self._reports: Dict[str, Dict[str, Any]] = {}
        if self.orchestrator.report:
            self._reports[self.orchestrator.report["report_id"]] = deepcopy(
                self.orchestrator.report
            )

    # Session management -------------------------------------------------
    def create_session(
        self,
        target_sources: List[str],
        search_keywords: Dict[str, List[str]],
        weekly_targets: Optional[Dict[str, int]] = None,
        session_id: Optional[str] = None,
    ) -> ResearchSession:
        if session_id and session_id != self.session.session_id:
            self.session.session_id = session_id
        self.session.target_sources = target_sources
        self.session.search_keywords = search_keywords
        if weekly_targets is not None:
            self.session.weekly_targets = weekly_targets
        self.session.start_date = datetime.utcnow()
        self.orchestrator.session = self.session
        self.orchestrator.state = SessionState()
        return self.session

    def list_sessions(self) -> List[ResearchSession]:
        return [self.session]

    def get_session(self, session_id: str) -> ResearchSession:
        if session_id != self.session.session_id:
            raise ValueError(f"Session {session_id} not found")
        return self.session

    def update_session(
        self,
        session_id: str,
        target_sources: Optional[List[str]] = None,
        search_keywords: Optional[Dict[str, List[str]]] = None,
        weekly_targets: Optional[Dict[str, int]] = None,
        current_phase: Optional[str] = None,
    ) -> ResearchSession:
        if session_id != self.session.session_id:
            raise ValueError(f"Session {session_id} not found")
        if target_sources is not None:
            self.session.target_sources = target_sources
        if search_keywords is not None:
            self.session.search_keywords = search_keywords
        if weekly_targets is not None:
            self.session.weekly_targets = weekly_targets
        if current_phase is not None:
            self.session.current_phase = current_phase
        return self.session

    def delete_session(self, session_id: str) -> None:
        if session_id != self.session.session_id:
            raise ValueError(f"Session {session_id} not found")

    # Discovery ----------------------------------------------------------
    def initiate_discovery(
        self,
        session_id: str,
        keywords: List[str],
        sources: Any,
    ) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        normalized_sources = sources
        if isinstance(normalized_sources, str):  # type: ignore[isinstance-second-argument-not-valid-type]
            normalized_sources = [normalized_sources]
        self.session.target_sources = normalized_sources
        self.session.search_keywords = {"default": keywords}
        self.orchestrator.update_sources(self.sources)
        return {
            "session_id": session_id,
            "total_sources": len(self.sources),
        }

    def get_sources(self, session_id: str) -> List[DatasetSource]:
        self.orchestrator._validate_session_id(session_id)
        return deepcopy(self.sources)

    def get_source(self, session_id: str, source_id: str) -> DatasetSource:
        self.orchestrator._validate_session_id(session_id)
        for source in self.sources:
            if source.source_id == source_id:
                return deepcopy(source)
        raise ValueError(f"Source {source_id} not found")

    # Evaluation ---------------------------------------------------------
    def initiate_evaluation(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        evaluations = [
            evaluation
            for evaluation in self.evaluations
            if not source_ids or evaluation.source_id in source_ids
        ]
        self.orchestrator.update_evaluations(evaluations)
        return {
            "session_id": session_id,
            "evaluations": [
                {
                    "evaluation_id": f"eval_{evaluation.source_id}",
                    "source_id": evaluation.source_id,
                    "overall_score": evaluation.overall_score,
                }
                for evaluation in evaluations
            ],
        }

    def get_evaluations(self, session_id: str) -> List[DatasetEvaluation]:
        self.orchestrator._validate_session_id(session_id)
        return deepcopy(self.evaluations)

    def get_evaluation(self, session_id: str, evaluation_id: str) -> DatasetEvaluation:
        self.orchestrator._validate_session_id(session_id)
        source_id = evaluation_id.replace("eval_", "")
        for evaluation in self.evaluations:
            if evaluation.source_id == source_id:
                return deepcopy(evaluation)
        raise ValueError(f"Evaluation {evaluation_id} not found")

    def update_evaluation(
        self,
        session_id: str,
        evaluation_id: str,
        therapeutic_relevance: Optional[int] = None,
        data_structure_quality: Optional[int] = None,
        training_integration: Optional[int] = None,
        ethical_accessibility: Optional[int] = None,
        priority_tier: Optional[str] = None,
    ) -> DatasetEvaluation:
        evaluation = self.get_evaluation(session_id, evaluation_id)
        if therapeutic_relevance is not None:
            evaluation.therapeutic_relevance = therapeutic_relevance
        if data_structure_quality is not None:
            evaluation.data_structure_quality = data_structure_quality
        if training_integration is not None:
            evaluation.training_integration = training_integration
        if ethical_accessibility is not None:
            evaluation.ethical_accessibility = ethical_accessibility
        if priority_tier is not None:
            evaluation.priority_tier = priority_tier
        return evaluation

    # Acquisition --------------------------------------------------------
    def initiate_acquisition(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        acquisitions = [
            acquisition
            for acquisition in self.acquisitions
            if not source_ids or acquisition.source_id in source_ids
        ]
        self.orchestrator.update_acquisitions(acquisitions)
        return {
            "session_id": session_id,
            "acquired": [acquisition.source_id for acquisition in acquisitions],
        }

    def get_acquisitions(self, session_id: str) -> List[AcquiredDataset]:
        self.orchestrator._validate_session_id(session_id)
        return deepcopy(self.acquisitions)

    def get_acquisition(self, session_id: str, acquisition_id: str) -> AcquiredDataset:
        self.orchestrator._validate_session_id(session_id)
        source_id = acquisition_id.replace("acq_", "")
        for acquisition in self.acquisitions:
            if acquisition.source_id == source_id:
                return deepcopy(acquisition)
        raise ValueError(f"Acquisition {acquisition_id} not found")

    def update_acquisition(
        self,
        session_id: str,
        acquisition_id: str,
        status: Optional[str] = None,
    ) -> AcquiredDataset:
        acquisition = self.get_acquisition(session_id, acquisition_id)
        if status == "completed":
            acquisition.storage_path = acquisition.storage_path or "/tmp/dataset.zip"
        return acquisition

    # Integration --------------------------------------------------------
    def initiate_integration(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
        target_format: str = "chatml",
    ) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        plans = [
            plan
            for plan in self.integration_plans
            if not source_ids or plan.source_id in source_ids
        ]
        for plan in plans:
            plan.dataset_format = target_format
        self.orchestrator.update_integration_plans(plans)
        return {
            "session_id": session_id,
            "plans": [
                {"source_id": plan.source_id, "target_format": target_format}
                for plan in plans
            ],
        }

    def get_integration_plans(self, session_id: str) -> List[IntegrationPlan]:
        self.orchestrator._validate_session_id(session_id)
        return deepcopy(self.integration_plans)

    def get_integration_plan(self, session_id: str, plan_id: str) -> IntegrationPlan:
        self.orchestrator._validate_session_id(session_id)
        source_id = plan_id.replace("plan_", "")
        for plan in self.integration_plans:
            if plan.source_id == source_id:
                return deepcopy(plan)
        raise ValueError(f"Integration plan {plan_id} not found")

    # Reports ------------------------------------------------------------
    def generate_report(
        self,
        session_id: str,
        report_type: str,
        format: str,
        date_range: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        report = {
            "report_id": f"report_{session_id}_{report_type}",
            "session_id": session_id,
            "report_type": report_type,
            "format": format,
            "generated_date": datetime.utcnow(),
            "content": {
                "summary": f"{report_type} generated in {format}",
                "date_range": date_range,
            },
            "file_path": None,
        }
        self._reports[report["report_id"]] = deepcopy(report)
        return report

    def get_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        self.orchestrator._validate_session_id(session_id)
        if report_id not in self._reports:
            raise ValueError(f"Report {report_id} not found")
        return deepcopy(self._reports[report_id])

    def list_reports(self, session_id: str) -> List[Dict[str, Any]]:
        self.orchestrator._validate_session_id(session_id)
        return [deepcopy(report) for report in self._reports.values()]


