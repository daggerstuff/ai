"""
Unit tests for the Research Orchestrator.
"""

from datetime import datetime, timedelta
from typing import Callable, List, Optional

import pytest

from ai.sourcing.journal.acquisition.acquisition_manager import DownloadProgress
from ai.sourcing.journal.models.dataset_models import (
    AccessRequest,
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchProgress,
)
from ai.sourcing.journal.orchestrator import (
    OrchestratorConfig,
    ResearchOrchestrator,
)


def _create_dataset_source(source_id: str, **overrides) -> DatasetSource:
    """Helper to create dataset sources for tests."""
    defaults = {
        "title": f"Dataset {source_id}",
        "authors": ["Researcher"],
        "publication_date": datetime(2024, 1, 1),
        "source_type": "repository",
        "url": f"https://example.com/{source_id}.zip",
        "doi": f"10.1000/{source_id}",
        "abstract": "Therapeutic dataset for testing",
        "keywords": ["therapy", "conversation"],
        "open_access": True,
        "data_availability": "available",
        "discovery_date": datetime.now(),
        "discovery_method": "repository_api",
    }
    defaults.update(overrides)
    return DatasetSource(source_id=source_id, **defaults)


class FakeDiscoveryService:
    """Simple discovery service returning predefined sources."""

    def __init__(self, sources: List[DatasetSource]):
        self.sources = sources
        self.invocations = 0

    def discover_sources(self, session):
        self.invocations += 1
        return list(self.sources)


class FlakyDiscoveryService(FakeDiscoveryService):
    """Discovery service that fails on first attempt to test retries."""

    def __init__(self, sources: List[DatasetSource], failures: int = 1):
        super().__init__(sources)
        self.failures = failures

    def discover_sources(self, session):
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("temporary discovery error")
        return super().discover_sources(session)


class FakeEvaluationEngine:
    """Evaluation engine returning deterministic results."""

    def __init__(self, score: float = 8.5):
        self.score = score

    def evaluate_dataset(self, source: DatasetSource, evaluator: str = "system"):
        return DatasetEvaluation(
            source_id=source.source_id,
            therapeutic_relevance=8,
            therapeutic_relevance_notes="High therapeutic value",
            data_structure_quality=8,
            data_structure_notes="Well structured",
            training_integration=7,
            integration_notes="Compatible with pipeline",
            ethical_accessibility=9,
            ethical_notes="Open access and anonymized",
            overall_score=self.score,
            priority_tier="high",
            evaluation_date=datetime.now(),
            evaluator=evaluator,
            competitive_advantages=["Contains therapy transcripts"],
        )


class FakeAcquisitionManager:
    """Acquisition manager that tracks requests and returns stub datasets."""

    def __init__(self):
        self.submissions = 0
        self.downloads = 0

    def submit_access_request(self, source: DatasetSource, access_method=None, notes: str = ""):
        self.submissions += 1
        return AccessRequest(
            source_id=source.source_id,
            access_method=access_method or "direct",
            request_date=datetime.now(),
            status="pending",
            access_url=source.url,
            credentials_required=False,
            institutional_affiliation_required=False,
            estimated_access_date=datetime.now() + timedelta(hours=1),
            notes=notes,
        )

    def download_dataset(
        self,
        source: DatasetSource,
        access_request: Optional[AccessRequest] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ):
        self.downloads += 1
        return AcquiredDataset(
            source_id=source.source_id,
            acquisition_date=datetime.now(),
            storage_path=f"/tmp/{source.source_id}.zip",
            file_format="zip",
            file_size_mb=25.0,
            license="CC-BY",
            usage_restrictions=[],
            attribution_required=False,
            checksum="abc123",
        )


class FakeIntegrationEngine:
    """Integration engine creating simple plans."""

    def __init__(self, feasible: bool = True):
        self.feasible = feasible
        self.requests = 0

    def create_integration_plan(
        self, dataset: AcquiredDataset, target_format: str = "chatml"
    ) -> IntegrationPlan:
        self.requests += 1
        return IntegrationPlan(
            source_id=dataset.source_id,
            dataset_format="zip",
            schema_mapping={"messages": "messages"},
            required_transformations=["format_conversion"],
            preprocessing_steps=["step1", "step2"],
            complexity="medium",
            estimated_effort_hours=4,
            dependencies=[],
            integration_priority=1,
        )

    def validate_integration_feasibility(self, plan: IntegrationPlan) -> bool:
        return self.feasible


class TestResearchOrchestrator:
    """Tests covering primary orchestrator functionality."""

    @pytest.fixture
    def sample_sources(self):
        return [
            _create_dataset_source("dataset-1"),
            _create_dataset_source("dataset-2"),
        ]

    @pytest.fixture
    def orchestrator(self, sample_sources):
        return ResearchOrchestrator(
            discovery_service=FakeDiscoveryService(sample_sources),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=OrchestratorConfig(max_retries=3, retry_delay_seconds=0.0),
        )

    def test_start_session_initializes_state(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=["pubmed", "zenodo"],
            search_keywords={"therapy": ["cbt", "dbt"]},
        )

        assert session.session_id in orchestrator.sessions
        assert orchestrator.progress_states[session.session_id] == ResearchProgress()
        assert orchestrator.get_activity_log(session.session_id)[0].activity_type == "session_start"

    def test_advance_phase_sequence(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )

        assert session.current_phase == "discovery"
        assert orchestrator.advance_phase(session.session_id) == "evaluation"
        assert orchestrator.advance_phase(session.session_id) == "acquisition"
        assert orchestrator.advance_phase(session.session_id) == "integration"
        # stays at final phase
        assert orchestrator.advance_phase(session.session_id) == "integration"

    def test_update_progress_records_snapshot(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        orchestrator.update_progress(session.session_id, {"sources_identified": 3})
        history = orchestrator.get_progress_history(session.session_id)

        assert len(history) == 1
        snapshot = history[0]
        assert snapshot.progress.sources_identified == 3
        assert snapshot.metrics["sources_identified"] == 3

    def test_generate_progress_report_contains_metrics(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        orchestrator.update_progress(session.session_id, {"sources_identified": 2})
        report = orchestrator.generate_progress_report(session.session_id)

        assert "Research Progress Report" in report
        assert "Sources Identified" in report
        assert "2" in report

    def test_generate_weekly_report_aggregates_metrics(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )
        orchestrator.update_progress(session.session_id, {"sources_identified": 2})
        orchestrator.update_progress(session.session_id, {"datasets_evaluated": 1})

        report = orchestrator.generate_weekly_report(session.session_id)

        assert report.sources_identified >= 0
        assert isinstance(report.key_findings, list)
        assert isinstance(report.next_week_priorities, list)

    def test_run_session_executes_full_workflow(self, orchestrator, sample_sources):
        session = orchestrator.start_research_session(
            target_sources=["zenodo"],
            search_keywords={"therapy": ["cbt"]},
        )

        state = orchestrator.run_session(session.session_id)

        assert len(state.sources) == len(sample_sources)
        assert len(state.evaluations) == len(sample_sources)
        assert len(state.access_requests) == len(sample_sources)
        assert len(state.acquired_datasets) == len(sample_sources)
        assert len(state.integration_plans) == len(sample_sources)
        assert orchestrator.progress_states[session.session_id].datasets_acquired == len(sample_sources)

    def test_retry_logic_handles_transient_errors(self, sample_sources):
        orchestrator = ResearchOrchestrator(
            discovery_service=FlakyDiscoveryService(sample_sources, failures=1),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=OrchestratorConfig(max_retries=3, retry_delay_seconds=0.0),
        )

        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )

        state = orchestrator.run_session(session.session_id)
        # should still discover sources after retry
        assert len(state.sources) == len(sample_sources)
        assert any("discovery" in entry["operation"] for entry in orchestrator.get_error_log(session.session_id))

    def test_session_persistence_round_trip(self, tmp_path, sample_sources):
        config = OrchestratorConfig(
            max_retries=2,
            retry_delay_seconds=0.0,
            session_storage_path=tmp_path,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FakeDiscoveryService(sample_sources),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=["zenodo"],
            search_keywords={"therapy": ["cbt"]},
            weekly_targets={"datasets_evaluated": len(sample_sources)},
        )
        orchestrator.run_session(session.session_id)

        saved_path = orchestrator.save_session_state(session.session_id)
        assert saved_path.exists()

        restored_orchestrator = ResearchOrchestrator(
            config=OrchestratorConfig(
                max_retries=2,
                retry_delay_seconds=0.0,
                session_storage_path=tmp_path,
            )
        )
        loaded_session = restored_orchestrator.load_session_state(session.session_id)

        assert loaded_session.session_id == session.session_id
        assert isinstance(loaded_session.start_date, datetime)
        restored_state = restored_orchestrator.session_states[session.session_id]
        assert len(restored_state.sources) == len(sample_sources)
        assert len(restored_orchestrator.progress_history[session.session_id]) >= 1

        visualization = restored_orchestrator.generate_progress_visualization_data(
            session.session_id
        )
        assert visualization["current_metrics"]["datasets_evaluated"] >= len(sample_sources)
        assert visualization["series"]

    def test_progress_visualization_reports_targets(self, orchestrator):
        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
            weekly_targets={"sources_identified": 5},
        )
        orchestrator.update_progress(session.session_id, {"sources_identified": 2})
        orchestrator.update_progress(session.session_id, {"sources_identified": 1})

        data = orchestrator.generate_progress_visualization_data(session.session_id)

        assert data["series"]
        assert data["last_updated"] is not None
        target_info = data["targets"]["sources_identified"]
        assert target_info["achieved"] == 3
        assert target_info["target"] == 5
        assert target_info["percent_complete"] > 0
        assert target_info["remaining"] == 2

    def test_exponential_backoff_in_retry_logic(self, sample_sources):
        import time

        config = OrchestratorConfig(
            max_retries=3,
            retry_delay_seconds=0.1,  # 100ms base delay
            fallback_on_failure=False,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FlakyDiscoveryService(sample_sources, failures=2),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=[],
            search_keywords={},
        )

        start_time = time.time()
        state = orchestrator.run_session(session.session_id)
        elapsed = time.time() - start_time

        # Should have retried twice with exponential backoff:
        # First retry: 0.1 * 2^0 = 0.1s
        # Second retry: 0.1 * 2^1 = 0.2s
        # Total delay should be at least 0.3s
        assert elapsed >= 0.3
        assert len(state.sources) == len(sample_sources)

    def test_fallback_strategy_for_discovery_failure(self, sample_sources):
        class FailingDiscoveryService:
            def discover_sources(self, session):
                raise RuntimeError("Discovery service unavailable")

        config = OrchestratorConfig(
            max_retries=2,
            retry_delay_seconds=0.0,
            fallback_on_failure=True,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FailingDiscoveryService(),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Should not raise, but continue with empty sources
        state = orchestrator.run_session(session.session_id)

        assert len(state.sources) == 0
        assert session.current_phase in ["evaluation", "acquisition", "integration"]
        # Should have logged the fallback
        activity_log = orchestrator.get_activity_log(session.session_id)
        fallback_activities = [
            log
            for log in activity_log
            if "fallback" in log.description.lower() or "fallback" in log.outcome.lower()
        ]
        assert len(fallback_activities) > 0

    def test_fallback_disabled_raises_on_discovery_failure(self, sample_sources):
        class FailingDiscoveryService:
            def discover_sources(self, session):
                raise RuntimeError("Discovery service unavailable")

        config = OrchestratorConfig(
            max_retries=2,
            retry_delay_seconds=0.0,
            fallback_on_failure=False,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FailingDiscoveryService(),
            evaluation_engine=FakeEvaluationEngine(),
            acquisition_manager=FakeAcquisitionManager(),
            integration_engine=FakeIntegrationEngine(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Should raise when fallback is disabled
        with pytest.raises(RuntimeError, match="Discovery service unavailable"):
            orchestrator.run_session(session.session_id)

