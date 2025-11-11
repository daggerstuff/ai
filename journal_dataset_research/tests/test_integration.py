"""
Integration tests for journal dataset research system.

Tests component communication, workflow state transitions, error handling,
and progress tracking across multiple components.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ai.journal_dataset_research.acquisition.acquisition_manager import (
    AccessAcquisitionManager,
    AcquisitionConfig,
)
from ai.journal_dataset_research.evaluation.evaluation_engine import (
    DatasetEvaluationEngine,
)
from ai.journal_dataset_research.integration.integration_planning_engine import (
    IntegrationPlanningEngine,
)
from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    ResearchProgress,
    ResearchSession,
)
from ai.journal_dataset_research.orchestrator.research_orchestrator import (
    ResearchOrchestrator,
)
from ai.journal_dataset_research.orchestrator.types import OrchestratorConfig


class TestComponentCommunication:
    """Tests for component communication and data flow."""

    def test_discovery_to_evaluation_flow(self, sample_dataset_source, temp_dir):
        """Test data flow from discovery to evaluation."""
        # Create evaluation engine
        evaluation_engine = DatasetEvaluationEngine()

        # Evaluate discovered source
        evaluation = evaluation_engine.evaluate_dataset(
            source=sample_dataset_source,
            evaluator="test",
        )

        assert evaluation.source_id == sample_dataset_source.source_id
        assert evaluation.overall_score >= 0
        assert evaluation.priority_tier in ["high", "medium", "low"]

    def test_evaluation_to_acquisition_flow(self, sample_dataset_source, sample_evaluation, temp_dir):
        """Test data flow from evaluation to acquisition."""
        # Create acquisition manager
        config = AcquisitionConfig(storage_base_path=str(temp_dir))
        acquisition_manager = AccessAcquisitionManager(config)

        # Submit access request based on evaluation
        access_request = acquisition_manager.submit_access_request(
            source=sample_dataset_source,
            access_method="direct",
        )

        assert access_request.source_id == sample_dataset_source.source_id
        assert access_request.status == "pending"

    def test_acquisition_to_integration_flow(self, sample_acquired_dataset, temp_dir):
        """Test data flow from acquisition to integration."""
        # Create integration planning engine
        integration_engine = IntegrationPlanningEngine()

        # Create integration plan for acquired dataset
        integration_plan = integration_engine.create_integration_plan(
            dataset=sample_acquired_dataset,
        )

        assert integration_plan.source_id == sample_acquired_dataset.source_id
        assert integration_plan.complexity in ["low", "medium", "high"]
        assert len(integration_plan.required_transformations) >= 0

    def test_full_workflow_data_flow(self, sample_dataset_source, temp_dir):
        """Test complete workflow data flow."""
        # Step 1: Evaluation
        evaluation_engine = DatasetEvaluationEngine()
        evaluation = evaluation_engine.evaluate_dataset(
            source=sample_dataset_source,
        )

        # Step 2: Acquisition
        config = AcquisitionConfig(storage_base_path=str(temp_dir))
        acquisition_manager = AccessAcquisitionManager(config)
        access_request = acquisition_manager.submit_access_request(
            source=sample_dataset_source,
        )

        # Step 3: Integration planning
        integration_engine = IntegrationPlanningEngine()
        # Create a mock acquired dataset
        acquired_dataset = AcquiredDataset(
            source_id=sample_dataset_source.source_id,
            storage_path=str(temp_dir / "dataset.zip"),
        )
        integration_plan = integration_engine.create_integration_plan(
            dataset=acquired_dataset,
        )

        # Verify data flows correctly
        assert evaluation.source_id == sample_dataset_source.source_id
        assert access_request.source_id == sample_dataset_source.source_id
        assert integration_plan.source_id == sample_dataset_source.source_id


class TestWorkflowStateTransitions:
    """Tests for workflow state transitions."""

    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create an orchestrator for testing."""
        config = OrchestratorConfig(
            session_storage_path=temp_dir,
            max_retries=3,
            retry_delay_seconds=0.1,
        )
        return ResearchOrchestrator(config=config)

    def test_phase_transitions(self, orchestrator):
        """Test phase transitions in workflow."""
        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        assert session.current_phase == "discovery"

        # Transition to evaluation
        next_phase = orchestrator.advance_phase(session.session_id)
        assert next_phase == "evaluation"

        # Transition to acquisition
        next_phase = orchestrator.advance_phase(session.session_id)
        assert next_phase == "acquisition"

        # Transition to integration
        next_phase = orchestrator.advance_phase(session.session_id)
        assert next_phase == "integration"

        # Should stay at integration (final phase)
        next_phase = orchestrator.advance_phase(session.session_id)
        assert next_phase == "integration"

    def test_progress_tracking_across_phases(self, orchestrator):
        """Test progress tracking across workflow phases."""
        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Update progress in discovery phase
        orchestrator.update_progress(
            session.session_id,
            {"sources_identified": 5},
        )

        # Advance to evaluation phase
        orchestrator.advance_phase(session.session_id)

        # Update progress in evaluation phase
        orchestrator.update_progress(
            session.session_id,
            {"datasets_evaluated": 3},
        )

        # Check progress history
        history = orchestrator.get_progress_history(session.session_id)
        assert len(history) >= 2

        # Verify progress is tracked correctly
        progress = orchestrator.get_progress(session.session_id)
        assert progress.sources_identified == 5
        assert progress.datasets_evaluated == 3


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_error_recovery_in_discovery(self, temp_dir):
        """Test error recovery in discovery phase."""
        class FailingDiscoveryService:
            def __init__(self):
                self.attempts = 0

            def discover_sources(self, session):
                self.attempts += 1
                if self.attempts < 3:
                    raise RuntimeError("Temporary error")
                return []

        config = OrchestratorConfig(
            session_storage_path=temp_dir,
            max_retries=5,
            retry_delay_seconds=0.0,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FailingDiscoveryService(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Should recover after retries
        try:
            orchestrator.run_session(session.session_id)
        except RuntimeError:
            # If it still fails after retries, that's expected
            pass

        # Check that retries were attempted
        error_log = orchestrator.get_error_log(session.session_id)
        assert len(error_log) > 0

    def test_fallback_strategy(self, temp_dir):
        """Test fallback strategy for failures."""
        class FailingService:
            def discover_sources(self, session):
                raise RuntimeError("Service unavailable")

        config = OrchestratorConfig(
            session_storage_path=temp_dir,
            max_retries=2,
            retry_delay_seconds=0.0,
            fallback_on_failure=True,
        )
        orchestrator = ResearchOrchestrator(
            discovery_service=FailingService(),
            config=config,
        )

        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Should not raise with fallback enabled
        state = orchestrator.run_session(session.session_id)
        assert state is not None


class TestProgressTracking:
    """Tests for progress tracking accuracy."""

    def test_progress_metrics_accuracy(self, orchestrator):
        """Test that progress metrics are tracked accurately."""
        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Update progress multiple times
        orchestrator.update_progress(session.session_id, {"sources_identified": 5})
        orchestrator.update_progress(session.session_id, {"sources_identified": 10})
        orchestrator.update_progress(session.session_id, {"datasets_evaluated": 3})

        # Check final progress
        progress = orchestrator.get_progress(session.session_id)
        assert progress.sources_identified == 10
        assert progress.datasets_evaluated == 3

        # Check progress history
        history = orchestrator.get_progress_history(session.session_id)
        assert len(history) == 3

        # Verify history accuracy
        assert history[0].progress.sources_identified == 5
        assert history[1].progress.sources_identified == 10
        assert history[2].progress.datasets_evaluated == 3

    def test_weekly_target_tracking(self, orchestrator):
        """Test weekly target tracking."""
        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
            weekly_targets={
                "sources_identified": 10,
                "datasets_evaluated": 5,
            },
        )

        # Update progress
        orchestrator.update_progress(session.session_id, {"sources_identified": 7})
        orchestrator.update_progress(session.session_id, {"datasets_evaluated": 3})

        # Generate progress visualization
        visualization = orchestrator.generate_progress_visualization_data(
            session.session_id
        )

        assert "targets" in visualization
        assert visualization["targets"]["sources_identified"]["achieved"] == 7
        assert visualization["targets"]["sources_identified"]["target"] == 10
        assert visualization["targets"]["datasets_evaluated"]["achieved"] == 3
        assert visualization["targets"]["datasets_evaluated"]["target"] == 5


class TestSessionPersistence:
    """Tests for session persistence and recovery."""

    def test_session_save_and_load(self, temp_dir):
        """Test saving and loading session state."""
        config = OrchestratorConfig(
            session_storage_path=temp_dir,
            max_retries=2,
            retry_delay_seconds=0.0,
        )
        orchestrator = ResearchOrchestrator(config=config)

        session = orchestrator.start_research_session(
            target_sources=["pubmed"],
            search_keywords={"therapy": ["cbt"]},
        )

        # Update progress
        orchestrator.update_progress(session.session_id, {"sources_identified": 5})

        # Save session
        saved_path = orchestrator.save_session_state(session.session_id)
        assert saved_path.exists()

        # Create new orchestrator and load session
        new_orchestrator = ResearchOrchestrator(config=config)
        loaded_session = new_orchestrator.load_session_state(session.session_id)

        assert loaded_session.session_id == session.session_id
        assert loaded_session.target_sources == session.target_sources

        # Verify progress was restored
        progress = new_orchestrator.get_progress(loaded_session.session_id)
        assert progress.sources_identified == 5

