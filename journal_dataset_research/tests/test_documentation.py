"""
Unit tests for documentation module.

Tests research logger, report generator, dataset catalog, tracking updater,
and progress visualization components.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ai.journal_dataset_research.documentation.dataset_catalog import DatasetCatalog
from ai.journal_dataset_research.documentation.progress_visualization import (
    ProgressVisualization,
)
from ai.journal_dataset_research.documentation.report_generator import ReportGenerator
from ai.journal_dataset_research.documentation.research_logger import ResearchLogger
from ai.journal_dataset_research.documentation.tracking_updater import (
    TrackingDocumentUpdater,
)
from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchProgress,
    ResearchSession,
    WeeklyReport,
)


class TestResearchLogger:
    """Tests for ResearchLogger."""

    def test_log_activity(self, temp_log_dir):
        """Test logging an activity."""
        logger = ResearchLogger(log_directory=str(temp_log_dir))
        log_entry = logger.log_activity(
            activity_type="search",
            description="Searching for datasets",
            outcome="success",
            duration_minutes=5,
            source_id="test-001",
        )

        assert log_entry.activity_type == "search"
        assert log_entry.description == "Searching for datasets"
        assert log_entry.outcome == "success"
        assert log_entry.source_id == "test-001"
        assert log_entry.duration_minutes == 5

    def test_log_activity_invalid_type(self, temp_log_dir):
        """Test that invalid activity type raises error."""
        logger = ResearchLogger(log_directory=str(temp_log_dir))
        with pytest.raises(ValueError, match="Invalid log entry"):
            logger.log_activity(
                activity_type="invalid_activity",
                description="Test",
                outcome="success",
            )

    def test_get_session_logs(self, temp_log_dir):
        """Test getting logs for a session."""
        logger = ResearchLogger(log_directory=str(temp_log_dir))
        session_id = "test-session-001"

        logger.log_activity(
            activity_type="search",
            description="Search 1",
            session_id=session_id,
        )
        logger.log_activity(
            activity_type="evaluation",
            description="Evaluation 1",
            session_id=session_id,
        )

        logs = logger.get_session_logs(session_id)
        assert len(logs) == 2
        assert logs[0].activity_type == "search"
        assert logs[1].activity_type == "evaluation"

    def test_query_logs_by_activity_type(self, temp_log_dir):
        """Test querying logs by activity type."""
        logger = ResearchLogger(log_directory=str(temp_log_dir))

        logger.log_activity(activity_type="search", description="Search 1")
        logger.log_activity(activity_type="search", description="Search 2")
        logger.log_activity(activity_type="evaluation", description="Evaluation 1")

        search_logs = logger.query_logs(activity_type="search")
        assert len(search_logs) == 2
        assert all(log.activity_type == "search" for log in search_logs)

    def test_query_logs_by_source_id(self, temp_log_dir):
        """Test querying logs by source ID."""
        logger = ResearchLogger(log_directory=str(temp_log_dir))

        logger.log_activity(
            activity_type="search",
            description="Search",
            source_id="test-001",
        )
        logger.log_activity(
            activity_type="evaluation",
            description="Evaluation",
            source_id="test-001",
        )
        logger.log_activity(
            activity_type="search",
            description="Search",
            source_id="test-002",
        )

        logs = logger.query_logs(source_id="test-001")
        assert len(logs) == 2
        assert all(log.source_id == "test-001" for log in logs)

    def test_rotate_logs(self, temp_log_dir):
        """Test log rotation."""
        logger = ResearchLogger(
            log_directory=str(temp_log_dir),
            max_log_size_mb=0.001,  # Very small to trigger rotation
        )

        # Log many activities to trigger rotation
        for i in range(100):
            logger.log_activity(
                activity_type="search",
                description=f"Search {i}",
            )

        # Rotation should have occurred
        log_files = list(temp_log_dir.glob("*.log"))
        assert len(log_files) > 0

    def test_archive_logs(self, temp_log_dir):
        """Test log archival."""
        logger = ResearchLogger(
            log_directory=str(temp_log_dir),
            max_log_age_days=0,  # Archive immediately
            enable_archival=True,
        )

        logger.log_activity(activity_type="search", description="Test")
        logger.archive_old_logs()

        # Check that archived directory exists
        archived_dir = temp_log_dir / "archived"
        assert archived_dir.exists()


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_generate_evaluation_report(self, temp_report_dir, sample_dataset_source, sample_evaluation):
        """Test generating an evaluation report."""
        generator = ReportGenerator(output_directory=str(temp_report_dir))
        report_path = generator.generate_evaluation_report(
            source=sample_dataset_source,
            evaluation=sample_evaluation,
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Dataset Evaluation Report" in content
        assert sample_dataset_source.source_id in content
        assert str(sample_evaluation.overall_score) in content
        assert sample_evaluation.priority_tier.upper() in content

    def test_generate_weekly_report(self, temp_report_dir, sample_research_session):
        """Test generating a weekly report."""
        generator = ReportGenerator(output_directory=str(temp_report_dir))
        
        progress = ResearchProgress(
            sources_identified=10,
            datasets_evaluated=5,
            datasets_acquired=2,
        )

        report_path = generator.generate_weekly_report(
            session=sample_research_session,
            progress=progress,
            week_number=1,
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Weekly Research Report" in content
        assert "10" in content  # sources_identified
        assert "5" in content  # datasets_evaluated

    def test_generate_research_summary_report(
        self, temp_report_dir, multiple_sources, sample_evaluation
    ):
        """Test generating a research summary report."""
        generator = ReportGenerator(output_directory=str(temp_report_dir))
        
        evaluations = [sample_evaluation]

        report_path = generator.generate_research_summary_report(
            sources=multiple_sources,
            evaluations=evaluations,
            start_date=datetime.now() - timedelta(days=7),
            end_date=datetime.now(),
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "# Research Summary Report" in content
        assert str(len(multiple_sources)) in content


class TestDatasetCatalog:
    """Tests for DatasetCatalog."""

    def test_add_source(self, sample_dataset_source):
        """Test adding a source to the catalog."""
        catalog = DatasetCatalog()
        catalog.add_source(sample_dataset_source)

        assert len(catalog.sources) == 1
        assert catalog.sources[0].source_id == sample_dataset_source.source_id

    def test_add_evaluation(self, sample_evaluation):
        """Test adding an evaluation to the catalog."""
        catalog = DatasetCatalog()
        catalog.add_evaluation(sample_evaluation)

        assert len(catalog.evaluations) == 1
        assert catalog.evaluations[0].source_id == sample_evaluation.source_id

    def test_add_acquired_dataset(self, sample_acquired_dataset):
        """Test adding an acquired dataset to the catalog."""
        catalog = DatasetCatalog()
        catalog.add_acquired_dataset(sample_acquired_dataset)

        assert len(catalog.acquired_datasets) == 1
        assert catalog.acquired_datasets[0].source_id == sample_acquired_dataset.source_id

    def test_add_integration_plan(self, sample_integration_plan):
        """Test adding an integration plan to the catalog."""
        catalog = DatasetCatalog()
        catalog.add_integration_plan(sample_integration_plan)

        assert len(catalog.integration_plans) == 1
        assert catalog.integration_plans[0].source_id == sample_integration_plan.source_id

    def test_export_markdown(self, temp_dir, sample_dataset_source, sample_evaluation):
        """Test exporting catalog to markdown."""
        catalog = DatasetCatalog()
        catalog.add_source(sample_dataset_source)
        catalog.add_evaluation(sample_evaluation)

        output_path = temp_dir / "catalog.md"
        catalog.export_markdown(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Dataset Catalog" in content
        assert sample_dataset_source.source_id in content

    def test_export_csv(self, temp_dir, sample_dataset_source):
        """Test exporting catalog to CSV."""
        catalog = DatasetCatalog()
        catalog.add_source(sample_dataset_source)

        output_path = temp_dir / "catalog.csv"
        catalog.export_csv(output_path)

        assert output_path.exists()
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(output_path)
        assert len(df) >= 1
        assert "source_id" in df.columns

    def test_export_json(self, temp_dir, sample_dataset_source):
        """Test exporting catalog to JSON."""
        catalog = DatasetCatalog()
        catalog.add_source(sample_dataset_source)

        output_path = temp_dir / "catalog.json"
        catalog.export_json(output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "sources" in data
        assert len(data["sources"]) == 1

    def test_get_statistics(self, sample_dataset_source, sample_evaluation, sample_acquired_dataset):
        """Test getting catalog statistics."""
        catalog = DatasetCatalog()
        catalog.add_source(sample_dataset_source)
        catalog.add_evaluation(sample_evaluation)
        catalog.add_acquired_dataset(sample_acquired_dataset)

        stats = catalog.get_statistics()

        assert stats["total_sources"] == 1
        assert stats["total_evaluations"] == 1
        assert stats["total_acquired_datasets"] == 1
        assert "average_score" in stats
        assert "high_priority_count" in stats


class TestProgressVisualization:
    """Tests for ProgressVisualization."""

    def test_generate_progress_chart_data(self, sample_research_session):
        """Test generating progress chart data."""
        viz = ProgressVisualization()
        
        progress_history = [
            ResearchProgress(sources_identified=5, last_updated=datetime.now() - timedelta(days=2)),
            ResearchProgress(sources_identified=10, last_updated=datetime.now() - timedelta(days=1)),
            ResearchProgress(sources_identified=15, last_updated=datetime.now()),
        ]

        chart_data = viz.generate_progress_chart_data(
            session=sample_research_session,
            progress_history=progress_history,
        )

        assert "labels" in chart_data
        assert "datasets" in chart_data
        assert len(chart_data["labels"]) == 3

    def test_generate_timeline_visualization(self, sample_research_session):
        """Test generating timeline visualization."""
        viz = ProgressVisualization()
        
        progress_history = [
            ResearchProgress(sources_identified=i, last_updated=datetime.now() - timedelta(days=10-i))
            for i in range(10)
        ]

        timeline_data = viz.generate_timeline_visualization(
            session=sample_research_session,
            progress_history=progress_history,
        )

        assert "timeline" in timeline_data
        assert len(timeline_data["timeline"]) == 10

    def test_generate_quality_score_distribution(self, sample_evaluation, high_score_evaluation, low_score_evaluation):
        """Test generating quality score distribution."""
        viz = ProgressVisualization()
        
        evaluations = [sample_evaluation, high_score_evaluation, low_score_evaluation]

        distribution = viz.generate_quality_score_distribution(evaluations)

        assert "scores" in distribution
        assert "distribution" in distribution
        assert len(distribution["scores"]) == len(evaluations)

    def test_export_visualization_html(self, temp_dir, sample_research_session):
        """Test exporting visualization to HTML."""
        viz = ProgressVisualization()
        
        progress_history = [
            ResearchProgress(sources_identified=10, last_updated=datetime.now()),
        ]

        output_path = temp_dir / "visualization.html"
        viz.export_visualization_html(
            session=sample_research_session,
            progress_history=progress_history,
            output_path=output_path,
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "<html" in content.lower()
        assert "chart" in content.lower() or "visualization" in content.lower()


class TestTrackingDocumentUpdater:
    """Tests for TrackingDocumentUpdater."""

    def test_update_progress_section(self, temp_dir, sample_research_session):
        """Test updating progress section in tracking document."""
        tracking_file = temp_dir / "JOURNAL_RESEARCH_TARGETS.md"
        tracking_file.write_text("""# Research Targets

<!-- PROGRESS_METRICS_START -->
Old progress
<!-- PROGRESS_METRICS_END -->
""")

        updater = TrackingDocumentUpdater(tracking_document_path=str(tracking_file))
        
        progress = ResearchProgress(
            sources_identified=10,
            datasets_evaluated=5,
            datasets_acquired=2,
        )

        updater.update_progress_section(progress=progress, session=sample_research_session)

        content = tracking_file.read_text()
        assert "10" in content  # sources_identified
        assert "5" in content  # datasets_evaluated

    def test_mark_task_completed(self, temp_dir):
        """Test marking a task as completed."""
        tracking_file = temp_dir / "JOURNAL_RESEARCH_TARGETS.md"
        tracking_file.write_text("""# Research Targets

## Tasks

<!-- COMPLETED_TASKS_START -->
<!-- COMPLETED_TASKS_END -->
""")

        updater = TrackingDocumentUpdater(tracking_document_path=str(tracking_file))
        
        updater.mark_task_completed(
            task_id="task-001",
            task_description="Test task",
        )

        content = tracking_file.read_text()
        assert "task-001" in content
        assert "Test task" in content
        assert "[x]" in content

    def test_update_status_summary(self, temp_dir, sample_research_session):
        """Test updating status summary."""
        tracking_file = temp_dir / "JOURNAL_RESEARCH_TARGETS.md"
        tracking_file.write_text("""# Research Targets

<!-- STATUS_SUMMARY_START -->
Old status
<!-- STATUS_SUMMARY_END -->
""")

        updater = TrackingDocumentUpdater(tracking_document_path=str(tracking_file))
        
        progress = ResearchProgress(
            sources_identified=10,
            datasets_evaluated=5,
        )

        updater.update_status_summary(progress=progress, session=sample_research_session)

        content = tracking_file.read_text()
        assert "10" in content or "5" in content

    def test_update_with_weekly_report(self, temp_dir, sample_research_session):
        """Test updating with weekly report."""
        tracking_file = temp_dir / "JOURNAL_RESEARCH_TARGETS.md"
        tracking_file.write_text("""# Research Targets

<!-- STATUS_SUMMARY_START -->
<!-- STATUS_SUMMARY_END -->
""")

        updater = TrackingDocumentUpdater(tracking_document_path=str(tracking_file))
        
        progress = ResearchProgress(sources_identified=10)
        
        weekly_report = WeeklyReport(
            week_number=1,
            start_date=datetime.now(),
            end_date=datetime.now(),
            sources_identified=10,
            datasets_evaluated=5,
        )

        updater.update_status_summary(
            progress=progress,
            session=sample_research_session,
            weekly_report=weekly_report,
        )

        content = tracking_file.read_text()
        assert "Week 1" in content or "10" in content

