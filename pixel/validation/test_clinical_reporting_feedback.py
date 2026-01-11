"""
Unit tests for Clinical Reporting and Feedback Loop System
"""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from automated_clinical_appropriateness import ClinicalAppropriatenessChecker
from clinical_accuracy_assessment import (
    ClinicalAccuracyAssessmentFramework,
    ClinicalDomain,
)
from clinical_reporting_feedback import (
    AlertSeverity,
    ClinicalReportingFeedbackSystem,
    FeedbackItem,
    FeedbackType,
    ImprovementAction,
    ReportType,
    ValidationMetrics,
)
from expert_validation_workflow import ExpertValidationWorkflow
from safety_ethics_compliance import SafetyEthicsComplianceValidator


class TestValidationMetrics:
    """Test validation metrics data structure."""

    def test_validation_metrics_creation(self):
        """Test creating validation metrics."""
        metrics = ValidationMetrics(
            total_validations=100,
            average_score=0.85,
            expert_consensus_rate=0.9,
            compliance_rate=0.95,
            safety_incidents=2,
        )

        assert metrics.total_validations == 100
        assert metrics.average_score == 0.85
        assert metrics.expert_consensus_rate == 0.9
        assert metrics.compliance_rate == 0.95
        assert metrics.safety_incidents == 2
        assert isinstance(metrics.timestamp, datetime)


class TestFeedbackItem:
    """Test feedback item functionality."""

    def test_feedback_item_creation(self):
        """Test creating feedback item."""
        item = FeedbackItem(
            feedback_id="test_feedback_001",
            feedback_type=FeedbackType.PERFORMANCE_METRICS,
            content_id="test_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            message="Test feedback message",
            severity=AlertSeverity.HIGH,
            actionable=True,
            recommendations=["Fix this", "Improve that"],
        )

        assert item.feedback_id == "test_feedback_001"
        assert item.feedback_type == FeedbackType.PERFORMANCE_METRICS
        assert item.severity == AlertSeverity.HIGH
        assert item.actionable is True
        assert len(item.recommendations) == 2
        assert not item.resolved


class TestImprovementAction:
    """Test improvement action functionality."""

    def test_improvement_action_creation(self):
        """Test creating improvement action."""
        action = ImprovementAction(
            action_id="action_001",
            action_type="accuracy_improvement",
            description="Improve accuracy metrics",
            target_domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            priority=AlertSeverity.HIGH,
            estimated_impact=0.15,
            implementation_steps=["Step 1", "Step 2"],
            success_metrics=["Metric 1", "Metric 2"],
        )

        assert action.action_id == "action_001"
        assert action.action_type == "accuracy_improvement"
        assert action.target_domain == ClinicalDomain.DSM5_DIAGNOSTIC
        assert action.priority == AlertSeverity.HIGH
        assert action.estimated_impact == 0.15
        assert action.status == "pending"


class TestClinicalReportingFeedbackSystem:
    """Test the main reporting and feedback system."""

    @pytest.fixture
    def validation_systems(self):
        """Create mock validation systems for testing."""
        assessment_framework = ClinicalAccuracyAssessmentFramework()
        expert_workflow = ExpertValidationWorkflow(assessment_framework)
        appropriateness_checker = ClinicalAppropriatenessChecker()
        compliance_validator = SafetyEthicsComplianceValidator()

        return {
            "assessment": assessment_framework,
            "expert": expert_workflow,
            "appropriateness": appropriateness_checker,
            "compliance": compliance_validator,
        }

    @pytest.fixture
    def reporting_system(self, validation_systems):
        """Create reporting system for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        system = ClinicalReportingFeedbackSystem(
            assessment_framework=validation_systems["assessment"],
            expert_workflow=validation_systems["expert"],
            appropriateness_checker=validation_systems["appropriateness"],
            compliance_validator=validation_systems["compliance"],
            db_path=db_path,
        )

        yield system

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_system_initialization(self, reporting_system):
        """Test system initialization."""
        assert reporting_system.db_path is not None
        assert Path(reporting_system.db_path).exists()
        assert isinstance(reporting_system.config, dict)
        assert "alert_thresholds" in reporting_system.config

    def test_database_initialization(self, reporting_system):
        """Test database initialization."""
        conn = sqlite3.connect(reporting_system.db_path)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "validation_metrics" in tables
        assert "feedback_items" in tables
        assert "improvement_actions" in tables

        conn.close()

    def test_generate_daily_summary_report(self, reporting_system):
        """Test generating daily summary report."""
        report = reporting_system.generate_comprehensive_report(ReportType.DAILY_SUMMARY)

        assert report["report_type"] == "daily_summary"
        assert "date" in report
        assert "summary" in report
        assert "accuracy_metrics" in report
        assert "expert_metrics" in report
        assert "appropriateness_metrics" in report
        assert "compliance_metrics" in report
        assert "alerts" in report
        assert "recommendations" in report

    def test_generate_weekly_analysis_report(self, reporting_system):
        """Test generating weekly analysis report."""
        report = reporting_system.generate_comprehensive_report(ReportType.WEEKLY_ANALYSIS)

        assert isinstance(report, dict)
        # Weekly analysis would have different structure than daily summary

    def test_calculate_accuracy_metrics_empty(self, reporting_system):
        """Test calculating accuracy metrics with empty data."""
        metrics = reporting_system._calculate_accuracy_metrics([])

        assert metrics["average_score"] == 0.0
        assert metrics["total"] == 0
        assert metrics["distribution"] == {}

    def test_calculate_accuracy_metrics_with_data(self, reporting_system, validation_systems):
        """Test calculating accuracy metrics with sample data."""
        # Create sample assessments
        assessment_framework = validation_systems["assessment"]

        assessment_id = assessment_framework.create_assessment(
            content_id="test_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="test_assessor",
        )

        assessment_framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Test content",
            assessor_id="test_assessor",
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
        )

        assessments = list(assessment_framework.assessments.values())
        metrics = reporting_system._calculate_accuracy_metrics(assessments)

        assert metrics["total"] == 1
        assert metrics["average_score"] > 0
        assert "distribution" in metrics

    def test_calculate_expert_metrics_empty(self, reporting_system):
        """Test calculating expert metrics with empty data."""
        metrics = reporting_system._calculate_expert_metrics([])

        assert metrics["total"] == 0
        assert metrics["completed"] == 0
        assert metrics["consensus_rate"] == 0.0

    def test_process_feedback_loop(self, reporting_system):
        """Test processing feedback loop."""
        actions = reporting_system.process_feedback_loop()

        assert isinstance(actions, list)
        # With empty data, should return empty list or minimal actions

    def test_store_feedback_items(self, reporting_system):
        """Test storing feedback items in database."""
        feedback_items = [
            FeedbackItem(
                feedback_id="test_001",
                feedback_type=FeedbackType.PERFORMANCE_METRICS,
                content_id="test_content",
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                message="Test feedback",
                severity=AlertSeverity.MEDIUM,
                actionable=True,
                recommendations=["Test recommendation"],
            )
        ]

        reporting_system._store_feedback_items(feedback_items)

        # Verify storage
        conn = sqlite3.connect(reporting_system.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM feedback_items")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_store_improvement_actions(self, reporting_system):
        """Test storing improvement actions in database."""
        actions = [
            ImprovementAction(
                action_id="action_001",
                action_type="test_improvement",
                description="Test improvement action",
                priority=AlertSeverity.HIGH,
                estimated_impact=0.1,
                implementation_steps=["Step 1"],
                success_metrics=["Metric 1"],
            )
        ]

        reporting_system._store_improvement_actions(actions)

        # Verify storage
        conn = sqlite3.connect(reporting_system.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM improvement_actions")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_get_recent_feedback_items(self, reporting_system):
        """Test getting recent feedback items."""
        # Store a feedback item first
        feedback_items = [
            FeedbackItem(
                feedback_id="recent_001",
                feedback_type=FeedbackType.ALERT,
                content_id="test_content",
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                message="Recent feedback",
                severity=AlertSeverity.LOW,
                actionable=False,
                recommendations=[],
            )
        ]

        reporting_system._store_feedback_items(feedback_items)

        # Retrieve recent items
        recent_items = reporting_system._get_recent_feedback_items(hours=24)

        assert len(recent_items) == 1
        assert recent_items[0].feedback_id == "recent_001"

    def test_get_active_improvement_actions(self, reporting_system):
        """Test getting active improvement actions."""
        # Store an improvement action first
        actions = [
            ImprovementAction(
                action_id="active_001",
                action_type="active_improvement",
                description="Active improvement action",
                priority=AlertSeverity.MEDIUM,
                estimated_impact=0.05,
                implementation_steps=["Active step"],
                success_metrics=["Active metric"],
                status="pending",
            )
        ]

        reporting_system._store_improvement_actions(actions)

        # Retrieve active actions
        active_actions = reporting_system._get_active_improvement_actions()

        assert len(active_actions) == 1
        assert active_actions[0].action_id == "active_001"
        assert active_actions[0].status == "pending"

    def test_get_dashboard_data(self, reporting_system):
        """Test getting dashboard data."""
        dashboard_data = reporting_system.get_dashboard_data()

        assert "timestamp" in dashboard_data
        assert "current_metrics" in dashboard_data
        assert "recent_feedback" in dashboard_data
        assert "active_actions" in dashboard_data
        assert "health_indicators" in dashboard_data

        # Verify structure
        assert isinstance(dashboard_data["recent_feedback"], list)
        assert isinstance(dashboard_data["active_actions"], list)
        assert isinstance(dashboard_data["health_indicators"], dict)

    def test_get_system_health_indicators(self, reporting_system):
        """Test getting system health indicators."""
        health = reporting_system._get_system_health_indicators()

        assert "database_status" in health
        assert "validation_systems_status" in health
        assert "expert_workflow_status" in health
        assert "feedback_loop_status" in health
        assert "last_update" in health

        assert health["database_status"] == "healthy"

    def test_generate_feedback_items_low_accuracy(self, reporting_system):
        """Test generating feedback items for low accuracy."""
        # Mock performance data with low accuracy
        performance_data = {
            "assessments": [
                Mock(overall_score=0.6),
                Mock(overall_score=0.65),
                Mock(overall_score=0.55),
            ],
            "expert_requests": [],
            "appropriateness_results": [],
            "compliance_results": [],
            "period": {"start": datetime.now() - timedelta(days=7), "end": datetime.now()},
        }

        feedback_items = reporting_system._generate_feedback_items(performance_data)

        # Should generate feedback for low accuracy
        accuracy_feedback = [f for f in feedback_items if "accuracy" in f.message.lower()]
        assert len(accuracy_feedback) > 0
        assert accuracy_feedback[0].severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]

    def test_create_improvement_actions_from_feedback(self, reporting_system):
        """Test creating improvement actions from feedback."""
        feedback_items = [
            FeedbackItem(
                feedback_id="test_feedback",
                feedback_type=FeedbackType.PERFORMANCE_METRICS,
                content_id="test_content",
                domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                message="Low accuracy detected",
                severity=AlertSeverity.HIGH,
                actionable=True,
                recommendations=["Improve accuracy"],
            )
        ]

        actions = reporting_system._create_improvement_actions(feedback_items)

        assert len(actions) > 0
        assert actions[0].action_type == "accuracy_improvement"
        assert actions[0].priority == AlertSeverity.HIGH
        assert len(actions[0].implementation_steps) > 0

    @pytest.mark.asyncio
    async def test_continuous_monitoring_single_iteration(self, reporting_system):
        """Test single iteration of continuous monitoring."""
        # Mock the continuous monitoring to run only once
        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            if seconds == 3600:  # Main interval
                raise KeyboardInterrupt("Stop monitoring")
            else:
                await original_sleep(seconds)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            try:
                await reporting_system.run_continuous_monitoring(interval_seconds=3600)
            except KeyboardInterrupt:
                pass  # Expected to stop monitoring

        # Should have processed at least one feedback loop iteration
        # This is mainly testing that the method doesn't crash


class TestReportGeneration:
    """Test different types of report generation."""

    @pytest.fixture
    def reporting_system_with_data(self, validation_systems):
        """Create reporting system with sample data."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        system = ClinicalReportingFeedbackSystem(
            assessment_framework=validation_systems["assessment"],
            expert_workflow=validation_systems["expert"],
            appropriateness_checker=validation_systems["appropriateness"],
            compliance_validator=validation_systems["compliance"],
            db_path=db_path,
        )

        # Add some sample data
        assessment_framework = validation_systems["assessment"]
        assessment_id = assessment_framework.create_assessment(
            content_id="sample_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="sample_assessor",
        )

        assessment_framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Sample therapeutic response",
            assessor_id="sample_assessor",
            individual_scores={"dsm5_diagnostic_accuracy": 0.8},
            feedback={"dsm5_diagnostic_accuracy": "Good assessment"},
        )

        yield system

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_domain_analysis_report(self, reporting_system_with_data):
        """Test generating domain analysis report."""
        report = reporting_system_with_data.generate_comprehensive_report(
            ReportType.DOMAIN_ANALYSIS, domain_filter=ClinicalDomain.DSM5_DIAGNOSTIC
        )

        assert isinstance(report, dict)
        # Domain analysis should focus on specific domain

    def test_expert_performance_report(self, reporting_system_with_data):
        """Test generating expert performance report."""
        report = reporting_system_with_data.generate_comprehensive_report(
            ReportType.EXPERT_PERFORMANCE
        )

        assert isinstance(report, dict)
        # Expert performance report should analyze expert metrics

    def test_compliance_audit_report(self, reporting_system_with_data):
        """Test generating compliance audit report."""
        report = reporting_system_with_data.generate_comprehensive_report(
            ReportType.COMPLIANCE_AUDIT
        )

        assert isinstance(report, dict)
        # Compliance audit should focus on compliance metrics

    def test_improvement_tracking_report(self, reporting_system_with_data):
        """Test generating improvement tracking report."""
        report = reporting_system_with_data.generate_comprehensive_report(
            ReportType.IMPROVEMENT_TRACKING
        )

        assert isinstance(report, dict)
        # Improvement tracking should show progress over time

    def test_alert_dashboard_report(self, reporting_system_with_data):
        """Test generating alert dashboard report."""
        report = reporting_system_with_data.generate_comprehensive_report(
            ReportType.ALERT_DASHBOARD
        )

        assert isinstance(report, dict)
        # Alert dashboard should show current alerts and issues


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    @pytest.fixture
    def complete_system(self):
        """Set up complete system with all components."""
        assessment_framework = ClinicalAccuracyAssessmentFramework()
        expert_workflow = ExpertValidationWorkflow(assessment_framework)
        appropriateness_checker = ClinicalAppropriatenessChecker()
        compliance_validator = SafetyEthicsComplianceValidator()

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        reporting_system = ClinicalReportingFeedbackSystem(
            assessment_framework=assessment_framework,
            expert_workflow=expert_workflow,
            appropriateness_checker=appropriateness_checker,
            compliance_validator=compliance_validator,
            db_path=db_path,
        )

        yield {
            "reporting": reporting_system,
            "assessment": assessment_framework,
            "expert": expert_workflow,
            "appropriateness": appropriateness_checker,
            "compliance": compliance_validator,
        }

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_end_to_end_feedback_loop(self, complete_system):
        """Test complete end-to-end feedback loop."""
        reporting_system = complete_system["reporting"]
        assessment_framework = complete_system["assessment"]

        # Step 1: Create and conduct assessment
        assessment_id = assessment_framework.create_assessment(
            content_id="e2e_test_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="e2e_assessor",
        )

        assessment_framework.conduct_assessment(
            assessment_id=assessment_id,
            content="End-to-end test content",
            assessor_id="e2e_assessor",
            individual_scores={"dsm5_diagnostic_accuracy": 0.6},  # Low score to trigger feedback
            feedback={"dsm5_diagnostic_accuracy": "Needs improvement"},
        )

        # Step 2: Process feedback loop
        improvement_actions = reporting_system.process_feedback_loop()

        # Step 3: Generate comprehensive report
        report = reporting_system.generate_comprehensive_report(ReportType.DAILY_SUMMARY)

        # Step 4: Get dashboard data
        dashboard = reporting_system.get_dashboard_data()

        # Verify the complete flow worked
        assert isinstance(improvement_actions, list)
        assert isinstance(report, dict)
        assert isinstance(dashboard, dict)
        assert "current_metrics" in dashboard
        assert report["summary"]["total_assessments"] >= 1

    def test_multi_system_integration(self, complete_system):
        """Test integration across all validation systems."""
        reporting_system = complete_system["reporting"]
        assessment_framework = complete_system["assessment"]
        expert_workflow = complete_system["expert"]
        appropriateness_checker = complete_system["appropriateness"]
        compliance_validator = complete_system["compliance"]

        # Create data in multiple systems
        # 1. Assessment
        assessment_id = assessment_framework.create_assessment(
            content_id="multi_test_content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            assessor_id="multi_assessor",
        )

        assessment_framework.conduct_assessment(
            assessment_id=assessment_id,
            content="Multi-system test content",
            assessor_id="multi_assessor",
            individual_scores={"dsm5_diagnostic_accuracy": 0.85},
            feedback={"dsm5_diagnostic_accuracy": "Good work"},
        )

        # 2. Expert validation request
        expert_workflow.create_validation_request(
            content_id="multi_test_content",
            content_text="Multi-system test content",
            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
            requester_id="multi_requester",
        )

        # 3. Appropriateness check
        appropriateness_result = appropriateness_checker.check_appropriateness(
            content_id="multi_test_content", content_text="Multi-system test content"
        )

        # 4. Compliance validation
        compliance_result = compliance_validator.validate_compliance(
            content_id="multi_test_content", content_text="Multi-system test content"
        )

        # Generate integrated report
        report = reporting_system.generate_comprehensive_report(ReportType.DAILY_SUMMARY)

        # Verify integration
        assert report["summary"]["total_assessments"] >= 1
        assert "accuracy_metrics" in report
        assert "expert_metrics" in report
        assert "appropriateness_metrics" in report
        assert "compliance_metrics" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
