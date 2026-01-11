"""
Unit tests for Clinical Reporting and Feedback Loop System

This module provides comprehensive unit tests for the clinical reporting
system, covering performance analytics, trend analysis, improvement
recommendations, and feedback mechanisms.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from .clinical_accuracy_validator import (
    ClinicalAccuracyLevel,
    ClinicalAccuracyResult,
    ClinicalContext,
    DSM5Assessment,
    PDM2Assessment,
    SafetyAssessment,
    SafetyRiskLevel,
    TherapeuticAppropriatenessScore,
    TherapeuticModality,
)
from .clinical_reporting_system import (
    ClinicalReport,
    ClinicalReportingSystem,
    FeedbackMessage,
    FeedbackType,
    PerformanceMetric,
    PerformanceSnapshot,
    ReportType,
)


class TestClinicalReportingSystem:
    """Test suite for ClinicalReportingSystem"""

    @pytest.fixture
    def reporting_system(self):
        """Create a reporting system instance for testing"""
        return ClinicalReportingSystem()

    @pytest.fixture
    def sample_context(self):
        """Create a sample clinical context"""
        return ClinicalContext(
            client_presentation="Sample client case",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="working",
        )

    @pytest.fixture
    def sample_assessment_result(self, sample_context):
        """Create a sample assessment result"""
        return ClinicalAccuracyResult(
            assessment_id="test_assessment_001",
            timestamp=datetime.now(),
            clinical_context=sample_context,
            dsm5_assessment=DSM5Assessment(
                primary_diagnosis="Major Depressive Disorder", diagnostic_confidence=0.85
            ),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.8),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.LOW),
            overall_accuracy=ClinicalAccuracyLevel.GOOD,
            confidence_score=0.82,
            expert_validation_needed=False,
            recommendations=["Continue current approach"],
            warnings=[],
        )

    @pytest.fixture
    def excellent_assessment_result(self, sample_context):
        """Create an excellent assessment result"""
        return ClinicalAccuracyResult(
            assessment_id="excellent_001",
            timestamp=datetime.now(),
            clinical_context=sample_context,
            dsm5_assessment=DSM5Assessment(
                primary_diagnosis="Anxiety Disorder", diagnostic_confidence=0.95
            ),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.95),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.MINIMAL),
            overall_accuracy=ClinicalAccuracyLevel.EXCELLENT,
            confidence_score=0.95,
            expert_validation_needed=False,
            recommendations=[],
            warnings=[],
        )

    @pytest.fixture
    def dangerous_assessment_result(self, sample_context):
        """Create a dangerous assessment result"""
        return ClinicalAccuracyResult(
            assessment_id="dangerous_001",
            timestamp=datetime.now(),
            clinical_context=sample_context,
            dsm5_assessment=DSM5Assessment(primary_diagnosis=None, diagnostic_confidence=0.2),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.3),
            safety_assessment=SafetyAssessment(
                overall_risk=SafetyRiskLevel.CRITICAL, safety_plan_needed=True
            ),
            overall_accuracy=ClinicalAccuracyLevel.DANGEROUS,
            confidence_score=0.25,
            expert_validation_needed=True,
            recommendations=["Immediate intervention required"],
            warnings=["Critical safety risk"],
        )

    def test_reporting_system_initialization(self, reporting_system):
        """Test reporting system initialization"""
        assert reporting_system is not None
        assert isinstance(reporting_system.assessment_results, list)
        assert isinstance(reporting_system.expert_validations, list)
        assert isinstance(reporting_system.appropriateness_checks, list)
        assert isinstance(reporting_system.safety_compliance_results, list)
        assert isinstance(reporting_system.performance_history, list)
        assert isinstance(reporting_system.generated_reports, dict)
        assert isinstance(reporting_system.feedback_messages, list)

    def test_config_loading_default(self):
        """Test default configuration loading"""
        reporting_system = ClinicalReportingSystem()

        assert reporting_system.config["cache_duration_minutes"] == 30
        assert "excellent" in reporting_system.config["performance_thresholds"]
        assert reporting_system.config["feedback_frequency"]["immediate"] is True

    def test_config_loading_custom(self):
        """Test custom configuration loading"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            custom_config = {
                "cache_duration_minutes": 60,
                "performance_thresholds": {"excellent": 0.95},
            }
            json.dump(custom_config, f)
            config_path = Path(f.name)

        try:
            reporting_system = ClinicalReportingSystem(config_path)
            assert reporting_system.config["cache_duration_minutes"] == 60
            assert reporting_system.config["performance_thresholds"]["excellent"] == 0.95
        finally:
            config_path.unlink()

    def test_add_assessment_result(self, reporting_system, sample_assessment_result):
        """Test adding assessment result"""
        initial_count = len(reporting_system.assessment_results)

        reporting_system.add_assessment_result(sample_assessment_result)

        assert len(reporting_system.assessment_results) == initial_count + 1
        assert reporting_system.assessment_results[-1] == sample_assessment_result

    def test_cache_invalidation(self, reporting_system, sample_assessment_result):
        """Test cache invalidation when adding new data"""
        # Add some data to cache
        reporting_system._analytics_cache["test_key"] = "test_value"
        reporting_system._cache_expiry["test_key"] = datetime.now() + timedelta(hours=1)

        # Add assessment result (should invalidate cache)
        reporting_system.add_assessment_result(sample_assessment_result)

        # Cache should be empty
        assert len(reporting_system._analytics_cache) == 0
        assert len(reporting_system._cache_expiry) == 0

    @pytest.mark.asyncio
    async def test_generate_individual_assessment_report(
        self, reporting_system, sample_assessment_result
    ):
        """Test generating individual assessment report"""
        # Add assessment to system
        reporting_system.add_assessment_result(sample_assessment_result)

        # Generate report
        report = await reporting_system.generate_individual_assessment_report(
            sample_assessment_result.assessment_id
        )

        assert report is not None
        assert isinstance(report, ClinicalReport)
        assert report.report_type == ReportType.INDIVIDUAL_ASSESSMENT
        assert report.report_id.startswith("individual_")
        assert len(report.summary) > 0
        assert isinstance(report.key_findings, list)
        assert isinstance(report.performance_metrics, dict)
        assert isinstance(report.improvement_recommendations, list)

    @pytest.mark.asyncio
    async def test_generate_individual_report_not_found(self, reporting_system):
        """Test generating report for non-existent assessment"""
        report = await reporting_system.generate_individual_assessment_report("nonexistent_id")

        assert report is None

    @pytest.mark.asyncio
    async def test_generate_aggregate_performance_report(
        self, reporting_system, sample_assessment_result, excellent_assessment_result
    ):
        """Test generating aggregate performance report"""
        # Add multiple assessments
        reporting_system.add_assessment_result(sample_assessment_result)
        reporting_system.add_assessment_result(excellent_assessment_result)

        # Generate report for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        report = await reporting_system.generate_aggregate_performance_report(start_date, end_date)

        assert isinstance(report, ClinicalReport)
        assert report.report_type == ReportType.AGGREGATE_PERFORMANCE
        assert report.report_id.startswith("aggregate_")
        assert len(report.summary) > 0
        assert "total_assessments" in report.performance_metrics
        assert report.performance_metrics["total_assessments"] == 2
        assert isinstance(report.trend_analyses, list)

    @pytest.mark.asyncio
    async def test_generate_aggregate_report_empty_data(self, reporting_system):
        """Test generating aggregate report with no data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        report = await reporting_system.generate_aggregate_performance_report(start_date, end_date)

        assert isinstance(report, ClinicalReport)
        assert report.performance_metrics["total_assessments"] == 0
        assert report.performance_metrics["overall_accuracy"] == 0.0

    def test_accuracy_to_score_conversion(self, reporting_system):
        """Test conversion of accuracy levels to numeric scores"""
        assert reporting_system._accuracy_to_score(ClinicalAccuracyLevel.EXCELLENT) == 1.0
        assert reporting_system._accuracy_to_score(ClinicalAccuracyLevel.GOOD) == 0.8
        assert reporting_system._accuracy_to_score(ClinicalAccuracyLevel.ACCEPTABLE) == 0.6
        assert reporting_system._accuracy_to_score(ClinicalAccuracyLevel.CONCERNING) == 0.4
        assert reporting_system._accuracy_to_score(ClinicalAccuracyLevel.DANGEROUS) == 0.0

    def test_safety_risk_to_score_conversion(self, reporting_system):
        """Test conversion of safety risk levels to numeric scores"""
        assert reporting_system._safety_risk_to_score(SafetyRiskLevel.MINIMAL) == 1.0
        assert reporting_system._safety_risk_to_score(SafetyRiskLevel.LOW) == 0.8
        assert reporting_system._safety_risk_to_score(SafetyRiskLevel.MODERATE) == 0.6
        assert reporting_system._safety_risk_to_score(SafetyRiskLevel.HIGH) == 0.3
        assert reporting_system._safety_risk_to_score(SafetyRiskLevel.CRITICAL) == 0.0

    def test_performance_categorization(self, reporting_system):
        """Test performance categorization based on scores"""
        assert reporting_system._categorize_performance(0.95) == "Excellent"
        assert reporting_system._categorize_performance(0.85) == "Good"
        assert reporting_system._categorize_performance(0.75) == "Acceptable"
        assert reporting_system._categorize_performance(0.55) == "Concerning"
        assert reporting_system._categorize_performance(0.35) == "Critical"

    @pytest.mark.asyncio
    async def test_immediate_feedback_excellent(
        self, reporting_system, excellent_assessment_result
    ):
        """Test immediate feedback for excellent assessment"""
        initial_feedback_count = len(reporting_system.feedback_messages)

        # Add excellent assessment (should trigger immediate feedback)
        reporting_system.add_assessment_result(excellent_assessment_result)

        # Wait a moment for async feedback generation
        await asyncio.sleep(0.1)

        # Should have generated positive feedback
        assert len(reporting_system.feedback_messages) > initial_feedback_count

        # Find the excellence feedback
        excellence_feedback = next(
            (
                msg
                for msg in reporting_system.feedback_messages
                if msg.feedback_type == FeedbackType.MILESTONE_ACHIEVEMENT
            ),
            None,
        )
        assert excellence_feedback is not None
        assert "Excellent" in excellence_feedback.title

    @pytest.mark.asyncio
    async def test_immediate_feedback_dangerous(
        self, reporting_system, dangerous_assessment_result
    ):
        """Test immediate feedback for dangerous assessment"""
        initial_feedback_count = len(reporting_system.feedback_messages)

        # Add dangerous assessment (should trigger immediate feedback)
        reporting_system.add_assessment_result(dangerous_assessment_result)

        # Wait a moment for async feedback generation
        await asyncio.sleep(0.1)

        # Should have generated alert feedback
        assert len(reporting_system.feedback_messages) > initial_feedback_count

        # Find the danger feedback
        danger_feedback = next(
            (
                msg
                for msg in reporting_system.feedback_messages
                if msg.feedback_type == FeedbackType.PERFORMANCE_ALERT and "DANGEROUS" in msg.title
            ),
            None,
        )
        assert danger_feedback is not None
        assert danger_feedback.priority == "high"
        assert len(danger_feedback.actionable_items) > 0

    def test_get_performance_snapshot_empty(self, reporting_system):
        """Test getting performance snapshot with no data"""
        snapshot = reporting_system.get_performance_snapshot()

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.total_assessments == 0
        assert snapshot.overall_accuracy == 0.0

    def test_get_performance_snapshot_with_data(
        self, reporting_system, sample_assessment_result, excellent_assessment_result
    ):
        """Test getting performance snapshot with data"""
        # Add assessments
        reporting_system.add_assessment_result(sample_assessment_result)
        reporting_system.add_assessment_result(excellent_assessment_result)

        snapshot = reporting_system.get_performance_snapshot()

        assert isinstance(snapshot, PerformanceSnapshot)
        assert snapshot.total_assessments == 2
        assert snapshot.overall_accuracy > 0.0
        assert 0.0 <= snapshot.dsm5_compliance <= 1.0
        assert 0.0 <= snapshot.therapeutic_appropriateness <= 1.0
        assert 0.0 <= snapshot.safety_compliance <= 1.0

    def test_get_metric_data_points(self, reporting_system, sample_assessment_result):
        """Test getting metric data points for trend analysis"""
        # Add assessment
        reporting_system.add_assessment_result(sample_assessment_result)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        data_points = reporting_system._get_metric_data_points(
            PerformanceMetric.OVERALL_ACCURACY, start_date, end_date
        )

        assert isinstance(data_points, list)
        if data_points:  # If assessment falls within date range
            assert len(data_points) > 0
            assert all(isinstance(point, tuple) and len(point) == 2 for point in data_points)
            assert all(
                isinstance(point[0], datetime) and isinstance(point[1], float)
                for point in data_points
            )

    @pytest.mark.asyncio
    async def test_trend_analysis_insufficient_data(self, reporting_system):
        """Test trend analysis with insufficient data points"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        trend = await reporting_system._analyze_metric_trend(
            PerformanceMetric.OVERALL_ACCURACY, start_date, end_date
        )

        # Should return None for insufficient data
        assert trend is None

    def test_generate_trend_insights(self, reporting_system):
        """Test generation of trend insights"""
        # Test improving trend
        insights = reporting_system._generate_trend_insights(
            PerformanceMetric.OVERALL_ACCURACY, "improving", 0.8, 0.1
        )
        assert len(insights) > 0
        assert any("positive trend" in insight.lower() for insight in insights)

        # Test declining trend
        insights = reporting_system._generate_trend_insights(
            PerformanceMetric.SAFETY_COMPLIANCE, "declining", 0.9, -0.1
        )
        assert len(insights) > 0
        assert any("decline" in insight.lower() for insight in insights)

    def test_generate_individual_recommendations_safety(
        self, reporting_system, dangerous_assessment_result
    ):
        """Test generating recommendations for safety concerns"""
        recommendations = reporting_system._generate_individual_recommendations(
            dangerous_assessment_result, None, None, None
        )

        assert len(recommendations) > 0

        # Should have safety recommendation
        safety_rec = next((rec for rec in recommendations if rec.category == "safety"), None)
        assert safety_rec is not None
        assert safety_rec.priority == "critical"
        assert len(safety_rec.specific_actions) > 0

    def test_generate_aggregate_recommendations_performance(self, reporting_system):
        """Test generating aggregate recommendations for poor performance"""
        metrics = {
            "overall_accuracy": 0.5,  # Below threshold
            "accuracy_std": 0.4,  # High variability
            "total_assessments": 10,
        }

        recommendations = reporting_system._generate_aggregate_recommendations(metrics, [])

        assert len(recommendations) > 0

        # Should have performance recommendation
        perf_rec = next((rec for rec in recommendations if rec.category == "performance"), None)
        assert perf_rec is not None
        assert perf_rec.priority == "high"

        # Should have consistency recommendation
        cons_rec = next((rec for rec in recommendations if rec.category == "consistency"), None)
        assert cons_rec is not None

    def test_export_report_json(self, reporting_system, sample_assessment_result):
        """Test exporting report to JSON format"""
        # Add assessment and generate report
        reporting_system.add_assessment_result(sample_assessment_result)

        # Create a mock report
        report = ClinicalReport(
            report_id="test_report_001",
            report_type=ReportType.INDIVIDUAL_ASSESSMENT,
            generated_at=datetime.now(),
            time_period=(datetime.now(), datetime.now()),
            summary="Test summary",
            key_findings=["Test finding"],
            performance_metrics={"test_metric": 0.8},
            improvement_recommendations=[],
        )

        reporting_system.generated_reports[report.report_id] = report

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            reporting_system.export_report(report.report_id, output_path, "json")

            # Verify file was created and contains expected data
            assert output_path.exists()

            with open(output_path, "r") as f:
                exported_data = json.load(f)

            assert exported_data["report_id"] == report.report_id
            assert exported_data["summary"] == report.summary
            assert exported_data["key_findings"] == report.key_findings

        finally:
            if output_path.exists():
                output_path.unlink()

    def test_export_report_not_found(self, reporting_system):
        """Test exporting non-existent report"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Report .* not found"):
                reporting_system.export_report("nonexistent", output_path)
        finally:
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.asyncio
    async def test_feedback_message_delivery(self, reporting_system):
        """Test feedback message delivery"""
        message = FeedbackMessage(
            message_id="test_msg_001",
            feedback_type=FeedbackType.IMMEDIATE_FEEDBACK,
            recipient="test_user",
            title="Test Message",
            content="Test content",
            priority="medium",
        )

        # Deliver message
        await reporting_system._deliver_feedback_message(message)

        # Should be marked as delivered
        assert message.delivered is True

    def test_cache_functionality(self, reporting_system):
        """Test caching functionality"""

        # Define a simple compute function
        def compute_test_value():
            return "computed_value"

        # First call should compute and cache
        result1 = reporting_system._get_cached_or_compute("test_key", compute_test_value)
        assert result1 == "computed_value"
        assert "test_key" in reporting_system._analytics_cache

        # Second call should return cached value
        result2 = reporting_system._get_cached_or_compute("test_key", compute_test_value)
        assert result2 == "computed_value"

    @pytest.mark.asyncio
    async def test_multiple_assessments_aggregate_report(self, reporting_system):
        """Test aggregate report with multiple assessments of different qualities"""
        # Create assessments with different timestamps and qualities
        base_time = datetime.now() - timedelta(days=5)

        assessments = []
        for i in range(5):
            context = ClinicalContext(
                client_presentation=f"Client case {i}",
                therapeutic_modality=TherapeuticModality.CBT,
                session_phase="working",
            )

            # Vary the quality
            if i < 2:
                accuracy = ClinicalAccuracyLevel.EXCELLENT
                confidence = 0.9
            elif i < 4:
                accuracy = ClinicalAccuracyLevel.GOOD
                confidence = 0.8
            else:
                accuracy = ClinicalAccuracyLevel.CONCERNING
                confidence = 0.5

            assessment = ClinicalAccuracyResult(
                assessment_id=f"multi_test_{i}",
                timestamp=base_time + timedelta(days=i),
                clinical_context=context,
                dsm5_assessment=DSM5Assessment(diagnostic_confidence=confidence),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(
                    overall_score=confidence
                ),
                safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.LOW),
                overall_accuracy=accuracy,
                confidence_score=confidence,
                expert_validation_needed=False,
                recommendations=[],
                warnings=[],
            )

            assessments.append(assessment)
            reporting_system.add_assessment_result(assessment)

        # Generate aggregate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        report = await reporting_system.generate_aggregate_performance_report(start_date, end_date)

        # Verify report contains expected data
        assert report.performance_metrics["total_assessments"] == 5
        assert report.performance_metrics["excellent_count"] == 2
        assert 0.0 < report.performance_metrics["overall_accuracy"] < 1.0
        assert len(report.key_findings) > 0


# Integration tests
class TestClinicalReportingSystemIntegration:
    """Integration tests for clinical reporting system"""

    @pytest.mark.asyncio
    async def test_complete_reporting_workflow(self):
        """Test complete reporting workflow from assessment to feedback"""
        reporting_system = ClinicalReportingSystem()

        # Create and add multiple assessments
        context = ClinicalContext(
            client_presentation="Integration test case",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="working",
        )

        # Add excellent assessment
        excellent_assessment = ClinicalAccuracyResult(
            assessment_id="integration_excellent",
            timestamp=datetime.now() - timedelta(hours=2),
            clinical_context=context,
            dsm5_assessment=DSM5Assessment(
                primary_diagnosis="Anxiety Disorder", diagnostic_confidence=0.95
            ),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.95),
            safety_assessment=SafetyAssessment(overall_risk=SafetyRiskLevel.MINIMAL),
            overall_accuracy=ClinicalAccuracyLevel.EXCELLENT,
            confidence_score=0.95,
            expert_validation_needed=False,
            recommendations=[],
            warnings=[],
        )

        # Add concerning assessment
        concerning_assessment = ClinicalAccuracyResult(
            assessment_id="integration_concerning",
            timestamp=datetime.now() - timedelta(hours=1),
            clinical_context=context,
            dsm5_assessment=DSM5Assessment(diagnostic_confidence=0.4),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.4),
            safety_assessment=SafetyAssessment(
                overall_risk=SafetyRiskLevel.HIGH, safety_plan_needed=True
            ),
            overall_accuracy=ClinicalAccuracyLevel.CONCERNING,
            confidence_score=0.4,
            expert_validation_needed=True,
            recommendations=["Seek immediate supervision"],
            warnings=["High safety risk"],
        )

        # Add assessments to system
        reporting_system.add_assessment_result(excellent_assessment)
        reporting_system.add_assessment_result(concerning_assessment)

        # Wait for immediate feedback generation
        await asyncio.sleep(0.2)

        # Generate individual reports
        excellent_report = await reporting_system.generate_individual_assessment_report(
            "integration_excellent"
        )
        concerning_report = await reporting_system.generate_individual_assessment_report(
            "integration_concerning"
        )

        # Verify individual reports
        assert excellent_report is not None
        assert concerning_report is not None
        assert len(concerning_report.improvement_recommendations) > 0

        # Generate aggregate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        aggregate_report = await reporting_system.generate_aggregate_performance_report(
            start_date, end_date
        )

        # Verify aggregate report
        assert aggregate_report.performance_metrics["total_assessments"] == 2
        assert aggregate_report.performance_metrics["excellent_count"] == 1
        assert len(aggregate_report.improvement_recommendations) > 0

        # Verify feedback messages were generated
        assert len(reporting_system.feedback_messages) > 0

        # Check for excellence feedback
        excellence_feedback = next(
            (
                msg
                for msg in reporting_system.feedback_messages
                if msg.feedback_type == FeedbackType.MILESTONE_ACHIEVEMENT
            ),
            None,
        )
        assert excellence_feedback is not None

        # Check for safety alert feedback
        safety_feedback = next(
            (
                msg
                for msg in reporting_system.feedback_messages
                if msg.feedback_type == FeedbackType.PERFORMANCE_ALERT and "SAFETY" in msg.title
            ),
            None,
        )
        assert safety_feedback is not None

        # Get performance snapshot
        snapshot = reporting_system.get_performance_snapshot()
        assert snapshot.total_assessments == 2
        assert 0.0 < snapshot.overall_accuracy < 1.0

        # Export a report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            reporting_system.export_report(aggregate_report.report_id, output_path, "json")
            assert output_path.exists()

            # Verify exported content
            with open(output_path, "r") as f:
                exported_data = json.load(f)
            assert exported_data["report_id"] == aggregate_report.report_id

        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
