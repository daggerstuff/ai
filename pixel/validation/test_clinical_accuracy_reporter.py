"""
Unit tests for Clinical Accuracy Reporter

Tests the comprehensive reporting and feedback loop system for clinical
accuracy assessments, including trend analysis, expert feedback integration,
and improvement recommendations.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from .clinical_accuracy_reporter import (
    ClinicalAccuracyReporter,
    ExpertFeedback,
    FeedbackPriority,
    ImprovementRecommendation,
    ReportType,
)
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


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_clinical_context():
    """Create sample clinical context"""
    return ClinicalContext(
        client_presentation="Client presents with symptoms of depression and anxiety",
        therapeutic_modality=TherapeuticModality.CBT,
        session_phase="working",
        crisis_indicators=["suicidal ideation"],
        cultural_factors=["Hispanic/Latino background"],
        contraindications=[],
    )


@pytest.fixture
def sample_assessment_result(sample_clinical_context):
    """Create sample assessment result"""
    return ClinicalAccuracyResult(
        assessment_id="test_assessment_001",
        timestamp=datetime.now(),
        clinical_context=sample_clinical_context,
        dsm5_assessment=DSM5Assessment(
            primary_diagnosis="Major Depressive Disorder",
            diagnostic_confidence=0.85,
            criteria_met=["Depressed mood", "Anhedonia", "Sleep disturbance"],
            severity_specifiers=["Moderate"],
        ),
        pdm2_assessment=PDM2Assessment(
            personality_patterns=["Depressive"],
            mental_functioning={"cognitive": 0.7, "emotional": 0.6},
            symptom_patterns=["Mood symptoms", "Anxiety symptoms"],
        ),
        therapeutic_appropriateness=TherapeuticAppropriatenessScore(
            intervention_appropriateness=0.8,
            timing_appropriateness=0.9,
            cultural_sensitivity=0.85,
            ethical_compliance=0.95,
            boundary_maintenance=0.9,
            overall_score=0.86,
            rationale="Appropriate CBT intervention for depression",
        ),
        safety_assessment=SafetyAssessment(
            suicide_risk=SafetyRiskLevel.MODERATE,
            overall_risk=SafetyRiskLevel.MODERATE,
            immediate_interventions=["Safety planning", "Crisis contact"],
            safety_plan_needed=True,
        ),
        overall_accuracy=ClinicalAccuracyLevel.GOOD,
        confidence_score=0.82,
        expert_validation_needed=True,
        recommendations=["Continue CBT", "Monitor safety"],
        warnings=["Suicide risk present"],
    )


@pytest.fixture
def sample_expert_feedback():
    """Create sample expert feedback"""
    return ExpertFeedback(
        feedback_id="feedback_001",
        assessment_id="test_assessment_001",
        expert_id="expert_001",
        expert_credentials="PhD, Licensed Clinical Psychologist",
        feedback_type="validation",
        accuracy_rating=0.85,
        detailed_comments="Good assessment overall, minor concerns about cultural factors",
        specific_corrections={"cultural_sensitivity": "Consider additional cultural factors"},
        recommendations=["Include family dynamics assessment"],
        priority=FeedbackPriority.MEDIUM,
    )


@pytest_asyncio.fixture
async def reporter(temp_db):
    """Create clinical accuracy reporter instance"""
    reporter = ClinicalAccuracyReporter(db_path=temp_db)
    return reporter


class TestClinicalAccuracyReporter:
    """Test cases for ClinicalAccuracyReporter"""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Test reporter initialization"""
        reporter = ClinicalAccuracyReporter(db_path=temp_db)

        assert reporter.db_path == temp_db
        assert reporter.config is not None
        assert reporter.validator is not None

        # Check database tables were created
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                "assessments",
                "expert_feedback",
                "trend_analysis",
                "improvement_recommendations",
            ]
            for table in expected_tables:
                assert table in tables

    @pytest.mark.asyncio
    async def test_store_assessment(self, reporter, sample_assessment_result):
        """Test storing assessment results"""
        await reporter.store_assessment(sample_assessment_result)

        # Verify storage
        with sqlite3.connect(reporter.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?",
                (sample_assessment_result.assessment_id,),
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == sample_assessment_result.assessment_id
            assert row[2] == sample_assessment_result.overall_accuracy.value
            assert row[3] == sample_assessment_result.confidence_score

    @pytest.mark.asyncio
    async def test_add_expert_feedback(self, reporter, sample_expert_feedback):
        """Test adding expert feedback"""
        await reporter.add_expert_feedback(sample_expert_feedback)

        # Verify storage
        with sqlite3.connect(reporter.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM expert_feedback WHERE feedback_id = ?",
                (sample_expert_feedback.feedback_id,),
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == sample_expert_feedback.feedback_id
            assert row[1] == sample_expert_feedback.assessment_id
            assert row[5] == sample_expert_feedback.accuracy_rating

    @pytest.mark.asyncio
    async def test_feedback_integration(
        self, reporter, sample_assessment_result, sample_expert_feedback
    ):
        """Test automatic feedback integration"""
        # Store assessment first
        await reporter.store_assessment(sample_assessment_result)

        # Configure for automatic integration
        reporter.config["feedback_integration_mode"] = "automatic"
        reporter.config["expert_validation_threshold"] = 0.9  # Higher than feedback rating

        # Add feedback (should trigger integration)
        await reporter.add_expert_feedback(sample_expert_feedback)

        # Check if improvement recommendation was created
        with sqlite3.connect(reporter.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM improvement_recommendations")
            count = cursor.fetchone()[0]
            assert count > 0

    @pytest.mark.asyncio
    async def test_trend_analysis_generation(self, reporter):
        """Test trend analysis generation"""
        # Create multiple sample assessments
        assessments = []
        for i in range(15):  # Above minimum threshold
            assessment = ClinicalAccuracyResult(
                assessment_id=f"test_assessment_{i:03d}",
                timestamp=datetime.now() - timedelta(days=i),
                clinical_context=ClinicalContext(
                    client_presentation=f"Test case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(overall_score=0.8),
                safety_assessment=SafetyAssessment(),
                overall_accuracy=ClinicalAccuracyLevel.GOOD,
                confidence_score=0.8 + (i % 3) * 0.05,  # Vary confidence
                expert_validation_needed=i % 3 == 0,
            )
            assessments.append(assessment)
            await reporter.store_assessment(assessment)

        # Generate trend analysis
        analysis = await reporter.generate_trend_analysis(days=30)

        assert analysis is not None
        assert analysis.total_assessments == 15
        assert analysis.average_confidence > 0
        assert len(analysis.accuracy_distribution) > 0
        assert len(analysis.common_issues) >= 0
        assert len(analysis.improvement_areas) >= 0

    @pytest.mark.asyncio
    async def test_insufficient_data_trend_analysis(self, reporter):
        """Test trend analysis with insufficient data"""
        # Create only a few assessments (below threshold)
        for i in range(3):
            assessment = ClinicalAccuracyResult(
                assessment_id=f"test_assessment_{i:03d}",
                timestamp=datetime.now() - timedelta(days=i),
                clinical_context=ClinicalContext(
                    client_presentation=f"Test case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                safety_assessment=SafetyAssessment(),
                overall_accuracy=ClinicalAccuracyLevel.GOOD,
                confidence_score=0.8,
                expert_validation_needed=False,
            )
            await reporter.store_assessment(assessment)

        # Should return None for insufficient data
        analysis = await reporter.generate_trend_analysis(days=30)
        assert analysis is None

    @pytest.mark.asyncio
    async def test_improvement_recommendation_addition(self, reporter):
        """Test adding improvement recommendations"""
        recommendation = ImprovementRecommendation(
            recommendation_id="rec_001",
            category="training",
            priority=FeedbackPriority.HIGH,
            description="Improve diagnostic accuracy training",
            rationale="Low confidence scores observed",
            implementation_steps=["Develop training module", "Schedule sessions"],
            expected_impact="Increased diagnostic confidence",
            resources_needed=["Training materials", "Expert time"],
            timeline="4 weeks",
            success_metrics=["Confidence score improvement", "Expert validation rate"],
        )

        await reporter.add_improvement_recommendation(recommendation)

        # Verify storage
        with sqlite3.connect(reporter.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM improvement_recommendations WHERE recommendation_id = ?",
                (recommendation.recommendation_id,),
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == recommendation.recommendation_id
            assert row[2] == recommendation.priority.value

    @pytest.mark.asyncio
    async def test_individual_assessment_report(self, reporter, sample_assessment_result):
        """Test individual assessment report generation"""
        await reporter.store_assessment(sample_assessment_result)

        report = await reporter.generate_comprehensive_report(
            ReportType.INDIVIDUAL_ASSESSMENT, assessment_id=sample_assessment_result.assessment_id
        )

        assert report["report_type"] == ReportType.INDIVIDUAL_ASSESSMENT.value
        assert "assessment" in report
        assert "summary" in report
        assert (
            report["summary"]["overall_accuracy"] == sample_assessment_result.overall_accuracy.value
        )

    @pytest.mark.asyncio
    async def test_trend_analysis_report(self, reporter):
        """Test trend analysis report generation"""
        # Create sample data
        for i in range(12):
            assessment = ClinicalAccuracyResult(
                assessment_id=f"trend_test_{i:03d}",
                timestamp=datetime.now() - timedelta(days=i),
                clinical_context=ClinicalContext(
                    client_presentation=f"Test case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                safety_assessment=SafetyAssessment(),
                overall_accuracy=ClinicalAccuracyLevel.GOOD,
                confidence_score=0.8,
                expert_validation_needed=False,
            )
            await reporter.store_assessment(assessment)

        report = await reporter.generate_comprehensive_report(ReportType.TREND_ANALYSIS, days=30)

        assert report["report_type"] == ReportType.TREND_ANALYSIS.value
        assert "summary" in report
        assert "accuracy_distribution" in report
        assert "improvement_areas" in report

    @pytest.mark.asyncio
    async def test_expert_feedback_report(
        self, reporter, sample_assessment_result, sample_expert_feedback
    ):
        """Test expert feedback report generation"""
        await reporter.store_assessment(sample_assessment_result)
        await reporter.add_expert_feedback(sample_expert_feedback)

        report = await reporter.generate_comprehensive_report(ReportType.EXPERT_FEEDBACK, days=30)

        assert report["report_type"] == ReportType.EXPERT_FEEDBACK.value
        assert "summary" in report
        assert "expert_statistics" in report
        assert report["summary"]["total_feedback"] > 0

    @pytest.mark.asyncio
    async def test_safety_alerts_report(self, reporter):
        """Test safety alerts report generation"""
        # Create assessment with safety concerns
        assessment = ClinicalAccuracyResult(
            assessment_id="safety_test_001",
            timestamp=datetime.now(),
            clinical_context=ClinicalContext(
                client_presentation="High risk client",
                therapeutic_modality=TherapeuticModality.CBT,
                session_phase="initial",
            ),
            dsm5_assessment=DSM5Assessment(),
            pdm2_assessment=PDM2Assessment(),
            therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
            safety_assessment=SafetyAssessment(
                suicide_risk=SafetyRiskLevel.HIGH, overall_risk=SafetyRiskLevel.HIGH
            ),
            overall_accuracy=ClinicalAccuracyLevel.GOOD,
            confidence_score=0.8,
            expert_validation_needed=True,
        )
        await reporter.store_assessment(assessment)

        report = await reporter.generate_comprehensive_report(ReportType.SAFETY_ALERTS, days=7)

        assert report["report_type"] == ReportType.SAFETY_ALERTS.value
        assert "summary" in report
        assert report["summary"]["total_safety_alerts"] > 0
        assert report["summary"]["high_risk_alerts"] > 0

    @pytest.mark.asyncio
    async def test_comparative_time_periods_report(self, reporter):
        """Test comparative analysis across time periods"""
        # Create assessments for two different periods
        now = datetime.now()

        # Recent period (better performance)
        for i in range(5):
            assessment = ClinicalAccuracyResult(
                assessment_id=f"recent_{i:03d}",
                timestamp=now - timedelta(days=i),
                clinical_context=ClinicalContext(
                    client_presentation=f"Recent case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                safety_assessment=SafetyAssessment(),
                overall_accuracy=ClinicalAccuracyLevel.GOOD,
                confidence_score=0.85,  # Higher confidence
                expert_validation_needed=False,
            )
            await reporter.store_assessment(assessment)

        # Previous period (lower performance)
        for i in range(5):
            assessment = ClinicalAccuracyResult(
                assessment_id=f"previous_{i:03d}",
                timestamp=now - timedelta(days=35 + i),  # 35+ days ago
                clinical_context=ClinicalContext(
                    client_presentation=f"Previous case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                safety_assessment=SafetyAssessment(),
                overall_accuracy=ClinicalAccuracyLevel.ACCEPTABLE,
                confidence_score=0.75,  # Lower confidence
                expert_validation_needed=True,
            )
            await reporter.store_assessment(assessment)

        report = await reporter.generate_comprehensive_report(
            ReportType.COMPARATIVE_ANALYSIS,
            comparison_type="time_periods",
            period1_days=30,
            period2_days=30,
        )

        assert report["report_type"] == ReportType.COMPARATIVE_ANALYSIS.value
        assert "period1" in report
        assert "period2" in report
        assert "comparison" in report
        assert report["comparison"]["improvement_trend"] == "improving"

    @pytest.mark.asyncio
    async def test_therapeutic_modalities_comparison(self, reporter):
        """Test comparison across therapeutic modalities"""
        modalities = [
            TherapeuticModality.CBT,
            TherapeuticModality.DBT,
            TherapeuticModality.PSYCHODYNAMIC,
        ]

        for i, modality in enumerate(modalities):
            for j in range(3):
                assessment = ClinicalAccuracyResult(
                    assessment_id=f"modality_{modality.value}_{j:03d}",
                    timestamp=datetime.now() - timedelta(days=j),
                    clinical_context=ClinicalContext(
                        client_presentation=f"Case for {modality.value}",
                        therapeutic_modality=modality,
                        session_phase="working",
                    ),
                    dsm5_assessment=DSM5Assessment(),
                    pdm2_assessment=PDM2Assessment(),
                    therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                    safety_assessment=SafetyAssessment(),
                    overall_accuracy=ClinicalAccuracyLevel.GOOD,
                    confidence_score=0.8 + i * 0.05,  # Vary by modality
                    expert_validation_needed=False,
                )
                await reporter.store_assessment(assessment)

        report = await reporter.generate_comprehensive_report(
            ReportType.COMPARATIVE_ANALYSIS, comparison_type="therapeutic_modalities", days=30
        )

        assert report["report_type"] == ReportType.COMPARATIVE_ANALYSIS.value
        assert "modality_comparison" in report
        assert "best_performing" in report
        assert len(report["modality_comparison"]) == 3

    @pytest.mark.asyncio
    async def test_report_export_json(self, reporter, temp_db):
        """Test JSON report export"""
        report_data = {
            "report_id": "test_report_001",
            "report_type": "test",
            "timestamp": datetime.now().isoformat(),
            "data": {"test": "value"},
        }

        output_path = await reporter.export_report(report_data, format="json")

        assert output_path.exists()
        assert output_path.suffix == ".json"

        # Verify content
        with open(output_path, "r") as f:
            exported_data = json.load(f)

        assert exported_data["report_id"] == report_data["report_id"]
        assert exported_data["data"]["test"] == "value"

        # Cleanup
        output_path.unlink()
        output_path.parent.rmdir()

    @pytest.mark.asyncio
    async def test_report_export_html(self, reporter):
        """Test HTML report export"""
        report_data = {
            "report_id": "test_report_002",
            "report_type": "test",
            "timestamp": datetime.now().isoformat(),
            "summary": {"metric1": "value1", "metric2": "value2"},
        }

        output_path = await reporter.export_report(report_data, format="html")

        assert output_path.exists()
        assert output_path.suffix == ".html"

        # Verify HTML content
        with open(output_path, "r") as f:
            html_content = f.read()

        assert "Clinical Accuracy Report" in html_content
        assert report_data["report_id"] in html_content
        assert "metric1" in html_content

        # Cleanup
        output_path.unlink()
        output_path.parent.rmdir()

    @pytest.mark.asyncio
    async def test_feedback_loop_status(
        self, reporter, sample_assessment_result, sample_expert_feedback
    ):
        """Test feedback loop status reporting"""
        # Add some data
        await reporter.store_assessment(sample_assessment_result)
        await reporter.add_expert_feedback(sample_expert_feedback)

        status = await reporter.get_feedback_loop_status()

        assert status["system_status"] == "active"
        assert "recent_activity" in status
        assert "configuration" in status
        assert "last_updated" in status
        assert status["recent_activity"]["assessments_last_7_days"] > 0
        assert status["recent_activity"]["expert_feedback_last_7_days"] > 0

    @pytest.mark.asyncio
    async def test_common_issues_identification(self, reporter):
        """Test identification of common issues"""
        # Create assessments with various issues
        issues_data = [
            (ClinicalAccuracyLevel.GOOD, 0.6, False, SafetyRiskLevel.MINIMAL),  # Low confidence
            (
                ClinicalAccuracyLevel.CONCERNING,
                0.8,
                True,
                SafetyRiskLevel.HIGH,
            ),  # Poor accuracy + safety
            (
                ClinicalAccuracyLevel.GOOD,
                0.9,
                True,
                SafetyRiskLevel.MINIMAL,
            ),  # Expert validation needed
            (
                ClinicalAccuracyLevel.DANGEROUS,
                0.7,
                False,
                SafetyRiskLevel.CRITICAL,
            ),  # Dangerous + critical safety
        ]

        for i, (accuracy, confidence, expert_needed, safety_risk) in enumerate(issues_data):
            assessment = ClinicalAccuracyResult(
                assessment_id=f"issues_test_{i:03d}",
                timestamp=datetime.now() - timedelta(days=i),
                clinical_context=ClinicalContext(
                    client_presentation=f"Issues test case {i}",
                    therapeutic_modality=TherapeuticModality.CBT,
                    session_phase="working",
                ),
                dsm5_assessment=DSM5Assessment(),
                pdm2_assessment=PDM2Assessment(),
                therapeutic_appropriateness=TherapeuticAppropriatenessScore(),
                safety_assessment=SafetyAssessment(overall_risk=safety_risk),
                overall_accuracy=accuracy,
                confidence_score=confidence,
                expert_validation_needed=expert_needed,
            )
            await reporter.store_assessment(assessment)

        # Generate trend analysis to identify issues
        analysis = await reporter.generate_trend_analysis(days=30)

        assert analysis is not None
        assert len(analysis.common_issues) > 0

        # Check that various issue types are identified
        issue_types = [issue[0] for issue in analysis.common_issues]
        expected_issues = ["Low confidence", "Expert validation", "Safety risk", "Poor accuracy"]

        # At least some of these should be identified
        identified_count = sum(
            1
            for expected in expected_issues
            if any(expected in issue_type for issue_type in issue_types)
        )
        assert identified_count > 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_assessment_id(self, reporter):
        """Test error handling for invalid assessment ID"""
        with pytest.raises(ValueError, match="Assessment .* not found"):
            await reporter.generate_comprehensive_report(
                ReportType.INDIVIDUAL_ASSESSMENT, assessment_id="nonexistent_id"
            )

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_export_format(self, reporter):
        """Test error handling for unsupported export format"""
        report_data = {"report_id": "test", "data": {}}

        with pytest.raises(ValueError, match="Unsupported export format"):
            await reporter.export_report(report_data, format="unsupported")

    @pytest.mark.asyncio
    async def test_error_handling_unsupported_comparison_type(self, reporter):
        """Test error handling for unsupported comparison type"""
        report = await reporter.generate_comprehensive_report(
            ReportType.COMPARATIVE_ANALYSIS, comparison_type="unsupported_type"
        )

        assert "error" in report
        assert "Unsupported comparison type" in report["error"]


if __name__ == "__main__":
    pytest.main([__file__])
