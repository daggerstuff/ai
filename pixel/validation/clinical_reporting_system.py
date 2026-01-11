"""
Clinical Accuracy Reporting and Feedback Loop System

This module provides comprehensive reporting and feedback mechanisms for
clinical accuracy validation, including performance analytics, trend analysis,
improvement recommendations, and automated feedback loops.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .automated_clinical_checker import AppropriatenessCheckResult, AppropriatenessLevel
from .clinical_accuracy_validator import ClinicalAccuracyLevel, ClinicalAccuracyResult
from .expert_validation_interface import ConsensusResult
from .safety_ethics_validator import SafetyEthicsComplianceResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of clinical accuracy reports"""

    INDIVIDUAL_ASSESSMENT = "individual_assessment"
    AGGREGATE_PERFORMANCE = "aggregate_performance"
    TREND_ANALYSIS = "trend_analysis"
    EXPERT_CONSENSUS = "expert_consensus"
    SAFETY_COMPLIANCE = "safety_compliance"
    IMPROVEMENT_RECOMMENDATIONS = "improvement_recommendations"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class FeedbackType(Enum):
    """Types of feedback mechanisms"""

    IMMEDIATE_FEEDBACK = "immediate_feedback"
    PERIODIC_SUMMARY = "periodic_summary"
    MILESTONE_ACHIEVEMENT = "milestone_achievement"
    PERFORMANCE_ALERT = "performance_alert"
    IMPROVEMENT_SUGGESTION = "improvement_suggestion"
    EXPERT_RECOMMENDATION = "expert_recommendation"


class PerformanceMetric(Enum):
    """Performance metrics for clinical accuracy"""

    OVERALL_ACCURACY = "overall_accuracy"
    DSM5_COMPLIANCE = "dsm5_compliance"
    THERAPEUTIC_APPROPRIATENESS = "therapeutic_appropriateness"
    SAFETY_COMPLIANCE = "safety_compliance"
    ETHICS_COMPLIANCE = "ethics_compliance"
    EXPERT_AGREEMENT = "expert_agreement"
    IMPROVEMENT_RATE = "improvement_rate"
    CONSISTENCY_SCORE = "consistency_score"


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""

    timestamp: datetime
    overall_accuracy: float
    dsm5_compliance: float
    therapeutic_appropriateness: float
    safety_compliance: float
    ethics_compliance: float
    expert_agreement: float
    total_assessments: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis results"""

    metric: PerformanceMetric
    time_period: str
    trend_direction: str  # "improving", "declining", "stable"
    trend_strength: float  # 0.0 to 1.0
    rate_of_change: float
    statistical_significance: float
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)


@dataclass
class ImprovementRecommendation:
    """Improvement recommendation based on analysis"""

    recommendation_id: str
    category: str
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    specific_actions: List[str] = field(default_factory=list)
    expected_impact: str = ""
    timeline: str = ""
    success_metrics: List[str] = field(default_factory=list)
    resources_needed: List[str] = field(default_factory=list)


@dataclass
class FeedbackMessage:
    """Feedback message for users"""

    message_id: str
    feedback_type: FeedbackType
    recipient: str
    title: str
    content: str
    priority: str  # "high", "medium", "low"
    actionable_items: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    delivered: bool = False


@dataclass
class ClinicalReport:
    """Comprehensive clinical accuracy report"""

    report_id: str
    report_type: ReportType
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    summary: str
    key_findings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    trend_analyses: List[TrendAnalysis] = field(default_factory=list)
    improvement_recommendations: List[ImprovementRecommendation] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)  # Paths to generated charts
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClinicalReportingSystem:
    """
    Comprehensive clinical accuracy reporting and feedback system

    This class provides analytics, reporting, and feedback mechanisms
    for clinical accuracy validation results.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the clinical reporting system"""
        self.config = self._load_config(config_path)

        # Data storage
        self.assessment_results: List[ClinicalAccuracyResult] = []
        self.expert_validations: List[ConsensusResult] = []
        self.appropriateness_checks: List[AppropriatenessCheckResult] = []
        self.safety_compliance_results: List[SafetyEthicsComplianceResult] = []

        # Performance tracking
        self.performance_history: List[PerformanceSnapshot] = []
        self.generated_reports: Dict[str, ClinicalReport] = {}
        self.feedback_messages: List[FeedbackMessage] = []

        # Analytics cache
        self._analytics_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        logger.info("Clinical reporting system initialized")

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        default_config = {
            "cache_duration_minutes": 30,
            "trend_analysis_periods": ["7d", "30d", "90d"],
            "performance_thresholds": {
                "excellent": 0.9,
                "good": 0.8,
                "acceptable": 0.7,
                "concerning": 0.5,
            },
            "feedback_frequency": {
                "immediate": True,
                "daily_summary": True,
                "weekly_report": True,
                "monthly_analysis": True,
            },
            "visualization_settings": {"figure_size": (12, 8), "dpi": 300, "style": "seaborn-v0_8"},
        }

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return default_config

    def add_assessment_result(self, result: ClinicalAccuracyResult) -> None:
        """Add a clinical accuracy assessment result"""
        self.assessment_results.append(result)
        self._invalidate_cache()

        # Generate immediate feedback if enabled
        if self.config["feedback_frequency"]["immediate"]:
            try:
                # Try to create task if event loop is running
                asyncio.create_task(self._generate_immediate_feedback(result))
            except RuntimeError:
                # No event loop running, skip immediate feedback
                logger.debug("No event loop running, skipping immediate feedback")
                pass

    def add_expert_validation(self, validation: ConsensusResult) -> None:
        """Add an expert validation result"""
        self.expert_validations.append(validation)
        self._invalidate_cache()

    def add_appropriateness_check(self, check: AppropriatenessCheckResult) -> None:
        """Add an appropriateness check result"""
        self.appropriateness_checks.append(check)
        self._invalidate_cache()

    def add_safety_compliance_result(self, result: SafetyEthicsComplianceResult) -> None:
        """Add a safety and ethics compliance result"""
        self.safety_compliance_results.append(result)
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate analytics cache"""
        self._analytics_cache.clear()
        self._cache_expiry.clear()

    def _get_cached_or_compute(self, cache_key: str, compute_func) -> Any:
        """Get cached result or compute and cache"""
        now = datetime.now()

        # Check if cache is valid
        if (
            cache_key in self._analytics_cache
            and cache_key in self._cache_expiry
            and now < self._cache_expiry[cache_key]
        ):
            return self._analytics_cache[cache_key]

        # Compute and cache
        result = compute_func()
        self._analytics_cache[cache_key] = result
        self._cache_expiry[cache_key] = now + timedelta(
            minutes=self.config["cache_duration_minutes"]
        )

        return result

    async def generate_individual_assessment_report(
        self, assessment_id: str
    ) -> Optional[ClinicalReport]:
        """Generate detailed report for individual assessment"""
        # Find the assessment
        assessment = next(
            (a for a in self.assessment_results if a.assessment_id == assessment_id), None
        )

        if not assessment:
            logger.error(f"Assessment {assessment_id} not found")
            return None

        # Find related validations and checks
        expert_validation = next(
            (v for v in self.expert_validations if assessment_id in str(v)), None
        )

        appropriateness_check = next(
            (c for c in self.appropriateness_checks if assessment_id in str(c)), None
        )

        safety_result = next(
            (s for s in self.safety_compliance_results if assessment_id in str(s)), None
        )

        # Generate report
        report = ClinicalReport(
            report_id=f"individual_{assessment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.INDIVIDUAL_ASSESSMENT,
            generated_at=datetime.now(),
            time_period=(assessment.timestamp, assessment.timestamp),
            summary=self._generate_individual_summary(assessment),
            key_findings=self._extract_individual_findings(
                assessment, expert_validation, appropriateness_check, safety_result
            ),
            performance_metrics=self._calculate_individual_metrics(assessment),
            improvement_recommendations=self._generate_individual_recommendations(
                assessment, expert_validation, appropriateness_check, safety_result
            ),
        )

        self.generated_reports[report.report_id] = report
        return report

    def _generate_individual_summary(self, assessment: ClinicalAccuracyResult) -> str:
        """Generate summary for individual assessment"""
        return f"""
        Clinical Accuracy Assessment Summary for {assessment.assessment_id}:
        
        Overall Accuracy: {assessment.overall_accuracy.value.title()}
        Confidence Score: {assessment.confidence_score:.2f}
        Expert Validation Required: {'Yes' if assessment.expert_validation_needed else 'No'}
        
        Key Components:
        - DSM-5 Assessment: {assessment.dsm5_assessment.primary_diagnosis or 'No primary diagnosis'}
        - Therapeutic Appropriateness: {assessment.therapeutic_appropriateness.overall_score:.2f}
        - Safety Risk Level: {assessment.safety_assessment.overall_risk.value.title()}
        
        Recommendations: {len(assessment.recommendations)} provided
        Warnings: {len(assessment.warnings)} identified
        """

    def _extract_individual_findings(
        self,
        assessment: ClinicalAccuracyResult,
        expert_validation: Optional[ConsensusResult],
        appropriateness_check: Optional[AppropriatenessCheckResult],
        safety_result: Optional[SafetyEthicsComplianceResult],
    ) -> List[str]:
        """Extract key findings from individual assessment"""
        findings = []

        # Assessment findings
        if assessment.overall_accuracy == ClinicalAccuracyLevel.EXCELLENT:
            findings.append("Excellent clinical accuracy achieved")
        elif assessment.overall_accuracy == ClinicalAccuracyLevel.DANGEROUS:
            findings.append("CRITICAL: Dangerous clinical response detected")

        # Expert validation findings
        if expert_validation:
            if expert_validation.expert_agreement_level > 0.8:
                findings.append("High expert consensus achieved")
            elif expert_validation.expert_agreement_level < 0.5:
                findings.append("Low expert consensus - conflicting opinions")

        # Appropriateness findings
        if appropriateness_check:
            if appropriateness_check.overall_level == AppropriatenessLevel.INAPPROPRIATE:
                findings.append("Clinical appropriateness concerns identified")

        # Safety findings
        if safety_result:
            if safety_result.safety_result.crisis_protocol_triggered:
                findings.append("Crisis intervention protocol triggered")

        return findings

    def _calculate_individual_metrics(self, assessment: ClinicalAccuracyResult) -> Dict[str, float]:
        """Calculate metrics for individual assessment"""
        return {
            "overall_accuracy": self._accuracy_to_score(assessment.overall_accuracy),
            "confidence_score": assessment.confidence_score,
            "dsm5_confidence": assessment.dsm5_assessment.diagnostic_confidence,
            "therapeutic_score": assessment.therapeutic_appropriateness.overall_score,
            "safety_score": self._safety_risk_to_score(assessment.safety_assessment.overall_risk),
        }

    def _accuracy_to_score(self, accuracy: ClinicalAccuracyLevel) -> float:
        """Convert accuracy level to numeric score"""
        mapping = {
            ClinicalAccuracyLevel.EXCELLENT: 1.0,
            ClinicalAccuracyLevel.GOOD: 0.8,
            ClinicalAccuracyLevel.ACCEPTABLE: 0.6,
            ClinicalAccuracyLevel.CONCERNING: 0.4,
            ClinicalAccuracyLevel.DANGEROUS: 0.0,
        }
        return mapping.get(accuracy, 0.5)

    def _safety_risk_to_score(self, risk_level) -> float:
        """Convert safety risk level to numeric score (higher = safer)"""
        from .clinical_accuracy_validator import SafetyRiskLevel

        mapping = {
            SafetyRiskLevel.MINIMAL: 1.0,
            SafetyRiskLevel.LOW: 0.8,
            SafetyRiskLevel.MODERATE: 0.6,
            SafetyRiskLevel.HIGH: 0.3,
            SafetyRiskLevel.CRITICAL: 0.0,
        }
        return mapping.get(risk_level, 0.5)

    def _generate_individual_recommendations(
        self,
        assessment: ClinicalAccuracyResult,
        expert_validation: Optional[ConsensusResult],
        appropriateness_check: Optional[AppropriatenessCheckResult],
        safety_result: Optional[SafetyEthicsComplianceResult],
    ) -> List[ImprovementRecommendation]:
        """Generate improvement recommendations for individual assessment"""
        recommendations = []

        # Based on accuracy level
        if assessment.overall_accuracy in [
            ClinicalAccuracyLevel.CONCERNING,
            ClinicalAccuracyLevel.DANGEROUS,
        ]:
            recommendations.append(
                ImprovementRecommendation(
                    recommendation_id=f"acc_{assessment.assessment_id}_1",
                    category="clinical_accuracy",
                    priority="high",
                    title="Improve Clinical Accuracy",
                    description="Clinical accuracy is below acceptable standards",
                    specific_actions=[
                        "Review clinical assessment protocols",
                        "Seek additional supervision",
                        "Complete targeted training modules",
                    ],
                    expected_impact="Significant improvement in clinical decision-making",
                    timeline="2-4 weeks",
                )
            )

        # Based on safety concerns
        if assessment.safety_assessment.safety_plan_needed:
            recommendations.append(
                ImprovementRecommendation(
                    recommendation_id=f"safety_{assessment.assessment_id}_1",
                    category="safety",
                    priority="critical",
                    title="Implement Safety Protocols",
                    description="Safety concerns require immediate attention",
                    specific_actions=[
                        "Develop comprehensive safety plan",
                        "Implement crisis intervention protocols",
                        "Increase monitoring frequency",
                    ],
                    expected_impact="Enhanced client safety and risk management",
                    timeline="Immediate",
                )
            )

        return recommendations

    async def generate_aggregate_performance_report(
        self, start_date: datetime, end_date: datetime
    ) -> ClinicalReport:
        """Generate aggregate performance report for time period"""

        def compute_aggregate_metrics():
            # Filter data by date range
            filtered_assessments = [
                a for a in self.assessment_results if start_date <= a.timestamp <= end_date
            ]

            if not filtered_assessments:
                return {"total_assessments": 0, "overall_accuracy": 0.0, "average_confidence": 0.0}

            # Calculate aggregate metrics
            accuracy_scores = [
                self._accuracy_to_score(a.overall_accuracy) for a in filtered_assessments
            ]
            confidence_scores = [a.confidence_score for a in filtered_assessments]

            return {
                "total_assessments": len(filtered_assessments),
                "overall_accuracy": statistics.mean(accuracy_scores),
                "average_confidence": statistics.mean(confidence_scores),
                "accuracy_std": (
                    statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0.0
                ),
                "excellent_count": sum(
                    1
                    for a in filtered_assessments
                    if a.overall_accuracy == ClinicalAccuracyLevel.EXCELLENT
                ),
                "dangerous_count": sum(
                    1
                    for a in filtered_assessments
                    if a.overall_accuracy == ClinicalAccuracyLevel.DANGEROUS
                ),
            }

        cache_key = f"aggregate_{start_date.isoformat()}_{end_date.isoformat()}"
        metrics = self._get_cached_or_compute(cache_key, compute_aggregate_metrics)

        # Generate trend analyses
        trend_analyses = await self._generate_trend_analyses(start_date, end_date)

        # Generate improvement recommendations
        recommendations = self._generate_aggregate_recommendations(metrics, trend_analyses)

        # Create report
        report = ClinicalReport(
            report_id=f"aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=ReportType.AGGREGATE_PERFORMANCE,
            generated_at=datetime.now(),
            time_period=(start_date, end_date),
            summary=self._generate_aggregate_summary(metrics, start_date, end_date),
            key_findings=self._extract_aggregate_findings(metrics, trend_analyses),
            performance_metrics=metrics,
            trend_analyses=trend_analyses,
            improvement_recommendations=recommendations,
        )

        self.generated_reports[report.report_id] = report
        return report

    def _generate_aggregate_summary(
        self, metrics: Dict[str, Any], start_date: datetime, end_date: datetime
    ) -> str:
        """Generate summary for aggregate performance report"""
        period_days = (end_date - start_date).days

        return f"""
        Aggregate Clinical Accuracy Performance Report
        Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({period_days} days)
        
        Overall Performance:
        - Total Assessments: {metrics['total_assessments']}
        - Average Accuracy Score: {metrics['overall_accuracy']:.3f}
        - Average Confidence: {metrics['average_confidence']:.3f}
        - Accuracy Standard Deviation: {metrics['accuracy_std']:.3f}
        
        Quality Distribution:
        - Excellent Assessments: {metrics['excellent_count']} ({metrics['excellent_count']/max(1, metrics['total_assessments'])*100:.1f}%)
        - Dangerous Assessments: {metrics['dangerous_count']} ({metrics['dangerous_count']/max(1, metrics['total_assessments'])*100:.1f}%)
        
        Performance Level: {self._categorize_performance(metrics['overall_accuracy'])}
        """

    def _categorize_performance(self, score: float) -> str:
        """Categorize performance based on score"""
        thresholds = self.config["performance_thresholds"]
        if score >= thresholds["excellent"]:
            return "Excellent"
        elif score >= thresholds["good"]:
            return "Good"
        elif score >= thresholds["acceptable"]:
            return "Acceptable"
        elif score >= thresholds["concerning"]:
            return "Concerning"
        else:
            return "Critical"

    def _extract_aggregate_findings(
        self, metrics: Dict[str, Any], trend_analyses: List[TrendAnalysis]
    ) -> List[str]:
        """Extract key findings from aggregate analysis"""
        findings = []

        # Performance findings
        if metrics["overall_accuracy"] >= 0.9:
            findings.append("Exceptional clinical accuracy performance maintained")
        elif metrics["overall_accuracy"] < 0.5:
            findings.append("CRITICAL: Clinical accuracy below acceptable standards")

        # Consistency findings
        if metrics["accuracy_std"] > 0.3:
            findings.append("High variability in assessment quality detected")
        elif metrics["accuracy_std"] < 0.1:
            findings.append("Excellent consistency in assessment quality")

        # Trend findings
        for trend in trend_analyses:
            if trend.trend_direction == "improving" and trend.trend_strength > 0.7:
                findings.append(f"Strong improvement trend in {trend.metric.value}")
            elif trend.trend_direction == "declining" and trend.trend_strength > 0.7:
                findings.append(f"Concerning decline in {trend.metric.value}")

        # Safety findings
        if metrics["dangerous_count"] > 0:
            findings.append(
                f"Safety concern: {metrics['dangerous_count']} dangerous assessments identified"
            )

        return findings

    async def _generate_trend_analyses(
        self, start_date: datetime, end_date: datetime
    ) -> List[TrendAnalysis]:
        """Generate trend analyses for various metrics"""
        analyses = []

        # Analyze overall accuracy trend
        accuracy_trend = await self._analyze_metric_trend(
            PerformanceMetric.OVERALL_ACCURACY, start_date, end_date
        )
        if accuracy_trend:
            analyses.append(accuracy_trend)

        # Analyze safety compliance trend
        safety_trend = await self._analyze_metric_trend(
            PerformanceMetric.SAFETY_COMPLIANCE, start_date, end_date
        )
        if safety_trend:
            analyses.append(safety_trend)

        return analyses

    async def _analyze_metric_trend(
        self, metric: PerformanceMetric, start_date: datetime, end_date: datetime
    ) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric"""
        # Get data points
        data_points = self._get_metric_data_points(metric, start_date, end_date)

        if len(data_points) < 3:  # Need at least 3 points for trend analysis
            return None

        # Calculate trend
        x_values = [(point[0] - start_date).total_seconds() for point in data_points]
        y_values = [point[1] for point in data_points]

        # Simple linear regression for trend
        n = len(data_points)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        # Calculate slope (rate of change)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Determine trend direction and strength
        if abs(slope) < 1e-10:  # Essentially zero
            trend_direction = "stable"
            trend_strength = 0.0
        elif slope > 0:
            trend_direction = "improving"
            trend_strength = min(1.0, abs(slope) * 1000)  # Scale appropriately
        else:
            trend_direction = "declining"
            trend_strength = min(1.0, abs(slope) * 1000)

        # Generate insights
        insights = self._generate_trend_insights(metric, trend_direction, trend_strength, slope)

        return TrendAnalysis(
            metric=metric,
            time_period=f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            rate_of_change=slope,
            statistical_significance=0.8,  # Simplified - would use proper statistical tests
            data_points=data_points,
            insights=insights,
        )

    def _get_metric_data_points(
        self, metric: PerformanceMetric, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, float]]:
        """Get data points for a specific metric over time period"""
        data_points = []

        # Group assessments by day
        daily_groups = defaultdict(list)
        for assessment in self.assessment_results:
            if start_date <= assessment.timestamp <= end_date:
                day = assessment.timestamp.date()
                daily_groups[day].append(assessment)

        # Calculate daily metric values
        for day, assessments in daily_groups.items():
            if metric == PerformanceMetric.OVERALL_ACCURACY:
                scores = [self._accuracy_to_score(a.overall_accuracy) for a in assessments]
                daily_value = statistics.mean(scores)
            elif metric == PerformanceMetric.SAFETY_COMPLIANCE:
                scores = [
                    self._safety_risk_to_score(a.safety_assessment.overall_risk)
                    for a in assessments
                ]
                daily_value = statistics.mean(scores)
            else:
                daily_value = 0.5  # Default value

            data_points.append((datetime.combine(day, datetime.min.time()), daily_value))

        return sorted(data_points, key=lambda x: x[0])

    def _generate_trend_insights(
        self,
        metric: PerformanceMetric,
        trend_direction: str,
        trend_strength: float,
        rate_of_change: float,
    ) -> List[str]:
        """Generate insights based on trend analysis"""
        insights = []

        if trend_direction == "improving":
            if trend_strength > 0.7:
                insights.append(f"Strong positive trend in {metric.value} - excellent progress")
            elif trend_strength > 0.4:
                insights.append(f"Moderate improvement in {metric.value} - good progress")
            else:
                insights.append(f"Slight improvement in {metric.value} - maintain current efforts")

        elif trend_direction == "declining":
            if trend_strength > 0.7:
                insights.append(
                    f"Significant decline in {metric.value} - immediate intervention needed"
                )
            elif trend_strength > 0.4:
                insights.append(
                    f"Moderate decline in {metric.value} - corrective action recommended"
                )
            else:
                insights.append(f"Slight decline in {metric.value} - monitor closely")

        else:  # stable
            insights.append(f"{metric.value} remains stable - consistent performance")

        return insights

    def _generate_aggregate_recommendations(
        self, metrics: Dict[str, Any], trend_analyses: List[TrendAnalysis]
    ) -> List[ImprovementRecommendation]:
        """Generate improvement recommendations based on aggregate analysis"""
        recommendations = []

        # Performance-based recommendations
        if metrics["overall_accuracy"] < 0.7:
            recommendations.append(
                ImprovementRecommendation(
                    recommendation_id=f"agg_perf_{datetime.now().strftime('%Y%m%d')}",
                    category="performance",
                    priority="high",
                    title="Improve Overall Clinical Accuracy",
                    description="Overall clinical accuracy is below acceptable standards",
                    specific_actions=[
                        "Implement comprehensive training program",
                        "Increase supervision frequency",
                        "Review and update clinical protocols",
                        "Conduct peer review sessions",
                    ],
                    expected_impact="20-30% improvement in clinical accuracy",
                    timeline="4-6 weeks",
                    success_metrics=["Overall accuracy > 0.8", "Reduced dangerous assessments"],
                )
            )

        # Consistency-based recommendations
        if metrics["accuracy_std"] > 0.3:
            recommendations.append(
                ImprovementRecommendation(
                    recommendation_id=f"agg_cons_{datetime.now().strftime('%Y%m%d')}",
                    category="consistency",
                    priority="medium",
                    title="Improve Assessment Consistency",
                    description="High variability in assessment quality detected",
                    specific_actions=[
                        "Standardize assessment procedures",
                        "Implement quality checklists",
                        "Provide consistency training",
                        "Regular calibration exercises",
                    ],
                    expected_impact="Reduced variability and improved reliability",
                    timeline="3-4 weeks",
                )
            )

        # Trend-based recommendations
        for trend in trend_analyses:
            if trend.trend_direction == "declining" and trend.trend_strength > 0.5:
                recommendations.append(
                    ImprovementRecommendation(
                        recommendation_id=f"trend_{trend.metric.value}_{datetime.now().strftime('%Y%m%d')}",
                        category="trend_correction",
                        priority="high",
                        title=f"Address Declining {trend.metric.value.replace('_', ' ').title()}",
                        description=f"Negative trend detected in {trend.metric.value}",
                        specific_actions=[
                            f"Investigate root causes of {trend.metric.value} decline",
                            "Implement targeted interventions",
                            "Monitor progress closely",
                        ],
                        expected_impact="Reverse negative trend and restore performance",
                        timeline="2-3 weeks",
                    )
                )

        return recommendations

    async def _generate_immediate_feedback(self, assessment: ClinicalAccuracyResult) -> None:
        """Generate immediate feedback for an assessment"""
        try:
            feedback_messages = []

            # Critical safety feedback
            if assessment.safety_assessment.overall_risk.value in ["high", "critical"]:
                feedback_messages.append(
                    FeedbackMessage(
                        message_id=f"safety_{assessment.assessment_id}",
                        feedback_type=FeedbackType.PERFORMANCE_ALERT,
                        recipient="clinician",
                        title="CRITICAL SAFETY ALERT",
                        content=f"High safety risk detected in assessment {assessment.assessment_id}. Immediate intervention required.",
                        priority="high",
                        actionable_items=[
                            "Review safety protocols immediately",
                            "Implement crisis intervention if needed",
                            "Seek immediate supervision",
                        ],
                    )
                )

            # Accuracy feedback
            if assessment.overall_accuracy == ClinicalAccuracyLevel.EXCELLENT:
                feedback_messages.append(
                    FeedbackMessage(
                        message_id=f"excellence_{assessment.assessment_id}",
                        feedback_type=FeedbackType.MILESTONE_ACHIEVEMENT,
                        recipient="clinician",
                        title="Excellent Clinical Accuracy Achieved",
                        content=f"Outstanding performance in assessment {assessment.assessment_id}. Keep up the excellent work!",
                        priority="medium",
                    )
                )
            elif assessment.overall_accuracy == ClinicalAccuracyLevel.DANGEROUS:
                feedback_messages.append(
                    FeedbackMessage(
                        message_id=f"danger_{assessment.assessment_id}",
                        feedback_type=FeedbackType.PERFORMANCE_ALERT,
                        recipient="clinician",
                        title="DANGEROUS CLINICAL RESPONSE DETECTED",
                        content=f"Critical issues identified in assessment {assessment.assessment_id}. Immediate review and correction required.",
                        priority="high",
                        actionable_items=[
                            "Stop current approach immediately",
                            "Seek immediate supervision",
                            "Review clinical guidelines",
                            "Consider additional training",
                        ],
                    )
                )

            # Store feedback messages
            self.feedback_messages.extend(feedback_messages)

            # Deliver feedback (in production, this would send notifications)
            for message in feedback_messages:
                await self._deliver_feedback_message(message)

        except Exception as e:
            logger.error(f"Failed to generate immediate feedback: {e}")

    async def _deliver_feedback_message(self, message: FeedbackMessage) -> None:
        """Deliver feedback message to recipient"""
        try:
            # In production, this would send actual notifications
            # For now, just log the message
            logger.info(f"Feedback delivered: {message.title} to {message.recipient}")
            message.delivered = True

        except Exception as e:
            logger.error(f"Failed to deliver feedback message: {e}")

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        if not self.assessment_results:
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                overall_accuracy=0.0,
                dsm5_compliance=0.0,
                therapeutic_appropriateness=0.0,
                safety_compliance=0.0,
                ethics_compliance=0.0,
                expert_agreement=0.0,
                total_assessments=0,
            )

        # Calculate current metrics
        recent_assessments = self.assessment_results[-100:]  # Last 100 assessments

        accuracy_scores = [self._accuracy_to_score(a.overall_accuracy) for a in recent_assessments]
        dsm5_scores = [a.dsm5_assessment.diagnostic_confidence for a in recent_assessments]
        therapeutic_scores = [
            a.therapeutic_appropriateness.overall_score for a in recent_assessments
        ]
        safety_scores = [
            self._safety_risk_to_score(a.safety_assessment.overall_risk) for a in recent_assessments
        ]

        return PerformanceSnapshot(
            timestamp=datetime.now(),
            overall_accuracy=statistics.mean(accuracy_scores),
            dsm5_compliance=statistics.mean(dsm5_scores),
            therapeutic_appropriateness=statistics.mean(therapeutic_scores),
            safety_compliance=statistics.mean(safety_scores),
            ethics_compliance=0.8,  # Would calculate from ethics results
            expert_agreement=0.75,  # Would calculate from expert validations
            total_assessments=len(self.assessment_results),
        )

    def export_report(self, report_id: str, output_path: Path, format: str = "json") -> None:
        """Export report to file"""
        try:
            report = self.generated_reports.get(report_id)
            if not report:
                raise ValueError(f"Report {report_id} not found")

            if format.lower() == "json":
                # Convert to JSON-serializable format
                report_dict = {
                    "report_id": report.report_id,
                    "report_type": report.report_type.value,
                    "generated_at": report.generated_at.isoformat(),
                    "time_period": [
                        report.time_period[0].isoformat(),
                        report.time_period[1].isoformat(),
                    ],
                    "summary": report.summary,
                    "key_findings": report.key_findings,
                    "performance_metrics": report.performance_metrics,
                    "improvement_recommendations": [
                        {
                            "recommendation_id": rec.recommendation_id,
                            "category": rec.category,
                            "priority": rec.priority,
                            "title": rec.title,
                            "description": rec.description,
                            "specific_actions": rec.specific_actions,
                            "expected_impact": rec.expected_impact,
                            "timeline": rec.timeline,
                        }
                        for rec in report.improvement_recommendations
                    ],
                }

                with open(output_path, "w") as f:
                    json.dump(report_dict, f, indent=2)

            logger.info(f"Report {report_id} exported to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Initialize reporting system
        reporting_system = ClinicalReportingSystem()

        # Create sample assessment result
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

        context = ClinicalContext(
            client_presentation="Sample client case",
            therapeutic_modality=TherapeuticModality.CBT,
            session_phase="working",
        )

        assessment = ClinicalAccuracyResult(
            assessment_id="sample_001",
            timestamp=datetime.now(),
            clinical_context=context,
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

        # Add assessment to reporting system
        reporting_system.add_assessment_result(assessment)

        # Generate individual report
        individual_report = await reporting_system.generate_individual_assessment_report(
            "sample_001"
        )
        if individual_report:
            print(f"Individual Report Generated: {individual_report.report_id}")
            print(f"Summary: {individual_report.summary}")
            print(f"Key Findings: {individual_report.key_findings}")

        # Generate aggregate report
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        aggregate_report = await reporting_system.generate_aggregate_performance_report(
            start_date, end_date
        )
        print(f"\nAggregate Report Generated: {aggregate_report.report_id}")
        print(f"Summary: {aggregate_report.summary}")

        # Get performance snapshot
        snapshot = reporting_system.get_performance_snapshot()
        print("\nCurrent Performance Snapshot:")
        print(f"Overall Accuracy: {snapshot.overall_accuracy:.3f}")
        print(f"Total Assessments: {snapshot.total_assessments}")

    # Run example
    asyncio.run(main())
