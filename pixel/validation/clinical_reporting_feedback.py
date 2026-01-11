"""
Clinical Accuracy Reporting and Feedback Loop System

This module provides comprehensive reporting and feedback mechanisms for clinical
accuracy validation, integrating all validation systems into a unified dashboard
and continuous improvement loop.
"""

import asyncio
import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from automated_clinical_appropriateness import (
    AppropriatenessResult,
    ClinicalAppropriatenessChecker,
)
from clinical_accuracy_assessment import (
    ClinicalAccuracyAssessmentFramework,
    ClinicalAssessment,
    ClinicalDomain,
)
from expert_validation_workflow import (
    ExpertValidationWorkflow,
    ValidationRequest,
    WorkflowStatus,
)
from safety_ethics_compliance import (
    ComplianceResult,
    SafetyEthicsComplianceValidator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of clinical accuracy reports."""

    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MONTHLY_TRENDS = "monthly_trends"
    EXPERT_PERFORMANCE = "expert_performance"
    DOMAIN_ANALYSIS = "domain_analysis"
    COMPLIANCE_AUDIT = "compliance_audit"
    IMPROVEMENT_TRACKING = "improvement_tracking"
    ALERT_DASHBOARD = "alert_dashboard"


class FeedbackType(Enum):
    """Types of feedback in the improvement loop."""

    EXPERT_FEEDBACK = "expert_feedback"
    AUTOMATED_INSIGHTS = "automated_insights"
    PERFORMANCE_METRICS = "performance_metrics"
    TREND_ANALYSIS = "trend_analysis"
    RECOMMENDATION = "recommendation"
    ALERT = "alert"


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""

    total_validations: int = 0
    accuracy_distribution: Dict[str, int] = field(default_factory=dict)
    average_score: float = 0.0
    expert_consensus_rate: float = 0.0
    compliance_rate: float = 0.0
    safety_incidents: int = 0
    processing_time_avg: float = 0.0
    domain_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FeedbackItem:
    """Individual feedback item in the improvement loop."""

    feedback_id: str
    feedback_type: FeedbackType
    content_id: str
    domain: ClinicalDomain
    message: str
    severity: AlertSeverity
    actionable: bool
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ImprovementAction:
    """Improvement action based on feedback analysis."""

    action_id: str
    action_type: str
    description: str
    target_domain: Optional[ClinicalDomain]
    priority: AlertSeverity
    estimated_impact: float
    implementation_steps: List[str]
    success_metrics: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"


class ClinicalReportingFeedbackSystem:
    """
    Comprehensive clinical accuracy reporting and feedback loop system that
    integrates all validation components and provides continuous improvement.
    """

    def __init__(
        self,
        assessment_framework: ClinicalAccuracyAssessmentFramework,
        expert_workflow: ExpertValidationWorkflow,
        appropriateness_checker: ClinicalAppropriatenessChecker,
        compliance_validator: SafetyEthicsComplianceValidator,
        db_path: str = "clinical_reporting.db",
    ):
        """Initialize the reporting and feedback system."""
        self.assessment_framework = assessment_framework
        self.expert_workflow = expert_workflow
        self.appropriateness_checker = appropriateness_checker
        self.compliance_validator = compliance_validator

        self.db_path = db_path
        self.feedback_items: List[FeedbackItem] = []
        self.improvement_actions: List[ImprovementAction] = []
        self.alert_callbacks: List[Callable] = []

        self._initialize_database()
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration for reporting and feedback."""
        return {
            "report_retention_days": 90,
            "alert_thresholds": {
                "low_accuracy_rate": 0.7,
                "high_safety_incidents": 5,
                "expert_disagreement_rate": 0.3,
                "compliance_violation_rate": 0.1,
            },
            "feedback_processing_interval": 3600,  # 1 hour
            "auto_improvement_enabled": True,
            "notification_channels": ["email", "dashboard", "webhook"],
        }

    def _initialize_database(self):
        """Initialize SQLite database for persistent storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for metrics tracking
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_validations INTEGER,
                average_score REAL,
                expert_consensus_rate REAL,
                compliance_rate REAL,
                safety_incidents INTEGER,
                processing_time_avg REAL,
                domain_breakdown TEXT,
                accuracy_distribution TEXT
            )
        """
        )

        # Create table for feedback items
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_items (
                feedback_id TEXT PRIMARY KEY,
                feedback_type TEXT NOT NULL,
                content_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                actionable BOOLEAN,
                recommendations TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT
            )
        """
        )

        # Create table for improvement actions
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS improvement_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                description TEXT NOT NULL,
                target_domain TEXT,
                priority TEXT NOT NULL,
                estimated_impact REAL,
                implementation_steps TEXT,
                success_metrics TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """
        )

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def generate_comprehensive_report(
        self,
        report_type: ReportType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        domain_filter: Optional[ClinicalDomain] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical accuracy report."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.now()

        logger.info(f"Generating {report_type.value} report from {start_date} to {end_date}")

        # Collect data from all validation systems
        assessment_data = self._collect_assessment_data(start_date, end_date, domain_filter)
        expert_data = self._collect_expert_data(start_date, end_date, domain_filter)
        appropriateness_data = self._collect_appropriateness_data(
            start_date, end_date, domain_filter
        )
        compliance_data = self._collect_compliance_data(start_date, end_date, domain_filter)

        # Generate report based on type
        if report_type == ReportType.DAILY_SUMMARY:
            return self._generate_daily_summary(
                assessment_data, expert_data, appropriateness_data, compliance_data
            )
        elif report_type == ReportType.WEEKLY_ANALYSIS:
            return self._generate_weekly_analysis(
                assessment_data, expert_data, appropriateness_data, compliance_data
            )
        elif report_type == ReportType.MONTHLY_TRENDS:
            return self._generate_monthly_trends(
                assessment_data, expert_data, appropriateness_data, compliance_data
            )
        elif report_type == ReportType.EXPERT_PERFORMANCE:
            return self._generate_expert_performance_report(expert_data)
        elif report_type == ReportType.DOMAIN_ANALYSIS:
            return self._generate_domain_analysis(
                assessment_data, expert_data, appropriateness_data, compliance_data, domain_filter
            )
        elif report_type == ReportType.COMPLIANCE_AUDIT:
            return self._generate_compliance_audit(compliance_data)
        elif report_type == ReportType.IMPROVEMENT_TRACKING:
            return self._generate_improvement_tracking_report()
        elif report_type == ReportType.ALERT_DASHBOARD:
            return self._generate_alert_dashboard()
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

    def _collect_assessment_data(
        self, start_date: datetime, end_date: datetime, domain_filter: Optional[ClinicalDomain]
    ) -> List[ClinicalAssessment]:
        """Collect assessment data from the framework."""
        assessments = []

        # Get assessments from the framework
        if domain_filter:
            domain_assessments = self.assessment_framework.get_assessments_by_domain(domain_filter)
        else:
            domain_assessments = list(self.assessment_framework.assessments.values())

        # Filter by date range
        for assessment in domain_assessments:
            if start_date <= assessment.created_at <= end_date:
                assessments.append(assessment)

        return assessments

    def _collect_expert_data(
        self, start_date: datetime, end_date: datetime, domain_filter: Optional[ClinicalDomain]
    ) -> List[ValidationRequest]:
        """Collect expert validation data."""
        requests = []

        for request in self.expert_workflow.validation_requests.values():
            if start_date <= request.created_at <= end_date:
                if domain_filter is None or request.domain == domain_filter:
                    requests.append(request)

        return requests

    def _collect_appropriateness_data(
        self, start_date: datetime, end_date: datetime, domain_filter: Optional[ClinicalDomain]
    ) -> List[AppropriatenessResult]:
        """Collect appropriateness checking data."""
        # This would typically come from a database or cache
        # For now, return empty list as we don't have persistent storage
        return []

    def _collect_compliance_data(
        self, start_date: datetime, end_date: datetime, domain_filter: Optional[ClinicalDomain]
    ) -> List[ComplianceResult]:
        """Collect compliance validation data."""
        # This would typically come from a database or cache
        # For now, return empty list as we don't have persistent storage
        return []

    def _generate_daily_summary(
        self,
        assessments: List[ClinicalAssessment],
        expert_requests: List[ValidationRequest],
        appropriateness_results: List[AppropriatenessResult],
        compliance_results: List[ComplianceResult],
    ) -> Dict[str, Any]:
        """Generate daily summary report."""
        return {
            "report_type": "daily_summary",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": {
                "total_assessments": len(assessments),
                "total_expert_requests": len(expert_requests),
                "total_appropriateness_checks": len(appropriateness_results),
                "total_compliance_checks": len(compliance_results),
            },
            "accuracy_metrics": self._calculate_accuracy_metrics(assessments),
            "expert_metrics": self._calculate_expert_metrics(expert_requests),
            "appropriateness_metrics": self._calculate_appropriateness_metrics(
                appropriateness_results
            ),
            "compliance_metrics": self._calculate_compliance_metrics(compliance_results),
            "alerts": self._generate_daily_alerts(
                assessments, expert_requests, appropriateness_results, compliance_results
            ),
            "recommendations": self._generate_daily_recommendations(
                assessments, expert_requests, appropriateness_results, compliance_results
            ),
        }

    def _calculate_accuracy_metrics(self, assessments: List[ClinicalAssessment]) -> Dict[str, Any]:
        """Calculate accuracy metrics from assessments."""
        if not assessments:
            return {"average_score": 0.0, "distribution": {}, "total": 0}

        scores = [a.overall_score for a in assessments]
        levels = [a.accuracy_level.value for a in assessments]

        return {
            "average_score": np.mean(scores),
            "median_score": np.median(scores),
            "distribution": dict(Counter(levels)),
            "total": len(assessments),
            "above_threshold": len([s for s in scores if s >= 0.8]),
        }

    def _calculate_expert_metrics(self, requests: List[ValidationRequest]) -> Dict[str, Any]:
        """Calculate expert validation metrics."""
        if not requests:
            return {"total": 0, "completed": 0, "consensus_rate": 0.0}

        completed = [r for r in requests if r.status == WorkflowStatus.COMPLETED]
        consensus_reached = [r for r in requests if r.status == WorkflowStatus.CONSENSUS_REACHED]

        return {
            "total": len(requests),
            "completed": len(completed),
            "consensus_reached": len(consensus_reached),
            "consensus_rate": len(consensus_reached) / len(requests) if requests else 0.0,
            "average_reviewers": (
                np.mean([len(r.completed_reviews) for r in completed]) if completed else 0.0
            ),
        }

    def _calculate_appropriateness_metrics(
        self, results: List[AppropriatenessResult]
    ) -> Dict[str, Any]:
        """Calculate appropriateness checking metrics."""
        if not results:
            return {"total": 0, "pass_rate": 0.0, "average_score": 0.0}

        scores = [r.overall_score for r in results]
        levels = [r.overall_level.value for r in results]

        return {
            "total": len(results),
            "average_score": np.mean(scores),
            "pass_rate": len([s for s in scores if s >= 0.7]) / len(scores),
            "level_distribution": dict(Counter(levels)),
            "violations_found": sum(len(r.violations) for r in results),
        }

    def _calculate_compliance_metrics(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Calculate compliance validation metrics."""
        if not results:
            return {"total": 0, "compliance_rate": 0.0, "critical_violations": 0}

        scores = [r.overall_score for r in results]
        levels = [r.overall_level.value for r in results]
        critical_actions = [r for r in results if r.requires_immediate_action]

        return {
            "total": len(results),
            "average_score": np.mean(scores),
            "compliance_rate": len([s for s in scores if s >= 0.8]) / len(scores),
            "level_distribution": dict(Counter(levels)),
            "critical_violations": len(critical_actions),
            "safety_incidents": sum(len(r.safety_assessments) for r in results),
        }

    def process_feedback_loop(self) -> List[ImprovementAction]:
        """Process the continuous improvement feedback loop."""
        logger.info("Processing feedback loop for continuous improvement")

        # Analyze recent performance data
        recent_data = self._analyze_recent_performance()

        # Generate feedback items
        feedback_items = self._generate_feedback_items(recent_data)

        # Create improvement actions
        improvement_actions = self._create_improvement_actions(feedback_items)

        # Store feedback and actions
        self._store_feedback_items(feedback_items)
        self._store_improvement_actions(improvement_actions)

        # Trigger alerts if necessary
        self._process_alerts(feedback_items)

        return improvement_actions

    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance across all validation systems."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days

        # Collect recent data
        assessments = self._collect_assessment_data(start_date, end_date, None)
        expert_requests = self._collect_expert_data(start_date, end_date, None)
        appropriateness_results = self._collect_appropriateness_data(start_date, end_date, None)
        compliance_results = self._collect_compliance_data(start_date, end_date, None)

        return {
            "assessments": assessments,
            "expert_requests": expert_requests,
            "appropriateness_results": appropriateness_results,
            "compliance_results": compliance_results,
            "period": {"start": start_date, "end": end_date},
        }

    def _generate_feedback_items(self, performance_data: Dict[str, Any]) -> List[FeedbackItem]:
        """Generate feedback items based on performance analysis."""
        feedback_items = []

        # Analyze accuracy trends
        assessments = performance_data["assessments"]
        if assessments:
            avg_score = np.mean([a.overall_score for a in assessments])
            if avg_score < self.config["alert_thresholds"]["low_accuracy_rate"]:
                feedback_items.append(
                    FeedbackItem(
                        feedback_id=f"accuracy_low_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        feedback_type=FeedbackType.PERFORMANCE_METRICS,
                        content_id="system_wide",
                        domain=ClinicalDomain.DSM5_DIAGNOSTIC,  # Default domain
                        message=f"Average accuracy score ({avg_score:.3f}) below threshold",
                        severity=AlertSeverity.HIGH,
                        actionable=True,
                        recommendations=[
                            "Review recent assessments for common issues",
                            "Provide additional training for low-performing areas",
                            "Adjust assessment criteria if necessary",
                        ],
                    )
                )

        # Analyze expert consensus
        expert_requests = performance_data["expert_requests"]
        if expert_requests:
            completed = [r for r in expert_requests if r.status == WorkflowStatus.COMPLETED]
            if completed:
                consensus_rate = len(
                    [r for r in expert_requests if r.status == WorkflowStatus.CONSENSUS_REACHED]
                ) / len(completed)
                if consensus_rate < (
                    1 - self.config["alert_thresholds"]["expert_disagreement_rate"]
                ):
                    feedback_items.append(
                        FeedbackItem(
                            feedback_id=f"consensus_low_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            feedback_type=FeedbackType.EXPERT_FEEDBACK,
                            content_id="expert_system",
                            domain=ClinicalDomain.DSM5_DIAGNOSTIC,
                            message=f"Expert consensus rate ({consensus_rate:.3f}) below expected level",
                            severity=AlertSeverity.MEDIUM,
                            actionable=True,
                            recommendations=[
                                "Review cases with low expert consensus",
                                "Provide additional expert training",
                                "Clarify assessment criteria",
                            ],
                        )
                    )

        return feedback_items

    def _create_improvement_actions(
        self, feedback_items: List[FeedbackItem]
    ) -> List[ImprovementAction]:
        """Create improvement actions based on feedback analysis."""
        actions = []

        # Group feedback by type and severity
        high_priority_items = [
            f for f in feedback_items if f.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]
        ]

        for item in high_priority_items:
            if item.feedback_type == FeedbackType.PERFORMANCE_METRICS:
                action = ImprovementAction(
                    action_id=f"improve_accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    action_type="accuracy_improvement",
                    description="Implement accuracy improvement measures",
                    target_domain=item.domain,
                    priority=item.severity,
                    estimated_impact=0.15,  # Estimated 15% improvement
                    implementation_steps=[
                        "Analyze low-scoring assessments",
                        "Identify common failure patterns",
                        "Update assessment criteria",
                        "Provide targeted training",
                        "Monitor improvement",
                    ],
                    success_metrics=[
                        "Average accuracy score > 0.8",
                        "Reduction in low-scoring assessments by 50%",
                        "Improved expert consensus rate",
                    ],
                )
                actions.append(action)

        return actions

    def _store_feedback_items(self, feedback_items: List[FeedbackItem]):
        """Store feedback items in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for item in feedback_items:
            cursor.execute(
                """
                INSERT OR REPLACE INTO feedback_items 
                (feedback_id, feedback_type, content_id, domain, message, severity, 
                 actionable, recommendations, metadata, created_at, resolved, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    item.feedback_id,
                    item.feedback_type.value,
                    item.content_id,
                    item.domain.value,
                    item.message,
                    item.severity.value,
                    item.actionable,
                    json.dumps(item.recommendations),
                    json.dumps(item.metadata),
                    item.created_at.isoformat(),
                    item.resolved,
                    item.resolved_at.isoformat() if item.resolved_at else None,
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(feedback_items)} feedback items")

    def _store_improvement_actions(self, actions: List[ImprovementAction]):
        """Store improvement actions in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for action in actions:
            cursor.execute(
                """
                INSERT OR REPLACE INTO improvement_actions
                (action_id, action_type, description, target_domain, priority,
                 estimated_impact, implementation_steps, success_metrics, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    action.action_id,
                    action.action_type,
                    action.description,
                    action.target_domain.value if action.target_domain else None,
                    action.priority.value,
                    action.estimated_impact,
                    json.dumps(action.implementation_steps),
                    json.dumps(action.success_metrics),
                    action.created_at.isoformat(),
                    action.status,
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"Stored {len(actions)} improvement actions")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        # Generate current metrics
        current_metrics = self.generate_comprehensive_report(ReportType.DAILY_SUMMARY)

        # Get recent feedback items
        recent_feedback = self._get_recent_feedback_items(hours=24)

        # Get active improvement actions
        active_actions = self._get_active_improvement_actions()

        # Get system health indicators
        health_indicators = self._get_system_health_indicators()

        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "recent_feedback": [
                {
                    "feedback_id": f.feedback_id,
                    "type": f.feedback_type.value,
                    "message": f.message,
                    "severity": f.severity.value,
                    "created_at": f.created_at.isoformat(),
                    "resolved": f.resolved,
                }
                for f in recent_feedback
            ],
            "active_actions": [
                {
                    "action_id": a.action_id,
                    "type": a.action_type,
                    "description": a.description,
                    "priority": a.priority.value,
                    "status": a.status,
                    "estimated_impact": a.estimated_impact,
                }
                for a in active_actions
            ],
            "health_indicators": health_indicators,
        }

    def _get_recent_feedback_items(self, hours: int = 24) -> List[FeedbackItem]:
        """Get recent feedback items from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute(
            """
            SELECT * FROM feedback_items 
            WHERE created_at > ? 
            ORDER BY created_at DESC
        """,
            (cutoff_time,),
        )

        rows = cursor.fetchall()
        conn.close()

        feedback_items = []
        for row in rows:
            feedback_items.append(
                FeedbackItem(
                    feedback_id=row[0],
                    feedback_type=FeedbackType(row[1]),
                    content_id=row[2],
                    domain=ClinicalDomain(row[3]),
                    message=row[4],
                    severity=AlertSeverity(row[5]),
                    actionable=bool(row[6]),
                    recommendations=json.loads(row[7]) if row[7] else [],
                    metadata=json.loads(row[8]) if row[8] else {},
                    created_at=datetime.fromisoformat(row[9]),
                    resolved=bool(row[10]),
                    resolved_at=datetime.fromisoformat(row[11]) if row[11] else None,
                )
            )

        return feedback_items

    def _get_active_improvement_actions(self) -> List[ImprovementAction]:
        """Get active improvement actions from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM improvement_actions 
            WHERE status IN ('pending', 'in_progress')
            ORDER BY priority DESC, created_at DESC
        """
        )

        rows = cursor.fetchall()
        conn.close()

        actions = []
        for row in rows:
            actions.append(
                ImprovementAction(
                    action_id=row[0],
                    action_type=row[1],
                    description=row[2],
                    target_domain=ClinicalDomain(row[3]) if row[3] else None,
                    priority=AlertSeverity(row[4]),
                    estimated_impact=row[5],
                    implementation_steps=json.loads(row[6]) if row[6] else [],
                    success_metrics=json.loads(row[7]) if row[7] else [],
                    created_at=datetime.fromisoformat(row[8]),
                    status=row[9],
                )
            )

        return actions

    def _get_system_health_indicators(self) -> Dict[str, Any]:
        """Get system health indicators."""
        return {
            "database_status": "healthy",
            "validation_systems_status": "operational",
            "expert_workflow_status": "operational",
            "feedback_loop_status": "active",
            "last_update": datetime.now().isoformat(),
        }

    async def run_continuous_monitoring(self, interval_seconds: int = 3600):
        """Run continuous monitoring and feedback processing."""
        logger.info(f"Starting continuous monitoring with {interval_seconds}s interval")

        while True:
            try:
                # Process feedback loop
                improvement_actions = self.process_feedback_loop()

                if improvement_actions:
                    logger.info(f"Generated {len(improvement_actions)} improvement actions")

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Example usage and testing
if __name__ == "__main__":
    # This would typically be initialized with actual validation systems
    # For testing, we'll create mock instances

    from automated_clinical_appropriateness import ClinicalAppropriatenessChecker
    from clinical_accuracy_assessment import ClinicalAccuracyAssessmentFramework
    from expert_validation_workflow import ExpertValidationWorkflow
    from safety_ethics_compliance import SafetyEthicsComplianceValidator

    # Initialize validation systems
    assessment_framework = ClinicalAccuracyAssessmentFramework()
    expert_workflow = ExpertValidationWorkflow(assessment_framework)
    appropriateness_checker = ClinicalAppropriatenessChecker()
    compliance_validator = SafetyEthicsComplianceValidator()

    # Initialize reporting system
    reporting_system = ClinicalReportingFeedbackSystem(
        assessment_framework=assessment_framework,
        expert_workflow=expert_workflow,
        appropriateness_checker=appropriateness_checker,
        compliance_validator=compliance_validator,
    )

    # Generate sample report
    report = reporting_system.generate_comprehensive_report(ReportType.DAILY_SUMMARY)
    print(json.dumps(report, indent=2))

    # Process feedback loop
    actions = reporting_system.process_feedback_loop()
    print(f"Generated {len(actions)} improvement actions")

    # Get dashboard data
    dashboard = reporting_system.get_dashboard_data()
    print(f"Dashboard data: {len(dashboard)} sections")
