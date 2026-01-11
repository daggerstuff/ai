"""
Clinical Accuracy Reporting and Feedback Loop System

This module provides comprehensive reporting capabilities for clinical accuracy
assessments, including trend analysis, expert feedback integration, and
continuous improvement mechanisms.
"""

import base64
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .clinical_accuracy_validator import (
    ClinicalAccuracyLevel,
    ClinicalAccuracyResult,
    ClinicalAccuracyValidator,
    SafetyRiskLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of clinical accuracy reports"""

    INDIVIDUAL_ASSESSMENT = "individual_assessment"
    TREND_ANALYSIS = "trend_analysis"
    EXPERT_FEEDBACK = "expert_feedback"
    IMPROVEMENT_RECOMMENDATIONS = "improvement_recommendations"
    SAFETY_ALERTS = "safety_alerts"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


class FeedbackPriority(Enum):
    """Priority levels for feedback items"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class ExpertFeedback:
    """Expert feedback on clinical accuracy assessments"""

    feedback_id: str
    assessment_id: str
    expert_id: str
    expert_credentials: str
    feedback_type: str  # validation, correction, enhancement
    accuracy_rating: float  # 0.0 to 1.0
    detailed_comments: str
    specific_corrections: Dict[str, str] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, reviewed, implemented


@dataclass
class TrendAnalysis:
    """Trend analysis results for clinical accuracy"""

    analysis_id: str
    time_period: Tuple[datetime, datetime]
    total_assessments: int
    accuracy_distribution: Dict[ClinicalAccuracyLevel, int]
    average_confidence: float
    common_issues: List[Tuple[str, int]]  # (issue, frequency)
    improvement_areas: List[str]
    safety_incidents: int
    expert_agreement_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementRecommendation:
    """Improvement recommendation based on analysis"""

    recommendation_id: str
    category: str  # training, guidelines, validation, safety
    priority: FeedbackPriority
    description: str
    rationale: str
    implementation_steps: List[str]
    expected_impact: str
    resources_needed: List[str]
    timeline: str
    success_metrics: List[str]
    status: str = "proposed"  # proposed, approved, in_progress, completed


class ClinicalAccuracyReporter:
    """
    Clinical accuracy reporting and feedback loop system

    Provides comprehensive reporting, trend analysis, and continuous
    improvement mechanisms for clinical accuracy assessments.
    """

    def __init__(self, db_path: Optional[Path] = None, config_path: Optional[Path] = None):
        """Initialize the clinical accuracy reporter"""
        self.db_path = db_path or Path("clinical_accuracy_reports.db")
        self.config = self._load_config(config_path)
        self.validator = ClinicalAccuracyValidator()

        # Initialize database
        self._init_database()

        # Report generation settings
        self.report_settings = {
            "include_visualizations": True,
            "export_formats": ["json", "html", "pdf"],
            "trend_analysis_window": 30,  # days
            "min_assessments_for_trends": 10,
        }

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration settings"""
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)

        return {
            "expert_validation_threshold": 0.7,
            "safety_alert_threshold": SafetyRiskLevel.MODERATE,
            "trend_analysis_frequency": "daily",
            "feedback_integration_mode": "automatic",
            "report_retention_days": 365,
        }

    def _init_database(self):
        """Initialize SQLite database for storing reports and feedback"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Assessments table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    assessment_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    overall_accuracy TEXT,
                    confidence_score REAL,
                    therapeutic_modality TEXT,
                    safety_risk_level TEXT,
                    expert_validation_needed BOOLEAN,
                    assessment_data TEXT
                )
            """
            )

            # Expert feedback table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS expert_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    assessment_id TEXT,
                    expert_id TEXT,
                    expert_credentials TEXT,
                    feedback_type TEXT,
                    accuracy_rating REAL,
                    detailed_comments TEXT,
                    priority TEXT,
                    timestamp DATETIME,
                    status TEXT,
                    feedback_data TEXT,
                    FOREIGN KEY (assessment_id) REFERENCES assessments (assessment_id)
                )
            """
            )

            # Trend analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trend_analysis (
                    analysis_id TEXT PRIMARY KEY,
                    start_date DATETIME,
                    end_date DATETIME,
                    total_assessments INTEGER,
                    average_confidence REAL,
                    safety_incidents INTEGER,
                    expert_agreement_rate REAL,
                    timestamp DATETIME,
                    analysis_data TEXT
                )
            """
            )

            # Improvement recommendations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS improvement_recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    category TEXT,
                    priority TEXT,
                    description TEXT,
                    rationale TEXT,
                    expected_impact TEXT,
                    timeline TEXT,
                    status TEXT,
                    timestamp DATETIME,
                    recommendation_data TEXT
                )
            """
            )

            conn.commit()

    async def store_assessment(self, result: ClinicalAccuracyResult):
        """Store clinical accuracy assessment result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO assessments 
                    (assessment_id, timestamp, overall_accuracy, confidence_score,
                     therapeutic_modality, safety_risk_level, expert_validation_needed,
                     assessment_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        result.assessment_id,
                        result.timestamp,
                        result.overall_accuracy.value,
                        result.confidence_score,
                        result.clinical_context.therapeutic_modality.value,
                        result.safety_assessment.overall_risk.value,
                        result.expert_validation_needed,
                        json.dumps(self._serialize_assessment(result)),
                    ),
                )

                conn.commit()
                logger.info(f"Stored assessment {result.assessment_id}")

        except Exception as e:
            logger.error(f"Error storing assessment: {e}")
            raise

    def _serialize_assessment(self, result: ClinicalAccuracyResult) -> Dict[str, Any]:
        """Serialize assessment result for storage"""
        return {
            "assessment_id": result.assessment_id,
            "timestamp": result.timestamp.isoformat(),
            "clinical_context": {
                "client_presentation": result.clinical_context.client_presentation,
                "therapeutic_modality": result.clinical_context.therapeutic_modality.value,
                "session_phase": result.clinical_context.session_phase,
                "crisis_indicators": result.clinical_context.crisis_indicators,
                "cultural_factors": result.clinical_context.cultural_factors,
                "contraindications": result.clinical_context.contraindications,
            },
            "dsm5_assessment": {
                "primary_diagnosis": result.dsm5_assessment.primary_diagnosis,
                "secondary_diagnoses": result.dsm5_assessment.secondary_diagnoses,
                "diagnostic_confidence": result.dsm5_assessment.diagnostic_confidence,
                "criteria_met": result.dsm5_assessment.criteria_met,
                "criteria_not_met": result.dsm5_assessment.criteria_not_met,
                "differential_diagnoses": result.dsm5_assessment.differential_diagnoses,
                "severity_specifiers": result.dsm5_assessment.severity_specifiers,
            },
            "pdm2_assessment": {
                "personality_patterns": result.pdm2_assessment.personality_patterns,
                "mental_functioning": result.pdm2_assessment.mental_functioning,
                "symptom_patterns": result.pdm2_assessment.symptom_patterns,
                "subjective_experience": result.pdm2_assessment.subjective_experience,
                "relational_patterns": result.pdm2_assessment.relational_patterns,
            },
            "therapeutic_appropriateness": {
                "intervention_appropriateness": float(
                    result.therapeutic_appropriateness.intervention_appropriateness
                ),
                "timing_appropriateness": float(
                    result.therapeutic_appropriateness.timing_appropriateness
                ),
                "cultural_sensitivity": float(
                    result.therapeutic_appropriateness.cultural_sensitivity
                ),
                "ethical_compliance": float(result.therapeutic_appropriateness.ethical_compliance),
                "boundary_maintenance": float(
                    result.therapeutic_appropriateness.boundary_maintenance
                ),
                "overall_score": float(result.therapeutic_appropriateness.overall_score),
                "rationale": result.therapeutic_appropriateness.rationale,
            },
            "safety_assessment": {
                "suicide_risk": result.safety_assessment.suicide_risk.value,
                "self_harm_risk": result.safety_assessment.self_harm_risk.value,
                "violence_risk": result.safety_assessment.violence_risk.value,
                "substance_abuse_risk": result.safety_assessment.substance_abuse_risk.value,
                "psychosis_risk": result.safety_assessment.psychosis_risk.value,
                "overall_risk": result.safety_assessment.overall_risk.value,
                "immediate_interventions": result.safety_assessment.immediate_interventions,
                "safety_plan_needed": result.safety_assessment.safety_plan_needed,
            },
            "overall_accuracy": result.overall_accuracy.value,
            "confidence_score": result.confidence_score,
            "expert_validation_needed": result.expert_validation_needed,
            "recommendations": result.recommendations,
            "warnings": result.warnings,
            "metadata": result.metadata,
        }

    async def add_expert_feedback(self, feedback: ExpertFeedback):
        """Add expert feedback for an assessment"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO expert_feedback 
                    (feedback_id, assessment_id, expert_id, expert_credentials,
                     feedback_type, accuracy_rating, detailed_comments, priority,
                     timestamp, status, feedback_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        feedback.feedback_id,
                        feedback.assessment_id,
                        feedback.expert_id,
                        feedback.expert_credentials,
                        feedback.feedback_type,
                        feedback.accuracy_rating,
                        feedback.detailed_comments,
                        feedback.priority.value,
                        feedback.timestamp,
                        feedback.status,
                        json.dumps(
                            {
                                "specific_corrections": feedback.specific_corrections,
                                "recommendations": feedback.recommendations,
                            }
                        ),
                    ),
                )

                conn.commit()
                logger.info(f"Added expert feedback {feedback.feedback_id}")

                # Trigger feedback integration if configured
                if self.config.get("feedback_integration_mode") == "automatic":
                    await self._integrate_feedback(feedback)

        except Exception as e:
            logger.error(f"Error adding expert feedback: {e}")
            raise

    async def _integrate_feedback(self, feedback: ExpertFeedback):
        """Integrate expert feedback into the system"""
        try:
            # Update assessment accuracy based on expert feedback
            if feedback.accuracy_rating < self.config.get("expert_validation_threshold", 0.7):
                # Generate improvement recommendation
                recommendation = ImprovementRecommendation(
                    recommendation_id=f"rec_{feedback.feedback_id}",
                    category="validation",
                    priority=feedback.priority,
                    description=f"Address expert concerns for assessment {feedback.assessment_id}",
                    rationale=feedback.detailed_comments,
                    implementation_steps=feedback.recommendations,
                    expected_impact="Improved clinical accuracy validation",
                    resources_needed=["Expert review", "Training updates"],
                    timeline="1-2 weeks",
                    success_metrics=["Improved expert agreement rate", "Higher accuracy scores"],
                )

                await self.add_improvement_recommendation(recommendation)

            # Update feedback status
            await self._update_feedback_status(feedback.feedback_id, "integrated")

        except Exception as e:
            logger.error(f"Error integrating feedback: {e}")

    async def _update_feedback_status(self, feedback_id: str, status: str):
        """Update feedback status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE expert_feedback SET status = ? WHERE feedback_id = ?
            """,
                (status, feedback_id),
            )
            conn.commit()

    async def generate_trend_analysis(self, days: int = 30) -> Optional[TrendAnalysis]:
        """Generate trend analysis for specified time period"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            with sqlite3.connect(self.db_path) as conn:
                # Get assessments in time period
                df = pd.read_sql_query(
                    """
                    SELECT * FROM assessments 
                    WHERE timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                """,
                    conn,
                    params=(start_date, end_date),
                )

                min_assessments = self.report_settings.get("min_assessments_for_trends", 10)
                if len(df) < min_assessments:
                    logger.warning(f"Insufficient data for trend analysis: {len(df)} assessments")
                    return None

                # Calculate accuracy distribution
                accuracy_dist = df["overall_accuracy"].value_counts().to_dict()
                accuracy_distribution = {
                    ClinicalAccuracyLevel(k): v for k, v in accuracy_dist.items()
                }

                # Calculate average confidence
                avg_confidence = df["confidence_score"].mean()

                # Identify common issues
                common_issues = self._identify_common_issues(df)

                # Count safety incidents
                safety_incidents = len(df[df["safety_risk_level"].isin(["high", "critical"])])

                # Calculate expert agreement rate
                expert_agreement_rate = await self._calculate_expert_agreement_rate(
                    df["assessment_id"].tolist()
                )

                # Generate improvement areas
                improvement_areas = self._identify_improvement_areas(df, common_issues)

                analysis = TrendAnalysis(
                    analysis_id=f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    time_period=(start_date, end_date),
                    total_assessments=len(df),
                    accuracy_distribution=accuracy_distribution,
                    average_confidence=avg_confidence,
                    common_issues=common_issues,
                    improvement_areas=improvement_areas,
                    safety_incidents=safety_incidents,
                    expert_agreement_rate=expert_agreement_rate,
                )

                # Store analysis
                await self._store_trend_analysis(analysis)

                return analysis

        except Exception as e:
            logger.error(f"Error generating trend analysis: {e}")
            raise

    def _identify_common_issues(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Identify common issues from assessment data"""
        issues = []

        # Low confidence assessments
        low_confidence = len(df[df["confidence_score"] < 0.7])
        if low_confidence > 0:
            issues.append(("Low confidence assessments", low_confidence))

        # Expert validation needed
        expert_validation = len(df[df["expert_validation_needed"]])
        if expert_validation > 0:
            issues.append(("Expert validation required", expert_validation))

        # Safety concerns
        safety_concerns = len(df[df["safety_risk_level"].isin(["moderate", "high", "critical"])])
        if safety_concerns > 0:
            issues.append(("Safety risk assessments", safety_concerns))

        # Poor accuracy ratings
        poor_accuracy = len(df[df["overall_accuracy"].isin(["concerning", "dangerous"])])
        if poor_accuracy > 0:
            issues.append(("Poor accuracy ratings", poor_accuracy))

        return sorted(issues, key=lambda x: x[1], reverse=True)

    async def _calculate_expert_agreement_rate(self, assessment_ids: List[str]) -> float:
        """Calculate expert agreement rate for assessments"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get expert feedback for these assessments
                cursor.execute(
                    """
                    SELECT assessment_id, accuracy_rating 
                    FROM expert_feedback 
                    WHERE assessment_id IN ({})
                    AND status = 'reviewed'
                """.format(
                        ",".join("?" * len(assessment_ids))
                    ),
                    assessment_ids,
                )

                feedback_data = cursor.fetchall()

                if not feedback_data:
                    return 0.0

                # Calculate agreement rate (ratings above threshold)
                agreement_threshold = self.config.get("expert_validation_threshold", 0.7)
                agreements = sum(1 for _, rating in feedback_data if rating >= agreement_threshold)

                return agreements / len(feedback_data) if feedback_data else 0.0

        except Exception as e:
            logger.error(f"Error calculating expert agreement rate: {e}")
            return 0.0

    def _identify_improvement_areas(
        self, df: pd.DataFrame, common_issues: List[Tuple[str, int]]
    ) -> List[str]:
        """Identify areas for improvement based on analysis"""
        areas = []

        # Based on common issues
        for issue, count in common_issues:
            if "Low confidence" in issue:
                areas.append("Improve diagnostic confidence through additional training")
            elif "Expert validation" in issue:
                areas.append("Enhance automated validation algorithms")
            elif "Safety risk" in issue:
                areas.append("Strengthen safety assessment protocols")
            elif "Poor accuracy" in issue:
                areas.append("Review and update clinical guidelines")

        # Based on accuracy distribution
        accuracy_counts = df["overall_accuracy"].value_counts()
        if (
            accuracy_counts.get("concerning", 0) + accuracy_counts.get("dangerous", 0)
            > len(df) * 0.1
        ):
            areas.append("Comprehensive review of assessment methodology")

        # Based on therapeutic modalities
        modality_accuracy = df.groupby("therapeutic_modality")["confidence_score"].mean()
        low_performing_modalities = modality_accuracy[modality_accuracy < 0.7].index.tolist()
        if low_performing_modalities:
            areas.append(
                f"Specialized training for {', '.join(low_performing_modalities)} modalities"
            )

        return list(set(areas))  # Remove duplicates

    async def _store_trend_analysis(self, analysis: TrendAnalysis):
        """Store trend analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO trend_analysis 
                    (analysis_id, start_date, end_date, total_assessments,
                     average_confidence, safety_incidents, expert_agreement_rate,
                     timestamp, analysis_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        analysis.analysis_id,
                        analysis.time_period[0],
                        analysis.time_period[1],
                        analysis.total_assessments,
                        analysis.average_confidence,
                        analysis.safety_incidents,
                        analysis.expert_agreement_rate,
                        analysis.timestamp,
                        json.dumps(
                            {
                                "accuracy_distribution": {
                                    k.value: v for k, v in analysis.accuracy_distribution.items()
                                },
                                "common_issues": analysis.common_issues,
                                "improvement_areas": analysis.improvement_areas,
                            }
                        ),
                    ),
                )

                conn.commit()
                logger.info(f"Stored trend analysis {analysis.analysis_id}")

        except Exception as e:
            logger.error(f"Error storing trend analysis: {e}")
            raise

    async def add_improvement_recommendation(self, recommendation: ImprovementRecommendation):
        """Add improvement recommendation"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO improvement_recommendations 
                    (recommendation_id, category, priority, description, rationale,
                     expected_impact, timeline, status, timestamp, recommendation_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        recommendation.recommendation_id,
                        recommendation.category,
                        recommendation.priority.value,
                        recommendation.description,
                        recommendation.rationale,
                        recommendation.expected_impact,
                        recommendation.timeline,
                        recommendation.status,
                        datetime.now(),
                        json.dumps(
                            {
                                "implementation_steps": recommendation.implementation_steps,
                                "resources_needed": recommendation.resources_needed,
                                "success_metrics": recommendation.success_metrics,
                            }
                        ),
                    ),
                )

                conn.commit()
                logger.info(f"Added improvement recommendation {recommendation.recommendation_id}")

        except Exception as e:
            logger.error(f"Error adding improvement recommendation: {e}")
            raise

    async def generate_comprehensive_report(
        self, report_type: ReportType, **kwargs
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical accuracy report"""
        try:
            report_data = {
                "report_id": f"{report_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "report_type": report_type.value,
                "timestamp": datetime.now().isoformat(),
                "metadata": kwargs,
            }

            if report_type == ReportType.INDIVIDUAL_ASSESSMENT:
                assessment_id = kwargs.get("assessment_id")
                if not assessment_id:
                    raise ValueError("assessment_id is required for individual assessment reports")
                report_data.update(await self._generate_individual_report(assessment_id))

            elif report_type == ReportType.TREND_ANALYSIS:
                days = kwargs.get("days", 30)
                trend_analysis = await self.generate_trend_analysis(days)
                if trend_analysis:
                    report_data.update(self._format_trend_report(trend_analysis))

            elif report_type == ReportType.EXPERT_FEEDBACK:
                report_data.update(await self._generate_expert_feedback_report(kwargs))

            elif report_type == ReportType.IMPROVEMENT_RECOMMENDATIONS:
                report_data.update(await self._generate_improvement_report(kwargs))

            elif report_type == ReportType.SAFETY_ALERTS:
                report_data.update(await self._generate_safety_alerts_report(kwargs))

            elif report_type == ReportType.COMPARATIVE_ANALYSIS:
                report_data.update(await self._generate_comparative_report(kwargs))

            # Add visualizations if enabled
            if self.report_settings["include_visualizations"]:
                report_data["visualizations"] = await self._generate_visualizations(
                    report_type, report_data
                )

            return report_data

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    async def _generate_individual_report(self, assessment_id: str) -> Dict[str, Any]:
        """Generate individual assessment report"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get assessment data
            cursor.execute("SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,))
            assessment_row = cursor.fetchone()

            if not assessment_row:
                raise ValueError(f"Assessment {assessment_id} not found")

            # Get expert feedback
            cursor.execute(
                """
                SELECT * FROM expert_feedback WHERE assessment_id = ?
                ORDER BY timestamp DESC
            """,
                (assessment_id,),
            )
            feedback_rows = cursor.fetchall()

            assessment_data = json.loads(assessment_row[7])  # assessment_data column

            return {
                "assessment": assessment_data,
                "expert_feedback": [
                    dict(zip([col[0] for col in cursor.description], row)) for row in feedback_rows
                ],
                "summary": {
                    "overall_accuracy": assessment_row[2],
                    "confidence_score": assessment_row[3],
                    "expert_validation_needed": bool(assessment_row[6]),
                    "feedback_count": len(feedback_rows),
                },
            }

    def _format_trend_report(self, analysis: TrendAnalysis) -> Dict[str, Any]:
        """Format trend analysis for report"""
        return {
            "analysis_id": analysis.analysis_id,
            "time_period": {
                "start": analysis.time_period[0].isoformat(),
                "end": analysis.time_period[1].isoformat(),
                "days": (analysis.time_period[1] - analysis.time_period[0]).days,
            },
            "summary": {
                "total_assessments": analysis.total_assessments,
                "average_confidence": round(analysis.average_confidence, 3),
                "safety_incidents": analysis.safety_incidents,
                "expert_agreement_rate": round(analysis.expert_agreement_rate, 3),
            },
            "accuracy_distribution": {
                k.value: v for k, v in analysis.accuracy_distribution.items()
            },
            "common_issues": analysis.common_issues,
            "improvement_areas": analysis.improvement_areas,
            "recommendations": self._generate_trend_recommendations(analysis),
        }

    def _generate_trend_recommendations(self, analysis: TrendAnalysis) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []

        # Low average confidence
        if analysis.average_confidence < 0.7:
            recommendations.append("Implement additional training to improve diagnostic confidence")

        # High safety incidents
        if analysis.safety_incidents > analysis.total_assessments * 0.1:
            recommendations.append("Review and strengthen safety assessment protocols")

        # Low expert agreement
        if analysis.expert_agreement_rate < 0.8:
            recommendations.append("Enhance validation algorithms and expert feedback integration")

        # Poor accuracy distribution
        poor_accuracy_count = sum(
            count
            for level, count in analysis.accuracy_distribution.items()
            if level in [ClinicalAccuracyLevel.CONCERNING, ClinicalAccuracyLevel.DANGEROUS]
        )
        if poor_accuracy_count > analysis.total_assessments * 0.05:
            recommendations.append("Comprehensive review of assessment methodology required")

        return recommendations

    async def _generate_expert_feedback_report(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expert feedback summary report"""
        days = kwargs.get("days", 30)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT ef.*, a.overall_accuracy, a.confidence_score
                FROM expert_feedback ef
                JOIN assessments a ON ef.assessment_id = a.assessment_id
                WHERE ef.timestamp BETWEEN ? AND ?
                ORDER BY ef.timestamp DESC
            """,
                conn,
                params=(start_date, end_date),
            )

            if df.empty:
                return {"message": "No expert feedback in specified period"}

            return {
                "summary": {
                    "total_feedback": len(df),
                    "average_expert_rating": round(df["accuracy_rating"].mean(), 3),
                    "feedback_by_priority": df["priority"].value_counts().to_dict(),
                    "feedback_by_status": df["status"].value_counts().to_dict(),
                },
                "expert_statistics": {
                    "unique_experts": df["expert_id"].nunique(),
                    "most_active_expert": (
                        df["expert_id"].value_counts().index[0] if not df.empty else None
                    ),
                    "average_rating_by_expert": df.groupby("expert_id")["accuracy_rating"]
                    .mean()
                    .to_dict(),
                },
                "correlation_analysis": {
                    "confidence_vs_expert_rating": (
                        round(df["confidence_score"].corr(df["accuracy_rating"]), 3)
                        if len(df) > 1
                        else 0.0
                    )
                },
            }

    async def _generate_improvement_report(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate improvement recommendations report"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM improvement_recommendations
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                conn,
                params=(kwargs.get("limit", 50),),
            )

            if df.empty:
                return {"message": "No improvement recommendations found"}

            return {
                "summary": {
                    "total_recommendations": len(df),
                    "by_category": df["category"].value_counts().to_dict(),
                    "by_priority": df["priority"].value_counts().to_dict(),
                    "by_status": df["status"].value_counts().to_dict(),
                },
                "active_recommendations": df[
                    df["status"].isin(["proposed", "approved", "in_progress"])
                ].to_dict("records"),
                "completed_recommendations": len(df[df["status"] == "completed"]),
                "implementation_timeline": self._analyze_implementation_timeline(df),
            }

    def _analyze_implementation_timeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze implementation timeline for recommendations"""
        timeline_analysis = {}

        # Group by timeline
        timeline_groups = df.groupby("timeline").size().to_dict()
        timeline_analysis["by_timeline"] = timeline_groups

        # Urgent recommendations (high/critical priority)
        urgent = df[df["priority"].isin(["high", "critical"])]
        timeline_analysis["urgent_count"] = len(urgent)

        return timeline_analysis

    async def _generate_safety_alerts_report(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate safety alerts report"""
        days = kwargs.get("days", 7)  # Default to last week for safety alerts
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM assessments 
                WHERE timestamp BETWEEN ? AND ?
                AND safety_risk_level IN ('moderate', 'high', 'critical')
                ORDER BY timestamp DESC
            """,
                conn,
                params=(start_date, end_date),
            )

            return {
                "summary": {
                    "total_safety_alerts": len(df),
                    "by_risk_level": df["safety_risk_level"].value_counts().to_dict(),
                    "critical_alerts": len(df[df["safety_risk_level"] == "critical"]),
                    "high_risk_alerts": len(df[df["safety_risk_level"] == "high"]),
                },
                "recent_alerts": df.head(10).to_dict("records"),
                "trends": {"daily_counts": df.groupby(df["timestamp"].dt.date).size().to_dict()},
            }

    async def _generate_comparative_report(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis report"""
        # Compare different time periods, modalities, or other dimensions
        comparison_type = kwargs.get("comparison_type", "time_periods")

        if comparison_type == "time_periods":
            return await self._compare_time_periods(kwargs)
        elif comparison_type == "therapeutic_modalities":
            return await self._compare_therapeutic_modalities(kwargs)
        else:
            return {"error": f"Unsupported comparison type: {comparison_type}"}

    async def _compare_time_periods(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Compare clinical accuracy across time periods"""
        period1_days = kwargs.get("period1_days", 30)
        period2_days = kwargs.get("period2_days", 30)

        # Period 1: Recent
        end_date = datetime.now()
        period1_start = end_date - timedelta(days=period1_days)

        # Period 2: Previous
        period2_end = period1_start
        period2_start = period2_end - timedelta(days=period2_days)

        with sqlite3.connect(self.db_path) as conn:
            # Get data for both periods
            df1 = pd.read_sql_query(
                """
                SELECT * FROM assessments 
                WHERE timestamp BETWEEN ? AND ?
            """,
                conn,
                params=(period1_start, end_date),
            )

            df2 = pd.read_sql_query(
                """
                SELECT * FROM assessments 
                WHERE timestamp BETWEEN ? AND ?
            """,
                conn,
                params=(period2_start, period2_end),
            )

            return {
                "period1": {
                    "timeframe": f"{period1_start.date()} to {end_date.date()}",
                    "total_assessments": len(df1),
                    "average_confidence": (
                        round(df1["confidence_score"].mean(), 3) if not df1.empty else 0
                    ),
                    "accuracy_distribution": df1["overall_accuracy"].value_counts().to_dict(),
                },
                "period2": {
                    "timeframe": f"{period2_start.date()} to {period2_end.date()}",
                    "total_assessments": len(df2),
                    "average_confidence": (
                        round(df2["confidence_score"].mean(), 3) if not df2.empty else 0
                    ),
                    "accuracy_distribution": df2["overall_accuracy"].value_counts().to_dict(),
                },
                "comparison": {
                    "assessment_change": len(df1) - len(df2),
                    "confidence_change": (
                        round((df1["confidence_score"].mean() - df2["confidence_score"].mean()), 3)
                        if not df1.empty and not df2.empty
                        else 0
                    ),
                    "improvement_trend": (
                        "improving"
                        if (df1["confidence_score"].mean() > df2["confidence_score"].mean())
                        else "declining" if not df1.empty and not df2.empty else "insufficient_data"
                    ),
                },
            }

    async def _compare_therapeutic_modalities(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Compare clinical accuracy across therapeutic modalities"""
        days = kwargs.get("days", 30)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM assessments 
                WHERE timestamp BETWEEN ? AND ?
            """,
                conn,
                params=(start_date, end_date),
            )

            if df.empty:
                return {"message": "No data available for comparison"}

            modality_stats = {}
            for modality in df["therapeutic_modality"].unique():
                modality_df = df[df["therapeutic_modality"] == modality]
                modality_stats[modality] = {
                    "count": len(modality_df),
                    "average_confidence": round(modality_df["confidence_score"].mean(), 3),
                    "accuracy_distribution": modality_df["overall_accuracy"]
                    .value_counts()
                    .to_dict(),
                    "safety_incidents": len(
                        modality_df[modality_df["safety_risk_level"].isin(["high", "critical"])]
                    ),
                }

            return {
                "modality_comparison": modality_stats,
                "best_performing": max(
                    modality_stats.keys(), key=lambda x: modality_stats[x]["average_confidence"]
                ),
                "most_used": max(modality_stats.keys(), key=lambda x: modality_stats[x]["count"]),
            }

    async def _generate_visualizations(
        self, report_type: ReportType, report_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate visualizations for reports"""
        visualizations = {}

        try:
            if report_type == ReportType.TREND_ANALYSIS and "accuracy_distribution" in report_data:
                # Accuracy distribution pie chart
                plt.figure(figsize=(10, 6))
                accuracy_data = report_data["accuracy_distribution"]
                plt.pie(accuracy_data.values(), labels=accuracy_data.keys(), autopct="%1.1f%%")
                plt.title("Clinical Accuracy Distribution")

                buffer = BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight")
                buffer.seek(0)
                visualizations["accuracy_distribution"] = base64.b64encode(
                    buffer.getvalue()
                ).decode()
                plt.close()

            # Add more visualization types as needed

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return visualizations

    async def export_report(
        self, report_data: Dict[str, Any], format: str = "json", output_path: Optional[Path] = None
    ) -> Path:
        """Export report in specified format"""
        if not output_path:
            output_path = Path(f"reports/{report_data['report_id']}.{format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "html":
            html_content = self._generate_html_report(report_data)
            with open(output_path, "w") as f:
                f.write(html_content)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Report exported to {output_path}")
        return output_path

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Accuracy Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9f4ff; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical Accuracy Report</h1>
                <p><strong>Report ID:</strong> {report_id}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Type:</strong> {report_type}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                {summary_content}
            </div>
            
            <div class="section">
                <h2>Details</h2>
                <pre>{details}</pre>
            </div>
        </body>
        </html>
        """

        summary_content = ""
        if "summary" in report_data:
            for key, value in report_data["summary"].items():
                summary_content += f'<div class="metric"><strong>{key}:</strong> {value}</div>'

        return html_template.format(
            report_id=report_data.get("report_id", "Unknown"),
            timestamp=report_data.get("timestamp", "Unknown"),
            report_type=report_data.get("report_type", "Unknown"),
            summary_content=summary_content,
            details=json.dumps(
                {
                    k: v
                    for k, v in report_data.items()
                    if k not in ["report_id", "timestamp", "report_type", "summary"]
                },
                indent=2,
                default=str,
            ),
        )

    async def get_feedback_loop_status(self) -> Dict[str, Any]:
        """Get current status of the feedback loop system"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get recent statistics
            cursor.execute(
                """
                SELECT COUNT(*) FROM assessments 
                WHERE timestamp > datetime('now', '-7 days')
            """
            )
            recent_assessments = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM expert_feedback 
                WHERE timestamp > datetime('now', '-7 days')
            """
            )
            recent_feedback = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*) FROM improvement_recommendations 
                WHERE status IN ('proposed', 'approved', 'in_progress')
            """
            )
            active_recommendations = cursor.fetchone()[0]

            return {
                "system_status": "active",
                "recent_activity": {
                    "assessments_last_7_days": recent_assessments,
                    "expert_feedback_last_7_days": recent_feedback,
                    "active_recommendations": active_recommendations,
                },
                "configuration": self.config,
                "last_updated": datetime.now().isoformat(),
            }
