#!/usr/bin/env python3
"""
Task 6.3: Hierarchical Quality Assessment Framework

This module implements a comprehensive quality assessment framework that evaluates
conversations across the 6-tier ecosystem with tier-specific quality standards
(Tier 1 = Gold Standard → Tier 6 = Reference).

Strategic Goal: Ensure quality consistency across 2.59M+ conversations with
tier-appropriate validation standards and automated quality scoring.
"""

import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Import ecosystem components
from distributed_architecture import DataTier


class QualityDimension(Enum):
    """Quality assessment dimensions."""
    THERAPEUTIC_ACCURACY = "therapeutic_accuracy"
    CONVERSATION_COHERENCE = "conversation_coherence"
    EMOTIONAL_AUTHENTICITY = "emotional_authenticity"
    CLINICAL_COMPLIANCE = "clinical_compliance"
    LANGUAGE_QUALITY = "language_quality"
    SAFETY_COMPLIANCE = "safety_compliance"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    EDUCATIONAL_VALUE = "educational_value"


@dataclass
class QualityMetric:
    """Individual quality metric with scoring details."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence: list[str]
    issues: list[str]
    recommendations: list[str]


@dataclass
class TierQualityStandards:
    """Quality standards for a specific tier."""
    tier: DataTier
    minimum_overall_score: float
    dimension_weights: dict[QualityDimension, float]
    required_dimensions: list[QualityDimension]
    validation_strictness: float  # 0.0 to 1.0
    sample_validation_rate: float  # Percentage of conversations to validate


@dataclass
class QualityAssessmentResult:
    """Complete quality assessment result for a conversation."""
    conversation_id: str
    tier: DataTier
    overall_score: float
    dimension_scores: dict[QualityDimension, QualityMetric]
    meets_tier_standards: bool
    quality_grade: str  # A, B, C, D, F
    validation_notes: list[str]
    assessment_timestamp: datetime
    assessor_version: str


class TherapeuticAccuracyAssessor:
    """Assesses therapeutic accuracy of conversations."""

    def __init__(self):
        self.therapeutic_techniques = {
            "active_listening": [
                r"\b(I hear you|I understand|tell me more|go on)\b",
                r"\b(what I\'m hearing|it sounds like|so you\'re saying)\b"
            ],
            "empathic_reflection": [
                r"\b(that must be|I can imagine|it sounds difficult)\b",
                r"\b(you\'re feeling|you seem|it appears)\b"
            ],
            "cognitive_restructuring": [
                r"\b(what evidence|alternative thought|different perspective)\b",
                r"\b(is that thought helpful|realistic thinking)\b"
            ],
            "validation": [
                r"\b(that\'s understandable|makes sense|valid feeling)\b",
                r"\b(anyone would feel|normal to feel)\b"
            ],
            "psychoeducation": [
                r"\b(research shows|studies indicate|it\'s common)\b",
                r"\b(many people experience|typical response)\b"
            ]
        }

        self.problematic_responses = [
            r"\b(just think positive|get over it|it\'s not that bad)\b",
            r"\b(you should|you need to|you have to)\b",
            r"\b(I know exactly how you feel|same thing happened to me)\b"
        ]

    def assess_therapeutic_accuracy(self, conversation: dict[str, Any]) -> QualityMetric:
        """Assess therapeutic accuracy of a conversation."""
        text = self._extract_therapist_text(conversation)

        if not text:
            return QualityMetric(
                dimension=QualityDimension.THERAPEUTIC_ACCURACY,
                score=0.0,
                confidence=1.0,
                evidence=[],
                issues=["No therapist responses found"],
                recommendations=["Ensure conversation includes therapist responses"]
            )

        # Detect therapeutic techniques
        technique_scores = []
        evidence = []

        for technique, patterns in self.therapeutic_techniques.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            if matches:
                technique_scores.append(1.0)
                evidence.append(f"{technique}: {len(matches)} instances")
            else:
                technique_scores.append(0.0)

        # Check for problematic responses
        issues = []
        for pattern in self.problematic_responses:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Problematic response pattern detected: {pattern}")

        # Calculate score
        base_score = statistics.mean(technique_scores) if technique_scores else 0.0

        # Penalize for problematic responses
        penalty = len(issues) * 0.2
        final_score = max(0.0, base_score - penalty)

        # Generate recommendations
        recommendations = []
        if final_score < 0.7:
            recommendations.append("Incorporate more evidence-based therapeutic techniques")
        if issues:
            recommendations.append("Avoid directive or minimizing language")
        if not evidence:
            recommendations.append("Include therapeutic responses that demonstrate active listening and empathy")

        return QualityMetric(
            dimension=QualityDimension.THERAPEUTIC_ACCURACY,
            score=final_score,
            confidence=0.8,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )

    def _extract_therapist_text(self, conversation: dict[str, Any]) -> str:
        """Extract therapist/counselor text from conversation."""
        therapist_text = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict):
                    role = message.get("role", "").lower()
                    if role in ["therapist", "counselor", "assistant", "helper"]:
                        content = message.get("content", "")
                        therapist_text.append(content)

        return " ".join(therapist_text)


class ConversationCoherenceAssessor:
    """Assesses conversation coherence and flow."""

    def assess_coherence(self, conversation: dict[str, Any]) -> QualityMetric:
        """Assess conversation coherence."""
        messages = conversation.get("messages", [])

        if len(messages) < 2:
            return QualityMetric(
                dimension=QualityDimension.CONVERSATION_COHERENCE,
                score=0.0,
                confidence=1.0,
                evidence=[],
                issues=["Insufficient messages for coherence assessment"],
                recommendations=["Ensure conversation has multiple exchanges"]
            )

        coherence_factors = []
        evidence = []
        issues = []

        # Check turn-taking pattern
        roles = [msg.get("role", "unknown") for msg in messages]
        role_changes = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i-1])

        if role_changes > 0:
            turn_taking_score = min(role_changes / (len(messages) - 1), 1.0)
            coherence_factors.append(turn_taking_score)
            evidence.append(f"Turn-taking score: {turn_taking_score:.2f}")
        else:
            issues.append("No role alternation detected")
            coherence_factors.append(0.0)

        # Check message length consistency
        message_lengths = [len(msg.get("content", "").split()) for msg in messages]
        if message_lengths:
            length_variance = statistics.variance(message_lengths)
            avg_length = statistics.mean(message_lengths)

            # Lower variance relative to mean indicates better consistency
            consistency_score = max(0.0, 1.0 - (length_variance / max(avg_length ** 2, 1)))
            coherence_factors.append(consistency_score)
            evidence.append(f"Length consistency score: {consistency_score:.2f}")

        # Check for conversation flow indicators
        flow_indicators = ["because", "so", "therefore", "however", "but", "and then", "also", "additionally"]
        text = " ".join([msg.get("content", "") for msg in messages]).lower()

        flow_count = sum(1 for indicator in flow_indicators if indicator in text)
        flow_score = min(flow_count / 5.0, 1.0)  # Normalize to max 1.0
        coherence_factors.append(flow_score)
        evidence.append(f"Flow indicators: {flow_count}")

        # Calculate overall coherence score
        overall_score = statistics.mean(coherence_factors) if coherence_factors else 0.0

        # Generate recommendations
        recommendations = []
        if overall_score < 0.6:
            recommendations.append("Improve conversation flow and logical progression")
        if turn_taking_score < 0.5:
            recommendations.append("Ensure proper turn-taking between participants")
        if flow_score < 0.3:
            recommendations.append("Add transitional phrases to improve flow")

        return QualityMetric(
            dimension=QualityDimension.CONVERSATION_COHERENCE,
            score=overall_score,
            confidence=0.7,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )


class SafetyComplianceAssessor:
    """Assesses safety and ethical compliance."""

    def __init__(self):
        self.safety_violations = {
            "harmful_advice": [
                r"\b(hurt yourself|end it all|not worth living)\b",
                r"\b(kill yourself|suicide|self-harm)\b"
            ],
            "inappropriate_boundaries": [
                r"\b(meet in person|personal relationship|date)\b",
                r"\b(my phone number|personal contact)\b"
            ],
            "medical_advice": [
                r"\b(stop taking medication|don\'t need therapy)\b",
                r"\b(diagnose you with|you definitely have)\b"
            ],
            "discriminatory_content": [
                r"\b(because of your race|gender makes you)\b",
                r"\b(people like you|your kind)\b"
            ]
        }

        self.safety_indicators = [
            "safety plan", "crisis hotline", "emergency services",
            "professional help", "qualified therapist", "medical professional"
        ]

    def assess_safety_compliance(self, conversation: dict[str, Any]) -> QualityMetric:
        """Assess safety and ethical compliance."""
        text = " ".join([
            msg.get("content", "") for msg in conversation.get("messages", [])
        ]).lower()

        violations = []
        evidence = []
        issues = []

        # Check for safety violations
        for violation_type, patterns in self.safety_violations.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append(violation_type)
                    issues.append(f"Safety violation detected: {violation_type}")

        # Check for safety indicators (positive)
        safety_count = sum(1 for indicator in self.safety_indicators if indicator in text)
        if safety_count > 0:
            evidence.append(f"Safety indicators present: {safety_count}")

        # Calculate safety score
        if violations:
            safety_score = 0.0  # Any violation results in zero safety score
        else:
            # Base score of 0.8, bonus for safety indicators
            safety_score = min(0.8 + (safety_count * 0.05), 1.0)

        # Generate recommendations
        recommendations = []
        if violations:
            recommendations.append("Remove all safety violations and harmful content")
            recommendations.append("Ensure ethical therapeutic boundaries are maintained")
        if safety_count == 0:
            recommendations.append("Consider adding appropriate safety resources when relevant")

        return QualityMetric(
            dimension=QualityDimension.SAFETY_COMPLIANCE,
            score=safety_score,
            confidence=0.9,
            evidence=evidence,
            issues=issues,
            recommendations=recommendations
        )


class HierarchicalQualityAssessmentFramework:
    """Main framework for hierarchical quality assessment across tiers."""

    def __init__(self):
        self.assessors = {
            QualityDimension.THERAPEUTIC_ACCURACY: TherapeuticAccuracyAssessor(),
            QualityDimension.CONVERSATION_COHERENCE: ConversationCoherenceAssessor(),
            QualityDimension.SAFETY_COMPLIANCE: SafetyComplianceAssessor()
        }

        # Define tier-specific quality standards
        self.tier_standards = self._initialize_tier_standards()

        # Assessment history
        self.assessment_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _initialize_tier_standards(self) -> dict[DataTier, TierQualityStandards]:
        """Initialize quality standards for each tier."""
        return {
            DataTier.TIER_1_PRIORITY: TierQualityStandards(
                tier=DataTier.TIER_1_PRIORITY,
                minimum_overall_score=0.85,
                dimension_weights={
                    QualityDimension.THERAPEUTIC_ACCURACY: 0.30,
                    QualityDimension.CONVERSATION_COHERENCE: 0.25,
                    QualityDimension.SAFETY_COMPLIANCE: 0.25,
                    QualityDimension.EMOTIONAL_AUTHENTICITY: 0.20
                },
                required_dimensions=[
                    QualityDimension.THERAPEUTIC_ACCURACY,
                    QualityDimension.SAFETY_COMPLIANCE
                ],
                validation_strictness=0.95,
                sample_validation_rate=1.0  # 100% validation for gold standard
            ),

            DataTier.TIER_2_PROFESSIONAL: TierQualityStandards(
                tier=DataTier.TIER_2_PROFESSIONAL,
                minimum_overall_score=0.75,
                dimension_weights={
                    QualityDimension.THERAPEUTIC_ACCURACY: 0.35,
                    QualityDimension.CONVERSATION_COHERENCE: 0.20,
                    QualityDimension.SAFETY_COMPLIANCE: 0.25,
                    QualityDimension.CLINICAL_COMPLIANCE: 0.20
                },
                required_dimensions=[
                    QualityDimension.THERAPEUTIC_ACCURACY,
                    QualityDimension.SAFETY_COMPLIANCE
                ],
                validation_strictness=0.85,
                sample_validation_rate=0.5  # 50% validation
            ),

            DataTier.TIER_3_COT: TierQualityStandards(
                tier=DataTier.TIER_3_COT,
                minimum_overall_score=0.70,
                dimension_weights={
                    QualityDimension.CONVERSATION_COHERENCE: 0.40,
                    QualityDimension.EDUCATIONAL_VALUE: 0.30,
                    QualityDimension.SAFETY_COMPLIANCE: 0.20,
                    QualityDimension.LANGUAGE_QUALITY: 0.10
                },
                required_dimensions=[
                    QualityDimension.CONVERSATION_COHERENCE,
                    QualityDimension.SAFETY_COMPLIANCE
                ],
                validation_strictness=0.75,
                sample_validation_rate=0.3  # 30% validation
            ),

            DataTier.TIER_4_REDDIT: TierQualityStandards(
                tier=DataTier.TIER_4_REDDIT,
                minimum_overall_score=0.60,
                dimension_weights={
                    QualityDimension.SAFETY_COMPLIANCE: 0.40,
                    QualityDimension.EMOTIONAL_AUTHENTICITY: 0.30,
                    QualityDimension.LANGUAGE_QUALITY: 0.20,
                    QualityDimension.CULTURAL_SENSITIVITY: 0.10
                },
                required_dimensions=[
                    QualityDimension.SAFETY_COMPLIANCE
                ],
                validation_strictness=0.65,
                sample_validation_rate=0.1  # 10% validation
            ),

            DataTier.TIER_5_RESEARCH: TierQualityStandards(
                tier=DataTier.TIER_5_RESEARCH,
                minimum_overall_score=0.65,
                dimension_weights={
                    QualityDimension.EDUCATIONAL_VALUE: 0.35,
                    QualityDimension.LANGUAGE_QUALITY: 0.25,
                    QualityDimension.SAFETY_COMPLIANCE: 0.25,
                    QualityDimension.CONVERSATION_COHERENCE: 0.15
                },
                required_dimensions=[
                    QualityDimension.SAFETY_COMPLIANCE,
                    QualityDimension.EDUCATIONAL_VALUE
                ],
                validation_strictness=0.70,
                sample_validation_rate=0.2  # 20% validation
            ),

            DataTier.TIER_6_KNOWLEDGE: TierQualityStandards(
                tier=DataTier.TIER_6_KNOWLEDGE,
                minimum_overall_score=0.55,
                dimension_weights={
                    QualityDimension.EDUCATIONAL_VALUE: 0.40,
                    QualityDimension.LANGUAGE_QUALITY: 0.30,
                    QualityDimension.SAFETY_COMPLIANCE: 0.30
                },
                required_dimensions=[
                    QualityDimension.SAFETY_COMPLIANCE
                ],
                validation_strictness=0.60,
                sample_validation_rate=0.05  # 5% validation
            )
        }

    def assess_conversation_quality(self, conversation_id: str, conversation: dict[str, Any],
                                  tier: DataTier) -> QualityAssessmentResult:
        """Perform comprehensive quality assessment for a conversation."""

        # Get tier standards
        standards = self.tier_standards[tier]

        # Assess each required dimension
        dimension_scores = {}

        for dimension in standards.dimension_weights:
            if dimension in self.assessors:
                assessor = self.assessors[dimension]

                if dimension == QualityDimension.THERAPEUTIC_ACCURACY:
                    metric = assessor.assess_therapeutic_accuracy(conversation)
                elif dimension == QualityDimension.CONVERSATION_COHERENCE:
                    metric = assessor.assess_coherence(conversation)
                elif dimension == QualityDimension.SAFETY_COMPLIANCE:
                    metric = assessor.assess_safety_compliance(conversation)
                else:
                    # Placeholder for other dimensions
                    metric = self._assess_placeholder_dimension(conversation, dimension)

                dimension_scores[dimension] = metric

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(dimension_scores, standards.dimension_weights)

        # Determine if conversation meets tier standards
        meets_standards = (
            overall_score >= standards.minimum_overall_score and
            all(
                dimension_scores.get(req_dim, QualityMetric(req_dim, 0.0, 0.0, [], [], [])).score >= 0.5
                for req_dim in standards.required_dimensions
            )
        )

        # Assign quality grade
        quality_grade = self._assign_quality_grade(overall_score)

        # Collect validation notes
        validation_notes = []
        for metric in dimension_scores.values():
            validation_notes.extend(metric.issues)
            if metric.score < 0.5:
                validation_notes.append(f"Low score in {metric.dimension.value}: {metric.score:.2f}")

        # Create assessment result
        result = QualityAssessmentResult(
            conversation_id=conversation_id,
            tier=tier,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            meets_tier_standards=meets_standards,
            quality_grade=quality_grade,
            validation_notes=validation_notes,
            assessment_timestamp=datetime.now(),
            assessor_version="1.0.0"
        )

        # Store in history
        self.assessment_history.append(result)

        self.logger.info(f"Assessed {conversation_id}: {overall_score:.3f} (grade: {quality_grade})")

        return result

    def _assess_placeholder_dimension(self, conversation: dict[str, Any],
                                    dimension: QualityDimension) -> QualityMetric:
        """Placeholder assessment for dimensions not yet implemented."""
        return QualityMetric(
            dimension=dimension,
            score=0.7,  # Neutral score
            confidence=0.5,
            evidence=[f"Placeholder assessment for {dimension.value}"],
            issues=[],
            recommendations=[f"Implement full assessment for {dimension.value}"]
        )

    def _calculate_weighted_score(self, dimension_scores: dict[QualityDimension, QualityMetric],
                                weights: dict[QualityDimension, float]) -> float:
        """Calculate weighted overall quality score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, weight in weights.items():
            if dimension in dimension_scores:
                weighted_sum += dimension_scores[dimension].score * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _assign_quality_grade(self, score: float) -> str:
        """Assign letter grade based on quality score."""
        if score >= 0.9:
            return "A"
        if score >= 0.8:
            return "B"
        if score >= 0.7:
            return "C"
        if score >= 0.6:
            return "D"
        return "F"

    def get_tier_quality_summary(self, tier: DataTier) -> dict[str, Any]:
        """Get quality summary for a specific tier."""
        tier_assessments = [a for a in self.assessment_history if a.tier == tier]

        if not tier_assessments:
            return {"message": f"No assessments found for tier {tier.value}"}

        scores = [a.overall_score for a in tier_assessments]
        grades = [a.quality_grade for a in tier_assessments]

        return {
            "tier": tier.value,
            "total_assessments": len(tier_assessments),
            "average_score": statistics.mean(scores),
            "score_distribution": {
                "min": min(scores),
                "max": max(scores),
                "median": statistics.median(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
            },
            "grade_distribution": dict(Counter(grades)),
            "standards_compliance": {
                "meets_standards": len([a for a in tier_assessments if a.meets_tier_standards]),
                "compliance_rate": len([a for a in tier_assessments if a.meets_tier_standards]) / len(tier_assessments)
            },
            "common_issues": self._get_common_issues(tier_assessments)
        }

    def _get_common_issues(self, assessments: list[QualityAssessmentResult]) -> list[str]:
        """Get most common quality issues for a set of assessments."""
        all_issues = []
        for assessment in assessments:
            all_issues.extend(assessment.validation_notes)

        issue_counts = Counter(all_issues)
        return [issue for issue, count in issue_counts.most_common(5)]

    def export_quality_report(self, output_path: str) -> bool:
        """Export comprehensive quality assessment report."""
        try:
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_assessments": len(self.assessment_history),
                    "assessor_version": "1.0.0"
                },
                "tier_summaries": {
                    tier.value: self.get_tier_quality_summary(tier)
                    for tier in DataTier
                },
                "overall_statistics": {
                    "average_score": statistics.mean([a.overall_score for a in self.assessment_history]) if self.assessment_history else 0,
                    "grade_distribution": dict(Counter([a.quality_grade for a in self.assessment_history])),
                    "standards_compliance_rate": len([a for a in self.assessment_history if a.meets_tier_standards]) / max(len(self.assessment_history), 1)
                }
            }

            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Exported quality report to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting quality report: {e}")
            return False


# Example usage
def main():
    """Example usage of the hierarchical quality assessment framework."""

    # Create assessment framework
    framework = HierarchicalQualityAssessmentFramework()

    # Sample conversations for different tiers
    sample_conversations = [
        {
            "id": "priority_conv_1",
            "tier": DataTier.TIER_1_PRIORITY,
            "conversation": {
                "messages": [
                    {"role": "client", "content": "I've been struggling with anxiety and panic attacks lately."},
                    {"role": "therapist", "content": "I hear that you're experiencing anxiety and panic attacks, and I can imagine how distressing that must be. Can you tell me more about when these panic attacks typically occur?"},
                    {"role": "client", "content": "They seem to happen when I'm in crowded places or before important meetings."},
                    {"role": "therapist", "content": "It sounds like social situations might be triggering your anxiety. That's actually quite common. Let's explore some coping strategies that might help you manage these feelings when they arise."}
                ]
            }
        },
        {
            "id": "reddit_conv_1",
            "tier": DataTier.TIER_4_REDDIT,
            "conversation": {
                "messages": [
                    {"role": "user", "content": "feeling really down today"},
                    {"role": "commenter", "content": "sorry to hear that. what's going on?"},
                    {"role": "user", "content": "just work stress and relationship issues"},
                    {"role": "commenter", "content": "that sounds tough. have you considered talking to someone professional?"}
                ]
            }
        }
    ]

    # Assess conversations
    for conv_data in sample_conversations:
        result = framework.assess_conversation_quality(
            conv_data["id"],
            conv_data["conversation"],
            conv_data["tier"]
        )


        for _dimension, _metric in result.dimension_scores.items():
            pass

        if result.validation_notes:
            pass

    # Get tier summaries
    for tier in [DataTier.TIER_1_PRIORITY, DataTier.TIER_4_REDDIT]:
        summary = framework.get_tier_quality_summary(tier)
        if "total_assessments" in summary:
            pass

    # Export report
    framework.export_quality_report("quality_assessment_report.json")


if __name__ == "__main__":
    main()
