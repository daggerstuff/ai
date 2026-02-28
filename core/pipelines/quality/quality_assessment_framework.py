#!/usr/bin/env python3
"""
Hierarchical Quality Assessment Framework for Task 6.3
Multi-tier quality assessment system for therapeutic conversations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Quality assessment tiers."""

    PRIORITY = ("priority", 0.99, "Highest quality therapeutic conversations")
    PROFESSIONAL = ("professional", 0.90, "Professional-grade therapeutic content")
    COT = ("cot", 0.85, "Chain-of-thought reasoning conversations")
    REDDIT = ("reddit", 0.75, "Community-sourced conversations")
    SYNTHETIC = ("synthetic", 0.70, "AI-generated conversations")
    ARCHIVE = ("archive", 0.60, "Archived or legacy conversations")

    def __init__(self, tier_name: str, min_score: float, description: str):
        self.tier_name = tier_name
        self.min_score = min_score
        self.description = description


class QualityDimension(Enum):
    """Quality assessment dimensions."""

    THERAPEUTIC_RELEVANCE = "therapeutic_relevance"
    CONVERSATION_COHERENCE = "conversation_coherence"
    EMOTIONAL_APPROPRIATENESS = "emotional_appropriateness"
    SAFETY_COMPLIANCE = "safety_compliance"
    LINGUISTIC_QUALITY = "linguistic_quality"
    ENGAGEMENT_LEVEL = "engagement_level"
    PROFESSIONAL_STANDARDS = "professional_standards"


@dataclass
class QualityMetrics:
    """Quality metrics for a conversation."""

    therapeutic_relevance: float = 0.0
    conversation_coherence: float = 0.0
    emotional_appropriateness: float = 0.0
    safety_compliance: float = 0.0
    linguistic_quality: float = 0.0
    engagement_level: float = 0.0
    professional_standards: float = 0.0
    overall_score: float = 0.0
    confidence: float = 0.0


@dataclass
class QualityAssessment:
    """Complete quality assessment result."""

    conversation_id: str
    metrics: QualityMetrics
    assigned_tier: QualityTier
    quality_issues: list[str] = field(default_factory=list)
    quality_strengths: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    assessment_timestamp: datetime = field(default_factory=datetime.now)


class QualityAssessmentFramework:
    """
    Hierarchical quality assessment framework for therapeutic conversations.
    """

    def __init__(self):
        """Initialize the quality assessment framework."""
        self.dimension_weights = {
            QualityDimension.THERAPEUTIC_RELEVANCE: 0.25,
            QualityDimension.CONVERSATION_COHERENCE: 0.20,
            QualityDimension.EMOTIONAL_APPROPRIATENESS: 0.15,
            QualityDimension.SAFETY_COMPLIANCE: 0.15,
            QualityDimension.LINGUISTIC_QUALITY: 0.10,
            QualityDimension.ENGAGEMENT_LEVEL: 0.10,
            QualityDimension.PROFESSIONAL_STANDARDS: 0.05,
        }

        self.therapeutic_keywords = self._load_therapeutic_keywords()
        self.safety_patterns = self._load_safety_patterns()
        self.quality_patterns = self._load_quality_patterns()

        logger.info("QualityAssessmentFramework initialized")

    def _load_therapeutic_keywords(self) -> dict[str, list[str]]:
        """Load therapeutic keywords for relevance assessment."""
        return {
            "emotions": [
                "feel",
                "feeling",
                "emotion",
                "emotional",
                "mood",
                "angry",
                "sad",
                "happy",
                "anxious",
                "worried",
                "stressed",
                "depressed",
                "excited",
            ],
            "therapeutic_terms": [
                "therapy",
                "counseling",
                "support",
                "help",
                "understand",
                "cope",
                "manage",
                "process",
                "explore",
                "discuss",
                "share",
                "express",
            ],
            "mental_health": [
                "anxiety",
                "depression",
                "stress",
                "trauma",
                "ptsd",
                "bipolar",
                "ocd",
                "adhd",
                "panic",
                "phobia",
                "addiction",
                "recovery",
            ],
            "therapeutic_techniques": [
                "mindfulness",
                "breathing",
                "relaxation",
                "cognitive",
                "behavioral",
                "exposure",
                "grounding",
                "coping strategies",
                "thought challenging",
            ],
            "relationship_terms": [
                "relationship",
                "family",
                "partner",
                "friend",
                "communication",
                "conflict",
                "boundary",
                "trust",
                "intimacy",
                "support",
            ],
        }

    def _load_safety_patterns(self) -> dict[str, list[str]]:
        """Load safety patterns for compliance assessment."""
        return {
            "crisis_indicators": [
                "suicide",
                "kill myself",
                "end it all",
                "not worth living",
                "hurt myself",
                "self-harm",
                "cutting",
                "overdose",
            ],
            "violence_indicators": [
                "hurt someone",
                "kill them",
                "violence",
                "weapon",
                "gun",
                "knife",
                "attack",
                "harm others",
            ],
            "inappropriate_content": [
                "sexual",
                "explicit",
                "inappropriate",
                "harassment",
                "discrimination",
                "hate speech",
                "offensive",
            ],
            "medical_advice": [
                "diagnose",
                "medication dosage",
                "stop taking",
                "medical advice",
                "prescription",
                "doctor said",
                "medical condition",
            ],
        }

    def _load_quality_patterns(self) -> dict[str, list[str]]:
        """Load quality patterns for linguistic assessment."""
        return {
            "positive_indicators": [
                "I understand",
                "tell me more",
                "how does that make you feel",
                "that sounds difficult",
                "you're not alone",
                "it's okay to feel",
                "let's explore",
                "what do you think",
                "how can I help",
            ],
            "negative_indicators": [
                "you should",
                "just get over it",
                "that's wrong",
                "don't feel",
                "stop thinking",
                "you're overreacting",
                "it's not that bad",
            ],
            "professional_language": [
                "therapeutic",
                "intervention",
                "assessment",
                "treatment",
                "clinical",
                "evidence-based",
                "approach",
                "technique",
            ],
            "engagement_indicators": [
                "?",
                "tell me",
                "share",
                "describe",
                "explain",
                "how",
                "what",
                "when",
                "where",
                "why",
                "feel free",
            ],
        }

    def assess_conversation(self, conversation: dict[str, Any]) -> QualityAssessment:
        """Assess quality of a conversation."""
        conversation_id = (
            conversation.get("conversation_id") or conversation.get("id") or "unknown"
        )

        # Calculate quality metrics
        metrics = self._calculate_quality_metrics(conversation)

        # Determine tier assignment
        assigned_tier = self._assign_quality_tier(metrics)

        # Identify issues and strengths
        issues = self._identify_quality_issues(conversation, metrics)
        strengths = self._identify_quality_strengths(conversation, metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, issues)

        assessment = QualityAssessment(
            conversation_id=conversation_id,
            metrics=metrics,
            assigned_tier=assigned_tier,
            quality_issues=issues,
            quality_strengths=strengths,
            recommendations=recommendations,
        )

        logger.info(
            f"Quality assessment completed for {conversation_id}: {assigned_tier.tier_name} tier"
        )
        return assessment

    def _calculate_quality_metrics(
        self, conversation: dict[str, Any]
    ) -> QualityMetrics:
        """Calculate quality metrics for conversation."""
        content = self._extract_content(conversation)
        turns = self._extract_turns(conversation)

        metrics = QualityMetrics()

        # Therapeutic relevance
        metrics.therapeutic_relevance = self._assess_therapeutic_relevance(content)

        # Conversation coherence
        metrics.conversation_coherence = self._assess_conversation_coherence(turns)

        # Emotional appropriateness
        metrics.emotional_appropriateness = self._assess_emotional_appropriateness(
            content
        )

        # Safety compliance
        metrics.safety_compliance = self._assess_safety_compliance(content)

        # Linguistic quality
        metrics.linguistic_quality = self._assess_linguistic_quality(content)

        # Engagement level
        metrics.engagement_level = self._assess_engagement_level(content, turns)

        # Professional standards
        metrics.professional_standards = self._assess_professional_standards(content)

        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)

        # Calculate confidence
        metrics.confidence = self._calculate_confidence(content, turns)

        return metrics

    def _extract_content(self, conversation: dict[str, Any]) -> str:
        """Extract content from conversation."""
        # Handle messages list (OpenAI format)
        if "messages" in conversation:
            messages = conversation["messages"]
            return " ".join(msg.get("content", "") for msg in messages)

        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)
        elif isinstance(content, dict):
            # Handle structured conversation format
            turns = conversation.get("turns", [])
            content = " ".join(turn.get("content", "") for turn in turns)
        return content

    def _extract_turns(self, conversation: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract turns from conversation."""
        # Handle messages list (OpenAI format)
        if "messages" in conversation:
            return [
                {
                    "content": msg.get("content", ""),
                    "speaker": msg.get("role", "unknown"),
                }
                for msg in conversation["messages"]
            ]

        if "turns" in conversation:
            return conversation["turns"]
        if "content" in conversation and isinstance(conversation["content"], list):
            return [
                {"content": turn, "speaker": "unknown"}
                for turn in conversation["content"]
            ]
        return [{"content": conversation.get("content", ""), "speaker": "unknown"}]

    def _assess_therapeutic_relevance(self, content: str) -> float:
        """Assess therapeutic relevance of content."""
        if not content:
            return 0.0

        content_lower = content.lower()
        total_score = 0.0

        # Check for therapeutic keywords
        for category, keywords in self.therapeutic_keywords.items():
            category_score = sum(1 for keyword in keywords if keyword in content_lower)

            # Weight different categories
            weights = {
                "emotions": 0.3,
                "therapeutic_terms": 0.25,
                "mental_health": 0.2,
                "therapeutic_techniques": 0.15,
                "relationship_terms": 0.1,
            }

            weight = weights.get(category, 0.1)
            # Cap the denominator to avoid penalizing short texts that don't use every keyword
            # We expect maybe 1-3 keywords from a category to show relevance
            denominator = min(len(keywords), 3)
            total_score += min(1.0, category_score / denominator) * weight

        return min(1.0, total_score)

    def _assess_conversation_coherence(self, turns: list[dict[str, Any]]) -> float:
        """Assess conversation coherence."""
        if len(turns) < 2:
            return 0.5  # Neutral score for single turn

        coherence_score = 0.0

        # Check for logical flow between turns
        for i in range(len(turns) - 1):
            current_turn = turns[i].get("content", "").lower()
            next_turn = turns[i + 1].get("content", "").lower()

            # Simple coherence indicators
            if any(
                indicator in next_turn
                for indicator in ["yes", "no", "i understand", "that's", "it feels"]
            ):
                coherence_score += 0.3

            # Check for question-answer patterns
            if "?" in current_turn and len(next_turn) > 5:
                coherence_score += 0.4

            # Check for topic continuity (simplified)
            current_words = set(w for w in current_turn.split() if len(w) > 3)
            next_words = set(w for w in next_turn.split() if len(w) > 3)
            overlap = len(current_words.intersection(next_words))

            if overlap >= 1:
                coherence_score += 0.3

        return min(1.0, coherence_score / (len(turns) - 1))

    def _assess_emotional_appropriateness(self, content: str) -> float:
        """Assess emotional appropriateness of content."""
        if not content:
            return 0.0

        content_lower = content.lower()
        score = 0.7  # Base score

        # Positive emotional indicators
        positive_patterns = [
            "i understand",
            "that sounds difficult",
            "it's okay to feel",
            "you're not alone",
            "i'm here for you",
            "that makes sense",
        ]

        for pattern in positive_patterns:
            if pattern in content_lower:
                score += 0.1

        # Negative emotional indicators
        negative_patterns = [
            "you're wrong",
            "that's stupid",
            "get over it",
            "stop complaining",
            "you're overreacting",
            "that's not important",
        ]

        for pattern in negative_patterns:
            if pattern in content_lower:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def _assess_safety_compliance(self, content: str) -> float:
        """Assess safety compliance of content."""
        if not content:
            return 1.0  # Empty content is safe

        content_lower = content.lower()
        safety_score = 1.0

        # Check for safety violations
        for category, patterns in self.safety_patterns.items():
            violations = sum(1 for pattern in patterns if pattern in content_lower)

            if violations > 0:
                # Different penalties for different violation types
                penalties = {
                    "crisis_indicators": 0.5,
                    "violence_indicators": 0.6,
                    "inappropriate_content": 0.3,
                    "medical_advice": 0.2,
                }

                penalty = penalties.get(category, 0.3)
                safety_score -= penalty * violations

        return max(0.0, safety_score)

    def _assess_linguistic_quality(self, content: str) -> float:
        """Assess linguistic quality of content."""
        if not content:
            return 0.0

        # Basic linguistic quality indicators
        word_count = len(content.split())
        sentence_count = len([s for s in content.split(".") if s.strip()])

        score = 0.5  # Base score

        # Length appropriateness
        if 20 <= word_count <= 500:
            score += 0.2
        elif word_count < 5 or word_count > 1000:
            score -= 0.2

        # Sentence structure
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if 8 <= avg_sentence_length <= 25:
                score += 0.1

        # Grammar and spelling (simplified)
        if content.count("?") > 0:  # Questions indicate engagement
            score += 0.1

        if content.count("!") <= 3:  # Not too many exclamations
            score += 0.05

        # Professional language
        professional_count = sum(
            1
            for pattern in self.quality_patterns["professional_language"]
            if pattern in content.lower()
        )
        score += min(0.15, professional_count * 0.05)

        return min(1.0, max(0.0, score))

    def _assess_engagement_level(
        self, content: str, turns: list[dict[str, Any]]
    ) -> float:
        """Assess engagement level of conversation."""
        if not content:
            return 0.0

        content_lower = content.lower()
        score = 0.0

        # Question indicators
        question_count = content.count("?")
        score += min(0.3, question_count * 0.1)

        # Engagement patterns
        engagement_count = sum(
            1
            for pattern in self.quality_patterns["engagement_indicators"]
            if pattern in content_lower
        )
        score += min(0.4, engagement_count * 0.05)

        # Turn-taking (for multi-turn conversations)
        if len(turns) > 1:
            speakers = {turn.get("speaker", "unknown") for turn in turns}
            if len(speakers) > 1:
                score += 0.3

        return min(1.0, score)

    def _assess_professional_standards(self, content: str) -> float:
        """Assess adherence to professional standards."""
        if not content:
            return 0.5

        content_lower = content.lower()
        score = 0.6  # Base score

        # Professional language indicators
        professional_count = sum(
            1
            for pattern in self.quality_patterns["professional_language"]
            if pattern in content_lower
        )
        score += min(0.2, professional_count * 0.05)

        # Positive professional patterns
        positive_count = sum(
            1
            for pattern in self.quality_patterns["positive_indicators"]
            if pattern in content_lower
        )
        score += min(0.3, positive_count * 0.1)

        # Negative professional patterns
        negative_count = sum(
            1
            for pattern in self.quality_patterns["negative_indicators"]
            if pattern in content_lower
        )
        score -= min(0.4, negative_count * 0.2)

        return max(0.0, min(1.0, score))

    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        weighted_sum = 0.0

        dimension_values = {
            QualityDimension.THERAPEUTIC_RELEVANCE: metrics.therapeutic_relevance,
            QualityDimension.CONVERSATION_COHERENCE: metrics.conversation_coherence,
            QualityDimension.EMOTIONAL_APPROPRIATENESS: metrics.emotional_appropriateness,
            QualityDimension.SAFETY_COMPLIANCE: metrics.safety_compliance,
            QualityDimension.LINGUISTIC_QUALITY: metrics.linguistic_quality,
            QualityDimension.ENGAGEMENT_LEVEL: metrics.engagement_level,
            QualityDimension.PROFESSIONAL_STANDARDS: metrics.professional_standards,
        }

        for dimension, value in dimension_values.items():
            weight = self.dimension_weights[dimension]
            weighted_sum += value * weight

        return weighted_sum

    def _calculate_confidence(self, content: str, turns: list[dict[str, Any]]) -> float:
        """Calculate confidence in quality assessment."""
        confidence = 0.5  # Base confidence

        # Content length factor
        word_count = len(content.split()) if content else 0
        if word_count > 50:
            confidence += 0.2
        elif word_count > 20:
            confidence += 0.1

        # Turn count factor
        if len(turns) > 2:
            confidence += 0.2
        elif len(turns) > 1:
            confidence += 0.1

        # Content richness factor
        if content:
            unique_words = len(set(content.lower().split()))
            if unique_words > 30:
                confidence += 0.1

        return min(1.0, confidence)

    def _assign_quality_tier(self, metrics: QualityMetrics) -> QualityTier:
        """Assign quality tier based on metrics."""
        overall_score = metrics.overall_score

        # Check tiers in order of quality (highest first)
        for tier in [
            QualityTier.PRIORITY,
            QualityTier.PROFESSIONAL,
            QualityTier.COT,
            QualityTier.REDDIT,
            QualityTier.SYNTHETIC,
            QualityTier.ARCHIVE,
        ]:
            if overall_score >= tier.min_score:
                return tier

        return QualityTier.ARCHIVE  # Fallback to lowest tier

    def _identify_quality_issues(
        self, conversation: dict[str, Any], metrics: QualityMetrics
    ) -> list[str]:
        """Identify quality issues in conversation."""
        issues = []

        if metrics.therapeutic_relevance < 0.5:
            issues.append("Low therapeutic relevance - lacks mental health focus")

        if metrics.conversation_coherence < 0.5:
            issues.append("Poor conversation flow - lacks coherence between turns")

        if metrics.emotional_appropriateness < 0.7:
            issues.append("Emotionally inappropriate content detected")

        if metrics.safety_compliance < 0.9:
            issues.append("Safety compliance issues - potential harmful content")

        if metrics.linguistic_quality < 0.6:
            issues.append("Poor linguistic quality - grammar or structure issues")

        if metrics.engagement_level < 0.4:
            issues.append("Low engagement - lacks interactive elements")

        if metrics.professional_standards < 0.6:
            issues.append("Below professional standards - unprofessional language")

        return issues

    def _identify_quality_strengths(
        self, conversation: dict[str, Any], metrics: QualityMetrics
    ) -> list[str]:
        """Identify quality strengths in conversation."""
        strengths = []

        if metrics.therapeutic_relevance > 0.8:
            strengths.append("High therapeutic relevance")

        if metrics.conversation_coherence > 0.8:
            strengths.append("Excellent conversation flow")

        if metrics.emotional_appropriateness > 0.9:
            strengths.append("Emotionally appropriate and supportive")

        if metrics.safety_compliance > 0.95:
            strengths.append("Excellent safety compliance")

        if metrics.linguistic_quality > 0.8:
            strengths.append("High linguistic quality")

        if metrics.engagement_level > 0.7:
            strengths.append("Highly engaging conversation")

        if metrics.professional_standards > 0.8:
            strengths.append("Meets professional standards")

        return strengths

    def _generate_recommendations(
        self, metrics: QualityMetrics, issues: list[str]
    ) -> list[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if metrics.therapeutic_relevance < 0.6:
            recommendations.append(
                "Include more therapeutic language and mental health terminology"
            )

        if metrics.conversation_coherence < 0.6:
            recommendations.append(
                "Improve conversation flow with better turn transitions"
            )

        if metrics.emotional_appropriateness < 0.8:
            recommendations.append("Use more empathetic and supportive language")

        if metrics.safety_compliance < 0.9:
            recommendations.append(
                "Review content for safety violations and harmful language"
            )

        if metrics.engagement_level < 0.5:
            recommendations.append("Add more questions and interactive elements")

        if metrics.professional_standards < 0.7:
            recommendations.append("Use more professional therapeutic language")

        return recommendations

    def assess_batch(
        self, conversations: list[dict[str, Any]]
    ) -> list[QualityAssessment]:
        """Assess quality of multiple conversations."""
        assessments = []

        for conversation in conversations:
            assessment = self.assess_conversation(conversation)
            assessments.append(assessment)

        logger.info(f"Batch assessment completed: {len(assessments)} conversations")
        return assessments

    def get_quality_summary(
        self, assessments: list[QualityAssessment]
    ) -> dict[str, Any]:
        """Get summary of quality assessments."""
        if not assessments:
            return {"status": "empty"}

        # Tier distribution
        tier_counts = {}
        for assessment in assessments:
            tier = assessment.assigned_tier.tier_name
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Average scores
        avg_scores = {
            "overall": sum(a.metrics.overall_score for a in assessments)
            / len(assessments),
            "therapeutic_relevance": sum(
                a.metrics.therapeutic_relevance for a in assessments
            )
            / len(assessments),
            "safety_compliance": sum(a.metrics.safety_compliance for a in assessments)
            / len(assessments),
            "engagement": sum(a.metrics.engagement_level for a in assessments)
            / len(assessments),
        }

        # Common issues
        all_issues = []
        for assessment in assessments:
            all_issues.extend(assessment.quality_issues)

        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]

        return {
            "total_assessments": len(assessments),
            "tier_distribution": tier_counts,
            "average_scores": avg_scores,
            "common_issues": common_issues,
            "high_quality_rate": len(
                [a for a in assessments if a.metrics.overall_score > 0.8]
            )
            / len(assessments),
        }


def main():
    """Test the quality assessment framework."""
    framework = QualityAssessmentFramework()

    # Test conversations
    test_conversations = [
        {
            "id": "conv_1",
            "content": "I'm feeling really anxious about my upcoming presentation. I keep having negative thoughts about failing. Can you help me work through this?",
            "turns": [
                {
                    "speaker": "user",
                    "content": "I'm feeling really anxious about my upcoming presentation.",
                },
                {
                    "speaker": "therapist",
                    "content": "I understand that presentations can be anxiety-provoking. Tell me more about what specifically worries you.",
                },
                {
                    "speaker": "user",
                    "content": "I keep thinking I'll forget what to say and everyone will judge me.",
                },
                {
                    "speaker": "therapist",
                    "content": "Those thoughts sound really distressing. Let's explore some coping strategies to help manage this anxiety.",
                },
            ],
        },
        {
            "id": "conv_2",
            "content": "Just get over it. Everyone gets nervous sometimes. You're overreacting.",
        },
        {
            "id": "conv_3",
            "content": "I understand you're feeling anxious about your presentation. That's a very common experience. Let's work together to develop some strategies that can help you feel more confident and prepared. What aspects of the presentation worry you most?",
        },
    ]

    # Assess individual conversations
    assessments = []
    for conv in test_conversations:
        assessment = framework.assess_conversation(conv)
        assessments.append(assessment)

        if assessment.quality_issues:
            pass

        if assessment.quality_strengths:
            pass

    # Generate summary
    summary = framework.get_quality_summary(assessments)

    if summary["common_issues"]:
        pass


if __name__ == "__main__":
    main()
