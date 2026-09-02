"""Hierarchical quality assessment framework for conversation records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ai.tools.utilities.pipelines.schemas.conversation_schema import Conversation


class QualityTier(Enum):
    """Quality tiers used by dataset release policy."""

    PRIORITY = "priority"
    PROFESSIONAL = "professional"
    CORE = "core"
    BASIC = "basic"
    ARCHIVE = "archive"

    @property
    def tier_name(self) -> str:
        return self.value

    @classmethod
    def from_score(cls, score: float) -> QualityTier:
        if score >= 0.99:
            return cls.PRIORITY
        if score >= 0.90:
            return cls.PROFESSIONAL
        if score >= 0.80:
            return cls.CORE
        if score >= 0.70:
            return cls.BASIC
        return cls.ARCHIVE


class InquiryType(str, Enum):
    """
    Inquiry type taxonomy from the Inquiry-Diagnosis paper (arXiv 2501.09484).

    The paper identifies four mutually exclusive inquiry types that a
    therapist can use, each with distinct diagnostic-utility profiles.

    - CLOSED_ENDED: Verification / fact-checking questions.
    - OPEN_ENDED:   Exploratory / patient-led prompts.
    - GUIDED:       Hypothesis-testing / channeled prompts.
    - REFLECTIVE:   Meta-cognitive / therapeutic reflections.
    """

    CLOSED_ENDED = "closed_ended"
    OPEN_ENDED = "open_ended"
    GUIDED = "guided"
    REFLECTIVE = "reflective"
    UNKNOWN = "unknown"


@dataclass
class QualityMetrics:
    therapeutic_score: float
    safety_score: float
    linguistic_score: float
    metadata_score: float
    overall_score: float


@dataclass
class QualityAssessment:
    conversation_id: str
    metrics: QualityMetrics
    assigned_tier: QualityTier
    quality_strengths: list[str]
    quality_gaps: list[str]


class QualityAssessmentFramework:
    """Rule-based quality assessment for therapeutic conversations."""

    THERAPEUTIC_KEYWORDS = {
        "anxiety",
        "depress",
        "trauma",
        "coping",
        "safety",
        "ground",
        "emotion",
        "therapy",
        "feeling",
        "support",
        "resources",
    }

    CRISIS_KEYWORDS = {
        "suicidal",
        "suicide",
        "hurt myself",
        "kill myself",
        "self-harm",
        "self harm",
    }

    def _normalize_message(self, message: str) -> str:
        return re.sub(r"\s+", " ", (message or "").strip().lower())

    def _extract_content(self, conversation: Conversation | dict[str, Any]) -> list[str]:
        if isinstance(conversation, Conversation):
            return [self._normalize_message(msg.content) for msg in conversation.messages]

        if isinstance(conversation, dict):
            messages = conversation.get("messages", [])
            content = []
            for item in messages:
                if isinstance(item, dict):
                    content.append(self._normalize_message(item.get("content", "")))
                elif hasattr(item, "content"):
                    content.append(self._normalize_message(item.content))
            return content

        raise TypeError("Unsupported conversation type")

    def _assess_therapeutic_relevance(self, content: str) -> float:
        normalized = self._normalize_message(content)
        tokens = set(normalized.split())
        if not tokens:
            return 0.0
        hits = len(tokens & self.THERAPEUTIC_KEYWORDS)
        # Penalize very short / numeric-only payloads heavily.
        if normalized.isdigit() or re.fullmatch(r"[\d\s]+", normalized):
            return 0.0
        return min(1.0, (hits / 3.0) + 0.1)

    def _assess_safety_compliance(self, content: str) -> float:
        normalized = self._normalize_message(content)
        if not normalized:
            return 0.0
        crisis_hits = sum(1 for keyword in self.CRISIS_KEYWORDS if keyword in normalized)
        if crisis_hits == 0:
            return 1.0
        # Down-weight if crisis language is present but no safety context.
        safety_terms = {"help", "support", "crisis", "talk", "emergency", "grounding"}
        if any(term in normalized for term in safety_terms):
            return max(0.45, 1.0 - 0.12 * crisis_hits)
        return max(0.05, 1.0 - 0.25 * crisis_hits)

    def _assess_linguistic_quality(self, content: str) -> float:
        normalized = self._normalize_message(content)
        if not normalized:
            return 0.0
        words = normalized.split()
        if len(words) < 3:
            return 0.15

        unique_ratio = len(set(words)) / max(1, len(words))
        repeated_penalty = 1.0 if len(set(words)) > 5 else 0.35
        length_penalty = 1.0
        if len(normalized) > 1000:
            length_penalty = 0.75
        punctuation_score = min(1.0, (len(re.findall(r"[.!?]", content)) + 1) / 3)

        score = 0.4 * unique_ratio + 0.35 * repeated_penalty + 0.25 * punctuation_score
        return max(0.0, min(1.0, score * length_penalty))

    def _assess_metadata(self, conversation: Conversation | dict[str, Any]) -> float:
        if isinstance(conversation, Conversation):
            score = 1.0
            if not conversation.conversation_id:
                score -= 0.4
            if not conversation.messages:
                score -= 0.6
            return max(0.0, score)
        if isinstance(conversation, dict):
            score = 1.0
            if not conversation.get("conversation_id"):
                score -= 0.4
            if not conversation.get("messages"):
                score -= 0.6
            return max(0.0, score)
        return 0.0

    def assess_conversation(self, conversation: Conversation | dict[str, Any]) -> QualityAssessment:
        content_lines = self._extract_content(conversation)
        if not content_lines:
            raise ValueError("Conversation has no content")

        merged = " ".join(content_lines)
        strengths: list[str] = []
        gaps: list[str] = []

        therapeutic_score = self._assess_therapeutic_relevance(merged)
        if therapeutic_score > 0.4:
            strengths.append("therapeutic relevance detected")
        else:
            gaps.append("limited therapeutic keyword coverage")

        safety_score = self._assess_safety_compliance(merged)
        if safety_score == 1.0:
            strengths.append("safe language patterns")
        elif safety_score < 0.5:
            gaps.append("potential safety risk terms")

        linguistic_score = self._assess_linguistic_quality(merged)
        if linguistic_score > 0.5:
            strengths.append("readable linguistic structure")
        else:
            gaps.append("low linguistic quality")

        metadata_score = self._assess_metadata(conversation)
        if metadata_score >= 0.8:
            strengths.append("valid conversation metadata")
        else:
            gaps.append("metadata incomplete")

        overall = 0.40 * therapeutic_score + 0.30 * safety_score + 0.20 * linguistic_score + 0.10 * metadata_score
        overall = float(max(0.0, min(1.0, round(overall, 4))))

        assigned_tier = QualityTier.from_score(overall)

        if isinstance(conversation, Conversation):
            conv_id = conversation.conversation_id
        else:
            conv_id = str(conversation.get("conversation_id", "unknown"))

        return QualityAssessment(
            conversation_id=conv_id,
            metrics=QualityMetrics(
                therapeutic_score=therapeutic_score,
                safety_score=safety_score,
                linguistic_score=linguistic_score,
                metadata_score=metadata_score,
                overall_score=overall,
            ),
            assigned_tier=assigned_tier,
            quality_strengths=strengths,
            quality_gaps=gaps,
        )


__all__ = ["QualityAssessment", "QualityAssessmentFramework", "QualityMetrics", "QualityTier"]
