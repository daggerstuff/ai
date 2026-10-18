"""
PIX-35: Stage Classifier for Dataset Slicing.

Classifies normalized records into training stages based on content_type,
therapeutic_modality, topic_tags, and source metadata.

Stage definitions (from MasterTrainingPlan.md):
  stage1_foundation — Core psychology knowledge, standard therapeutic conversations
  stage2_therapeutic_expertise — Specialized therapeutic expertise, advanced modalities
  stage3_edge_stress_test — Edge cases, adversarial inputs, stress testing
  stage4_voice_persona — Voice persona, dual persona, personality-specific training
  supplementary — Everything else that doesn't fit above stages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    """Training stage identifiers."""

    STAGE1_FOUNDATION = "stage1_foundation"
    STAGE2_THERAPEUTIC_EXPERTISE = "stage2_therapeutic_expertise"
    STAGE3_EDGE_STRESS_TEST = "stage3_edge_stress_test"
    STAGE4_VOICE_PERSONA = "stage4_voice_persona"
    SUPPLEMENTARY = "supplementary"


# Stage 4 voice/persona indicators
_VOICE_PERSONA_SOURCES: frozenset[str] = frozenset(
    {
        "pixel_voice",
        "voice_persona",
        "dual_persona",
        "persona",
    }
)

_VOICE_PERSONA_TAGS: frozenset[str] = frozenset(
    {
        "voice",
        "persona",
        "dual_persona",
        "personality",
        "character",
        "role_play",
        "character_voice",
        "persona_training",
    }
)

# Stage 3 edge/stress test indicators
_EDGE_CASE_SOURCES: frozenset[str] = frozenset(
    {
        "edge_cases",
        "adversarial",
        "stress_test",
        "jailbreak",
        "red_team",
        "safety_test",
    }
)

_EDGE_CASE_TAGS: frozenset[str] = frozenset(
    {
        "edge_case",
        "adversarial",
        "stress_test",
        "jailbreak",
        "red_team",
        "safety",
        "crisis",
        "boundary_test",
        "ambiguous",
        "contradictory",
        "multi_intent",
    }
)

# Stage 2 therapeutic expertise indicators
_THERAPEUTIC_MODALITIES: frozenset[str] = frozenset(
    {
        "cbt",
        "dbt",
        "psychodynamic",
        "emdr",
        "act",
        "mbct",
        "ipt",
        "sfbt",
        "gottman",
        "somatic",
        "trauma_informed",
        "attachment_based",
    }
)

_THERAPEUTIC_TAGS: frozenset[str] = frozenset(
    {
        "therapy",
        "counseling",
        "psychotherapy",
        "clinical",
        "diagnosis",
        "treatment_plan",
        "intervention",
        "therapeutic_technique",
        "case_study",
        "supervision",
        "advanced_therapy",
        "specialized_treatment",
    }
)

# Stage 1 foundation indicators
_FOUNDATION_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "conversational",
        "instructional",
        "reference",
    }
)

_FOUNDATION_TAGS: frozenset[str] = frozenset(
    {
        "psychology",
        "mental_health",
        "general",
        "education",
        "self_help",
        "wellness",
        "mindfulness",
        "communication",
        "emotional_intelligence",
        "stress_management",
    }
)


@dataclass
class ClassificationResult:
    """Result of classifying a single record."""

    stage: Stage
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class StageCounts:
    """Aggregated stage counts for a dataset."""

    stage1_foundation: int = 0
    stage2_therapeutic_expertise: int = 0
    stage3_edge_stress_test: int = 0
    stage4_voice_persona: int = 0
    supplementary: int = 0

    def increment(self, stage: Stage) -> None:
        attr = stage.value
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    def total(self) -> int:
        return (
            self.stage1_foundation
            + self.stage2_therapeutic_expertise
            + self.stage3_edge_stress_test
            + self.stage4_voice_persona
            + self.supplementary
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "stage1_foundation": self.stage1_foundation,
            "stage2_therapeutic_expertise": self.stage2_therapeutic_expertise,
            "stage3_edge_stress_test": self.stage3_edge_stress_test,
            "stage4_voice_persona": self.stage4_voice_persona,
            "supplementary": self.supplementary,
        }


class StageClassifier:
    """
    Classifies records into training stages based on content analysis.

    Priority order (highest to lowest):
      1. Stage 4 — Voice/persona content
      2. Stage 3 — Edge cases and stress tests
      3. Stage 2 — Therapeutic expertise
      4. Stage 1 — Foundation content
      5. Supplementary — Everything else
    """

    def __init__(
        self,
        stage_targets: dict[str, int] | None = None,
        enforce_targets: bool = False,
    ) -> None:
        """
        Args:
            stage_targets: Target sample counts per stage
                (e.g., {"stage1_foundation": 40}).
            enforce_targets: If True, cap each stage at its target count.
        """
        self.stage_targets = stage_targets or {}
        self.enforce_targets = enforce_targets
        self._stage_counts: dict[str, int] = dict.fromkeys(self.stage_targets, 0)

    def classify(self, record: dict[str, Any]) -> ClassificationResult:
        """
        Classify a single record into a training stage.

        Args:
            record: Normalized JSONL record with content_type, metadata, source fields.

        Returns:
            ClassificationResult with stage, confidence, and reasons.
        """
        source = (record.get("source") or "").lower()
        content_type = (record.get("content_type") or "").lower()
        metadata = record.get("metadata", {}) or {}
        therapeutic_modality = (metadata.get("therapeutic_modality") or "").lower()
        topic_tags = [t.lower() for t in (metadata.get("topic_tags") or [])]

        # Check Stage 4: Voice/Persona
        stage4_result = self._check_stage4(source, topic_tags)
        if stage4_result is not None:
            return stage4_result

        # Check Stage 3: Edge/Stress Test
        stage3_result = self._check_stage3(source, topic_tags, content_type)
        if stage3_result is not None:
            return stage3_result

        # Check Stage 2: Therapeutic Expertise
        stage2_result = self._check_stage2(therapeutic_modality, topic_tags, content_type, source)
        if stage2_result is not None:
            return stage2_result

        # Check Stage 1: Foundation
        stage1_result = self._check_stage1(content_type, topic_tags, source)
        if stage1_result is not None:
            return stage1_result

        # Default: Supplementary
        return ClassificationResult(
            stage=Stage.SUPPLEMENTARY,
            confidence=0.5,
            reasons=["No specific stage indicators matched"],
        )

    def classify_batch(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[tuple[dict[str, Any], ClassificationResult]], StageCounts]:
        """
        Classify a batch of records and return stage counts.

        Args:
            records: List of normalized JSONL records.

        Returns:
            Tuple of (classified_records, stage_counts).
        """
        classified: list[tuple[dict[str, Any], ClassificationResult]] = []
        counts = StageCounts()

        for record in records:
            result = self.classify(record)

            # Enforce target caps if enabled
            if self.enforce_targets and result.stage.value in self.stage_targets:
                target = self.stage_targets[result.stage.value]
                current = self._stage_counts.get(result.stage.value, 0)
                if current >= target:
                    # Cap reached — demote to supplementary
                    result = ClassificationResult(
                        stage=Stage.SUPPLEMENTARY,
                        confidence=0.3,
                        reasons=[
                            f"Stage {result.stage.value} target reached ({current}/{target})",
                        ],
                    )
                else:
                    self._stage_counts[result.stage.value] = current + 1

            counts.increment(result.stage)
            classified.append((record, result))

        return classified, counts

    # ------------------------------------------------------------------
    # Stage-specific checkers
    # ------------------------------------------------------------------

    def _check_stage4(self, source: str, topic_tags: list[str]) -> ClassificationResult | None:
        """Check for voice/persona content (Stage 4)."""
        reasons: list[str] = []

        if any(src in source for src in _VOICE_PERSONA_SOURCES):
            reasons.append(f"Source matches voice/persona: {source}")

        matching_tags = [t for t in topic_tags if t in _VOICE_PERSONA_TAGS]
        if matching_tags:
            reasons.append(f"Topic tags match voice/persona: {matching_tags}")

        if reasons:
            confidence = 0.95 if len(reasons) > 1 else 0.85
            return ClassificationResult(
                stage=Stage.STAGE4_VOICE_PERSONA,
                confidence=confidence,
                reasons=reasons,
            )
        return None

    def _check_stage3(self, source: str, topic_tags: list[str], content_type: str) -> ClassificationResult | None:
        """Check for edge case/stress test content (Stage 3)."""
        reasons: list[str] = []

        if any(src in source for src in _EDGE_CASE_SOURCES):
            reasons.append(f"Source matches edge case: {source}")

        matching_tags = [t for t in topic_tags if t in _EDGE_CASE_TAGS]
        if matching_tags:
            reasons.append(f"Topic tags match edge case: {matching_tags}")

        if "stress_test" in content_type or "adversarial" in content_type:
            reasons.append(f"Content type indicates edge case: {content_type}")

        if reasons:
            confidence = 0.95 if len(reasons) > 1 else 0.85
            return ClassificationResult(
                stage=Stage.STAGE3_EDGE_STRESS_TEST,
                confidence=confidence,
                reasons=reasons,
            )
        return None

    def _check_stage2(
        self,
        therapeutic_modality: str,
        topic_tags: list[str],
        _content_type: str,
        _source: str,
    ) -> ClassificationResult | None:
        """Check for therapeutic expertise content (Stage 2)."""
        reasons: list[str] = []

        if therapeutic_modality and therapeutic_modality in _THERAPEUTIC_MODALITIES:
            reasons.append(f"Therapeutic modality: {therapeutic_modality}")

        matching_tags = [t for t in topic_tags if t in _THERAPEUTIC_TAGS]
        if matching_tags:
            reasons.append(f"Topic tags match therapeutic: {matching_tags}")

        has_modality_match = therapeutic_modality and therapeutic_modality in _THERAPEUTIC_MODALITIES
        if has_modality_match or len(matching_tags) >= 2:
            confidence = 0.9 if len(reasons) > 1 else 0.75
            return ClassificationResult(
                stage=Stage.STAGE2_THERAPEUTIC_EXPERTISE,
                confidence=confidence,
                reasons=reasons,
            )
        return None

    def _check_stage1(self, content_type: str, topic_tags: list[str], source: str) -> ClassificationResult | None:
        """Check for foundation content (Stage 1)."""
        reasons: list[str] = []

        if content_type in _FOUNDATION_CONTENT_TYPES:
            reasons.append(f"Content type: {content_type}")

        matching_tags = [t for t in topic_tags if t in _FOUNDATION_TAGS]
        if matching_tags:
            reasons.append(f"Topic tags match foundation: {matching_tags}")

        # Academic sources are typically foundation
        academic_sources = {
            "pubmed",
            "zenodo",
            "dryad",
            "core",
            "gutenberg",
            "clinicaltrials",
            "who_iris",
            "openalex",
        }
        if any(src in source for src in academic_sources):
            reasons.append(f"Academic source: {source}")

        if reasons:
            confidence = 0.8 if len(reasons) > 1 else 0.65
            return ClassificationResult(
                stage=Stage.STAGE1_FOUNDATION,
                confidence=confidence,
                reasons=reasons,
            )
        return None


__all__ = [
    "ClassificationResult",
    "Stage",
    "StageClassifier",
    "StageCounts",
]
