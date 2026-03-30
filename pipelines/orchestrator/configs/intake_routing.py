"""
Canonical intake routing for promoted feeder families.

This module centralizes the mapping from feeder/source family names to the
four-stage ladder plus the continuity holdout lane.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai.pipelines.orchestrator.configs.stages import (
    STAGE1_ID,
    STAGE2_ID,
    STAGE3_ID,
    STAGE4_ID,
)

CONTINUITY_HOLDOUT_LANE = "continuity_holdout"


@dataclass(frozen=True)
class IntakeRoute:
    """Canonical route for a feeder family."""

    source_family: str
    target_lane: str
    split_preference: str | None
    reason: str


_CANONICAL_ROUTES: dict[str, IntakeRoute] = {
    "docs_manual": IntakeRoute(
        source_family="docs_manual",
        target_lane=STAGE1_ID,
        split_preference=None,
        reason="Structured manual and educational knowledge corpus.",
    ),
    "psych_book": IntakeRoute(
        source_family="psych_book",
        target_lane=STAGE1_ID,
        split_preference=None,
        reason="Long-form psychology book knowledge source.",
    ),
    "clinical_reference_pdf": IntakeRoute(
        source_family="clinical_reference_pdf",
        target_lane=STAGE1_ID,
        split_preference=None,
        reason="Clinical reference or manual extraction source.",
    ),
    "psychology_knowledge": IntakeRoute(
        source_family="psychology_knowledge",
        target_lane=STAGE1_ID,
        split_preference=None,
        reason="Structured psychology knowledge corpus.",
    ),
    "psych8k": IntakeRoute(
        source_family="psych8k",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Session-style therapeutic conversation corpus.",
    ),
    "therapist_sft_format": IntakeRoute(
        source_family="therapist_sft_format",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Therapist SFT conversation corpus.",
    ),
    "soulchat2": IntakeRoute(
        source_family="soulchat2",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Multi-turn counseling dataset.",
    ),
    "mental_health_counseling_conversations": IntakeRoute(
        source_family="mental_health_counseling_conversations",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Counseling conversation corpus.",
    ),
    "standard_therapeutic": IntakeRoute(
        source_family="standard_therapeutic",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="General therapeutic conversation banks feed Stage 2 expertise.",
    ),
    "journal_clinical": IntakeRoute(
        source_family="journal_clinical",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Clinical or therapeutic dataset admitted via journal acquisition.",
    ),
    "journal_reference": IntakeRoute(
        source_family="journal_reference",
        target_lane=STAGE1_ID,
        split_preference=None,
        reason="Educational or reference material admitted via journal acquisition.",
    ),
    "nightmare_scenarios": IntakeRoute(
        source_family="nightmare_scenarios",
        target_lane=STAGE3_ID,
        split_preference=None,
        reason="Crisis and adversity scenario bank.",
    ),
    "cot_reasoning": IntakeRoute(
        source_family="cot_reasoning",
        target_lane=STAGE3_ID,
        split_preference=None,
        reason="Reasoning and edge-case scaffold corpus.",
    ),
    "edge_case": IntakeRoute(
        source_family="edge_case",
        target_lane=STAGE3_ID,
        split_preference=None,
        reason="High-intensity edge-case dataset.",
    ),
    "youtube_transcript": IntakeRoute(
        source_family="youtube_transcript",
        target_lane=STAGE4_ID,
        split_preference=None,
        reason="Transcript-derived persona and voice corpus.",
    ),
    "voice_persona": IntakeRoute(
        source_family="voice_persona",
        target_lane=STAGE4_ID,
        split_preference=None,
        reason="Voice and persona corpus.",
    ),
    "dual_persona": IntakeRoute(
        source_family="dual_persona",
        target_lane=STAGE4_ID,
        split_preference=None,
        reason="Persona-structured therapeutic dialogues belong in Stage 4.",
    ),
    "tim_fletcher_transcript": IntakeRoute(
        source_family="tim_fletcher_transcript",
        target_lane=STAGE4_ID,
        split_preference=None,
        reason="Tim Fletcher transcript family.",
    ),
    "long_running_therapy": IntakeRoute(
        source_family="long_running_therapy",
        target_lane=CONTINUITY_HOLDOUT_LANE,
        split_preference="test",
        reason="Continuity-heavy therapy sessions reserved for holdout by default.",
    ),
    # Recovered feeder families (training data branches)
    "training_data_v3": IntakeRoute(
        source_family="training_data_v3",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Consolidated training data v3 branch for therapeutic expertise.",
    ),
    "training_data_v2": IntakeRoute(
        source_family="training_data_v2",
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Consolidated training data v2 branch for therapeutic expertise.",
    ),
}

_ALIASES = {
    "xmu_psych_books": "psych_book",
    "docs": "docs_manual",
    "therapist_sft": "therapist_sft_format",
    "therapist-sft-format": "therapist_sft_format",
    "soulchat_2_0": "soulchat2",
    "tier4_voice_persona": "voice_persona",
    "persona_transcript": "youtube_transcript",
    # Training data branches (recovered feeders)
    "training_v3": "training_data_v3",
    "training_v2": "training_data_v2",
    "training-data-v3": "training_data_v3",
    "training-data-v2": "training_data_v2",
}


def normalize_source_family(source_family: str) -> str:
    """Return a canonical source family key."""

    candidate = source_family.strip().lower()
    return _ALIASES.get(candidate, candidate)


def resolve_intake_route(source_family: str) -> IntakeRoute:
    """Resolve the canonical intake route for a source family."""

    canonical = normalize_source_family(source_family)
    if canonical in _CANONICAL_ROUTES:
        return _CANONICAL_ROUTES[canonical]

    return IntakeRoute(
        source_family=canonical,
        target_lane=STAGE2_ID,
        split_preference=None,
        reason="Unknown feeder defaults to therapeutic expertise review lane.",
    )


__all__ = [
    "CONTINUITY_HOLDOUT_LANE",
    "IntakeRoute",
    "normalize_source_family",
    "resolve_intake_route",
]
