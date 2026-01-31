"""
Stage configuration for the four-stage Pixelated Empathy training ladder.

This mirrors the MasterTrainingPlan.md and the expanded project brief:
- Stage 1 – Foundation & Rapport (~40%)
- Stage 2 – Therapeutic Expertise & Reasoning (~25%)
- Stage 3 – Edge Stress Test & Scenario Bank (~20%)
- Stage 4 – Voice, Persona & Delivery (~15%)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class StageConfig:
    """Configuration for a single training stage."""

    id: str
    name: str
    target_share: float  # fraction in [0, 1]
    description: str


STAGE1_ID = "stage1_foundation"
STAGE2_ID = "stage2_therapeutic_expertise"
STAGE3_ID = "stage3_edge_stress_test"
STAGE4_ID = "stage4_voice_persona"


STAGES: Dict[str, StageConfig] = {
    STAGE1_ID: StageConfig(
        id=STAGE1_ID,
        name="Stage 1 – Foundation & Rapport",
        target_share=0.40,
        description=(
            "Baseline therapeutic tone, reflective listening, and low‑risk support "
            "dialogs drawn from consolidated foundation corpora."
        ),
    ),
    STAGE2_ID: StageConfig(
        id=STAGE2_ID,
        name="Stage 2 – Therapeutic Expertise & Reasoning",
        target_share=0.25,
        description=(
            "Structured reasoning, diagnosis scaffolding, knowledge grounding, and "
            "summarization style examples."
        ),
    ),
    STAGE3_ID: StageConfig(
        id=STAGE3_ID,
        name="Stage 3 – Edge Stress Test & Scenario Bank",
        target_share=0.20,
        description=(
            "Nightmare edge cases, crisis scenarios, Reddit/raw edge pipelines, and "
            "high‑intensity stress tests."
        ),
    ),
    STAGE4_ID: StageConfig(
        id=STAGE4_ID,
        name="Stage 4 – Voice, Persona & Delivery",
        target_share=0.15,
        description=(
            "Tim‑Fletcher‑style voice, persona consistency, dual‑persona scripts, "
            "and delivery-focused training."
        ),
    ),
}


def get_stage_config(stage_id: str) -> StageConfig:
    """Return the StageConfig for a given ID, raising KeyError if unknown."""

    return STAGES[stage_id]


def get_all_stages() -> List[StageConfig]:
    """Return all stage configs in a stable order (1 → 4)."""

    return [
        STAGES[STAGE1_ID],
        STAGES[STAGE2_ID],
        STAGES[STAGE3_ID],
        STAGES[STAGE4_ID],
    ]


def total_target_share() -> float:
    """Return the sum of all target shares (should be ~= 1.0)."""

    return sum(stage.target_share for stage in get_all_stages())


__all__ = [
    "StageConfig",
    "STAGE1_ID",
    "STAGE2_ID",
    "STAGE3_ID",
    "STAGE4_ID",
    "STAGES",
    "get_stage_config",
    "get_all_stages",
    "total_target_share",
]


