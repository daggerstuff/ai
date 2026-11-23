"""Smoke test for the defense-mechanism inference endpoint."""

import asyncio
from pathlib import Path

import pytest

from api.defense_service import (
    DefenseAnalysisRequest,
    DefenseAnalysisResponse,
    DialogueTurn,
    analyze_defense,
    load_defense_model,
)

CHECKPOINT = Path(__file__).resolve().parents[1] / "models/defense_mechanisms/fold_0/best_model.pt"

DIALOGUE = [
    DialogueTurn(speaker="Supporter", text="How are you handling the stress at work?"),
    DialogueTurn(
        speaker="Seeker",
        text=("Oh, it's fine. I just work 80 hours a week and ignore my family. It's the only way to get ahead."),
    ),
    DialogueTurn(speaker="Supporter", text="That sounds like it might be taking a toll on you."),
    DialogueTurn(speaker="Seeker", text="Are you kidding? I'm invincible. Sleep is for the weak anyway."),
]


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason=f"needs a trained defense model checkpoint at {CHECKPOINT}",
)
def test_analyze_defense_returns_classification():
    """The endpoint classifies a target utterance once a checkpoint is loaded."""

    load_defense_model(str(CHECKPOINT))
    request = DefenseAnalysisRequest(
        dialogue=DIALOGUE,
        target_utterance="Are you kidding? I'm invincible. Sleep is for the weak anyway.",
    )

    response = asyncio.run(analyze_defense(request))

    assert isinstance(response, DefenseAnalysisResponse)
    assert 0 <= response.label <= 8
    assert response.label_name
    assert 0.0 <= response.confidence <= 1.0
