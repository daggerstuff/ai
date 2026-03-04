"""
Defense Mechanism Analysis API Endpoint

FastAPI endpoint for real-time defense mechanism classification
of utterances within conversational context.
"""

import logging
from typing import Optional

from ai.training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["defense-mechanisms"])

# Global model reference set by load_defense_model()
_defense_model = None
_defense_tokenizer = None


class DialogueTurn(BaseModel):
    """A single turn in a conversation."""

    speaker: str = Field(description="Speaker role (Seeker, Supporter, etc.)")
    text: str = Field(description="Utterance text content")


class DefenseAnalysisRequest(BaseModel):
    """Request body for defense mechanism analysis."""

    dialogue: list[DialogueTurn] = Field(
        description="Conversation history as a list of turns"
    )
    target_utterance: str = Field(description="The specific utterance to classify")
    max_turns: int = Field(
        default=40,
        description="Maximum dialogue turns to include in context",
    )


class DefenseAnalysisResponse(BaseModel):
    """Response body with defense mechanism classification."""

    label: int = Field(description="Defense level label (0-8)")
    label_name: str = Field(description="Human-readable defense level name")
    confidence: float = Field(description="Prediction confidence (0.0-1.0)")
    maturity_score: Optional[float] = Field(
        default=None,
        description="Defense maturity normalized to 0.0-1.0. "
        "None for Neutral (0) and Needs More Info (8).",
    )
    probabilities: dict[str, float] = Field(
        description="Per-class probability distribution"
    )


def load_defense_model(
    checkpoint_path: str = None,
    device: str = "cpu",
):
    """
    Initialize the NVIDIA NIM remote defense mechanism classifier for API serving.

    Call this once at application startup. The model is stored
    as a module-level singleton.

    Args:
        checkpoint_path: Legacy argument, preserved for compatibility.
        device: Legacy argument, preserved for compatibility.
    """
    global _defense_model, _defense_tokenizer

    from ai.training.defense_mechanisms.model import NIMEmbeddingClassifier

    try:
        model = NIMEmbeddingClassifier()

        _defense_model = model
        _defense_tokenizer = "NIM_REMOTE"  # Stub to pass `is not None` checks

        logger.info(
            "Defense model initialized using %s (Remote NVIDIA NIM API)",
            model.model_name,
        )
    except Exception as e:
        logger.error(f"Failed to initialize NIMDefenseClassifier: {e}")
        _defense_model = None
        _defense_tokenizer = None


@router.post(
    "/defense",
    response_model=DefenseAnalysisResponse,
    summary="Classify psychological defense mechanism",
    description=(
        "Given a multi-turn dialogue and a target utterance, "
        "classify the defense mechanism level (DMRS) of the "
        "target utterance based on its conversational context."
    ),
)
async def analyze_defense(
    request: DefenseAnalysisRequest,
) -> DefenseAnalysisResponse:
    """Classify the defense mechanism for a target utterance."""
    if _defense_model is None or _defense_tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Defense mechanism model not loaded. "
                "Call load_defense_model() at startup."
            ),
        )

    from ai.training.defense_mechanisms.dataset import format_dialogue

    turns = [{"speaker": t.speaker, "text": t.text} for t in request.dialogue]

    formatted = format_dialogue(
        turns=turns,
        target_text=request.target_utterance,
        max_turns=request.max_turns,
    )

    predictions = _defense_model.predict([formatted])
    pred = predictions[0]

    prob_dict = {
        DEFENSE_LABELS.get(i, str(i)): round(p, 4)
        for i, p in enumerate(pred.probabilities)
    }

    return DefenseAnalysisResponse(
        label=pred.label,
        label_name=pred.label_name,
        confidence=round(pred.confidence, 4),
        maturity_score=pred.maturity_score,
        probabilities=prob_dict,
    )


@router.get(
    "/defense/labels",
    summary="List defense mechanism labels",
    description="Return the DMRS defense mechanism label taxonomy.",
)
async def list_defense_labels() -> dict:
    """Return the defense mechanism label taxonomy and maturity mapping."""
    return {
        "labels": DEFENSE_LABELS,
        "maturity_scores": {str(k): v for k, v in DEFENSE_MATURITY.items()},
        "hierarchy": [
            {"level": 0, "name": "Neutral", "maturity": None, "category": "auxiliary"},
            {
                "level": 1,
                "name": "Action Defenses",
                "maturity": 0.0,
                "category": "immature",
            },
            {
                "level": 2,
                "name": "Major Image-Distorting",
                "maturity": 0.14,
                "category": "immature",
            },
            {"level": 3, "name": "Disavowal", "maturity": 0.29, "category": "immature"},
            {
                "level": 4,
                "name": "Minor Image-Distorting",
                "maturity": 0.43,
                "category": "intermediate",
            },
            {
                "level": 5,
                "name": "Neurotic",
                "maturity": 0.57,
                "category": "intermediate",
            },
            {
                "level": 6,
                "name": "Obsessional",
                "maturity": 0.71,
                "category": "mature-adjacent",
            },
            {
                "level": 7,
                "name": "High-Adaptive",
                "maturity": 1.0,
                "category": "mature",
            },
            {
                "level": 8,
                "name": "Needs More Info",
                "maturity": None,
                "category": "auxiliary",
            },
        ],
    }
