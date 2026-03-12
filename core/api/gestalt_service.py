"""
Gestalt Fusion API Service

Exposes the GestaltEngine as a FastAPI service for real-time
psychological signal fusion (Defense + Emotion + Personality).
"""

import logging
import os
from typing import Dict, List, Optional

from ai.core.gestalt_engine import CrisisLevel, GestaltEngine
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["gestalt"])

# Singleton engine instance
_gestalt_engine = GestaltEngine()


class DialogueTurn(BaseModel):
    speaker: str = Field(..., description="Speaker role (Seeker, Supporter, etc.)")
    text: str = Field(..., description="Utterance text content")


class GestaltAnalysisRequest(BaseModel):
    dialogue: List[DialogueTurn] = Field(..., description="Conversation history")
    target_utterance: str = Field(..., description="The utterance to classify")
    plutchik_scores: Dict[str, float] = Field(
        ..., description="Plutchik emotion scores (8 keys)"
    )
    ocean_scores: Dict[str, float] = Field(
        ..., description="OCEAN personality traits (5 keys)"
    )
    max_turns: int = Field(40, description="Context window")


class GestaltAnalysisResponse(BaseModel):
    defense_label: int
    defense_label_name: str
    defense_confidence: float
    defense_maturity: Optional[float]
    defense_probabilities: Dict[str, float]

    plutchik_scores: Dict[str, float]
    dominant_emotion: str
    dominant_emotion_intensity: float

    ocean_scores: Dict[str, float]

    crisis_level: CrisisLevel
    behavioral_prediction: str
    persona_directive: str
    breakthrough_score: float


def load_gestalt_model(checkpoint_path: str, device: str = "cpu"):
    """Load the underlying PsyDefDetect model into the engine."""
    _gestalt_engine.load_defense_model(checkpoint_path, device)


@router.post(
    "/gestalt",
    response_model=GestaltAnalysisResponse,
    summary="Fuse defense, emotion, and personality signals",
)
async def analyze_gestalt(request: GestaltAnalysisRequest) -> GestaltAnalysisResponse:
    """Run the Gestalt Fusion Engine."""
    if not _gestalt_engine.defense_model_loaded:
        # Attempt to load from environment if not loaded
        checkpoint = os.getenv("DEFENSE_MODEL_PATH")
        if checkpoint and os.path.exists(checkpoint):
            _gestalt_engine.load_defense_model(checkpoint)
        else:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Gestalt defense model not loaded and DEFENSE_MODEL_PATH not set."
                ),
            )

    try:
        # Convert Pydantic dialogue to dicts for the engine
        dialogue_dicts = [turn.model_dump() for turn in request.dialogue]

        state = _gestalt_engine.analyze_gestalt(
            dialogue=dialogue_dicts,
            target_utterance=request.target_utterance,
            plutchik_scores=request.plutchik_scores,
            ocean_scores=request.ocean_scores,
            max_turns=request.max_turns,
        )

        return GestaltAnalysisResponse(
            defense_label=state.defense_label,
            defense_label_name=state.defense_label_name,
            defense_confidence=round(state.defense_confidence, 4),
            defense_maturity=state.defense_maturity,
            defense_probabilities=state.defense_probabilities,
            plutchik_scores=state.plutchik_scores,
            dominant_emotion=state.dominant_emotion,
            dominant_emotion_intensity=round(state.dominant_emotion_intensity, 4),
            ocean_scores=state.ocean_scores,
            crisis_level=state.crisis_level,
            behavioral_prediction=state.behavioral_prediction,
            persona_directive=state.persona_directive,
            breakthrough_score=round(state.breakthrough_score, 4),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Gestalt analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail="Internal analysis engine error"
        ) from e


@router.post("/gestalt/reset")
async def reset_gestalt():
    """Reset the engine session (clears previous maturity)."""
    _gestalt_engine.reset_session()
    return {"status": "success", "message": "Gestalt session reset"}
