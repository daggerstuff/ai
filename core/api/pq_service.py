"""
Empathy PQ Metrics API Service

Exposes the EmpathyPQCalculator as a FastAPI service.
Each session gets its own isolated calculator instance.
"""

import logging
from typing import Dict, Optional

from ai.core.empathy_pq import EmpathyPQCalculator, PQScore
from ai.core.gestalt_engine import CrisisLevel, GestaltState
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pq", tags=["empathy-pq"])

# In-memory session store (keyed by session_id)
# Production would use Redis for distributed sessions
_pq_sessions: Dict[str, EmpathyPQCalculator] = {}


class PQUpdateRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    defense_maturity: Optional[float] = Field(
        None, description="Current defense maturity (0.0 to higher scores)"
    )
    defense_label: int = Field(0, description="Defense mechanism label")
    defense_label_name: str = Field("", description="Defense mechanism name")
    crisis_level: str = Field("NONE", description="CrisisLevel enum value")
    persona_directive: str = Field("", description="Persona directive from Gestalt")
    breakthrough_score: float = Field(0.0, description="Breakthrough delta score")


class PQSummaryResponse(BaseModel):
    final_pq: float
    start_maturity: Optional[float]
    end_maturity: Optional[float]
    net_growth: float
    performance_category: str


def _get_or_create_session(session_id: str) -> EmpathyPQCalculator:
    if session_id not in _pq_sessions:
        _pq_sessions[session_id] = EmpathyPQCalculator()
        logger.info(f"Created new PQ session: {session_id}")
    return _pq_sessions[session_id]


@router.post(
    "/update",
    response_model=PQScore,
    summary="Update PQ score for a training session turn",
)
async def update_pq(request: PQUpdateRequest) -> PQScore:
    """Record a patient defense state and return the updated PQ score."""
    calculator = _get_or_create_session(request.session_id)

    # Build minimal GestaltState from the request fields
    try:
        crisis = CrisisLevel[request.crisis_level.upper()]
    except KeyError:
        crisis = CrisisLevel.NONE

    state = GestaltState(
        defense_label=request.defense_label,
        defense_label_name=request.defense_label_name,
        defense_confidence=1.0,
        defense_maturity=request.defense_maturity,
        defense_probabilities={},
        plutchik_scores={},
        dominant_emotion="",
        dominant_emotion_intensity=0.0,
        ocean_scores={},
        crisis_level=crisis,
        behavioral_prediction="",
        persona_directive=request.persona_directive,
        breakthrough_score=request.breakthrough_score,
    )

    return calculator.calculate_pq_increment(state)


@router.get(
    "/summary/{session_id}",
    response_model=PQSummaryResponse,
    summary="Get session performance summary",
)
async def get_session_summary(session_id: str) -> PQSummaryResponse:
    """Get the final performance summary for a completed training session."""
    if session_id not in _pq_sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    calculator = _pq_sessions[session_id]
    summary = calculator.get_session_summary()

    if summary.get("status") == "no_data":
        raise HTTPException(status_code=422, detail="No data recorded for this session")

    return PQSummaryResponse(
        final_pq=summary["final_pq"],
        start_maturity=summary.get("start_maturity"),
        end_maturity=summary.get("end_maturity"),
        net_growth=summary.get("net_growth", 0.0),
        performance_category=summary["performance_category"],
    )


@router.delete(
    "/session/{session_id}",
    summary="Reset a PQ session",
)
async def reset_pq_session(session_id: str) -> dict:
    """Clear the PQ session state (call at start of a new training session)."""
    if session_id in _pq_sessions:
        del _pq_sessions[session_id]
    return {"status": "success", "message": f"Session {session_id!r} reset"}
