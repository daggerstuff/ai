"""PATIENT-Ψ FastAPI router.

Exposes the simulation engine as REST endpoints at /api/v1/patient-psi.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Query, status
from pydantic import BaseModel

from ai.platform.patient_psi.engine import (
    PatientPsiEngine,
    SessionNotActiveError,
    SessionNotFoundError,
    SimulationConfig,
    SimulationStatus,
    SimulationTurn,
)
from ai.platform.patient_psi.profiles import ProfileRegistry

# ── Module-level engine singleton ─────────────────────────────────────

_profile_registry = ProfileRegistry()

# Error message constants (avoid leaking internal exception details).
PROFILE_NOT_FOUND_MSG = "Profile not found"
SESSION_NOT_FOUND_MSG = "Session not found"
SESSION_NOT_ACTIVE_MSG = "Session is not active"

_engine = PatientPsiEngine(profile_registry=_profile_registry)

router = APIRouter(tags=["patient-psi"])


def create_app(prefix: str = "") -> FastAPI:
    """Build a FastAPI application with the PATIENT-Ψ router mounted.

    Parameters
    ----------
    prefix:
        Optional URL prefix. Pass ``/api/v1/patient-psi`` for standalone
        use; leave empty when mounting under a parent ASGI app that already
        strips the prefix (e.g. Starlette ``app.mount``).
    """
    app = FastAPI(title="PATIENT-Ψ Simulation Engine")
    app.include_router(router, prefix=prefix)
    return app


# ── Request / Response Schemas ────────────────────────────────────────


class CreateSessionRequest(SimulationConfig):
    """POST body for creating a new simulation session."""


class SessionResponse(BaseModel):
    session_id: str
    profile_name: str
    status: SimulationStatus
    phase: str
    turn_count: int
    created_at: datetime
    updated_at: datetime


class InteractRequest(BaseModel):
    message: str
    seed: int | None = None


class InteractResponse(BaseModel):
    session_id: str
    turn: SimulationTurn


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(request: CreateSessionRequest) -> SessionResponse:
    """Create a new PATIENT-Ψ simulation session."""
    try:
        session_id = _engine.create_session(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=PROFILE_NOT_FOUND_MSG) from exc

    session = _engine.get_session(session_id)
    assert session is not None

    return SessionResponse(
        session_id=session.session_id,
        profile_name=session.config.profile_name,
        status=SimulationStatus.ACTIVE,
        phase=session.state_machine.state.phase.value,
        turn_count=session.state_machine.state.turn_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.post("/sessions/{session_id}/interact", response_model=InteractResponse)
def interact(session_id: str, request: InteractRequest) -> InteractResponse:
    """Process a therapist utterance and return the patient response."""
    try:
        turn = _engine.interact(session_id, request.message, seed=request.seed)
    except SessionNotActiveError as exc:
        raise HTTPException(status_code=400, detail=SESSION_NOT_ACTIVE_MSG) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG) from exc

    return InteractResponse(session_id=session_id, turn=turn)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    """Get simulation session status."""
    session = _engine.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)

    return SessionResponse(
        session_id=session.session_id,
        profile_name=session.config.profile_name,
        status=session.status,
        phase=session.state_machine.state.phase.value,
        turn_count=session.state_machine.state.turn_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.post("/sessions/{session_id}/terminate", response_model=SessionResponse)
def terminate_session(session_id: str) -> SessionResponse:
    """Terminate a simulation session."""
    if not _engine.terminate_session(session_id):
        raise HTTPException(status_code=404, detail=SESSION_NOT_FOUND_MSG)

    session = _engine.get_session(session_id)
    assert session is not None

    return SessionResponse(
        session_id=session.session_id,
        profile_name=session.config.profile_name,
        status=SimulationStatus.TERMINATED,
        phase=session.state_machine.state.phase.value,
        turn_count=session.state_machine.state.turn_count,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


@router.get("/sessions", response_model=SessionListResponse)
def list_active_sessions(
    profile: Annotated[str | None, Query(description="Optional profile name filter")] = None,
) -> SessionListResponse:
    """List all active simulation sessions."""
    active = _engine.list_active_sessions()
    if profile:
        active = [s for s in active if s.config.profile_name == profile]
    return SessionListResponse(
        sessions=[
            SessionResponse(
                session_id=s.session_id,
                profile_name=s.config.profile_name,
                status=s.status,
                phase=s.state.phase.value,
                turn_count=s.turn_count,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in active
        ]
    )


@router.get("/profiles", response_model=list[str])
def list_profiles() -> list[str]:
    """List available patient profile names."""
    return list(_profile_registry.list_profiles())
