from __future__ import annotations

from typing import Any

import structlog

from orchestration.celery_app import celery_app
from orchestration.core.inference import InferenceEngine
from orchestration.core.state_machine import PersonaStateMachine
from orchestration.safety.guards import InputGuard, OutputGuard

# The provider factory lives at the pixelated backend boundary.
# At runtime the Celery worker runs with src/ on the Python path.
from pe.core.llm_factory import LLMProviderFactory

logger = structlog.get_logger(__name__)

# Simulated database/store
# In a real app, this would be Redis/Postgres
SESSION_STORE = {}
PERSONA_STORE = {
    "default": {
        "metadata": {"name": "John Doe", "role": "patient", "persona_id": "default"},
        "clinical_profile": {
            "condition": "Chest Pain",
            "symptoms": [{"name": "chest_pain", "severity": 7, "description": "Sharp pain"}],
            "medical_history": "Hypertension",
            "medications": ["Lisinopril"],
        },
        "communication_style": {"tone": "anxious", "vocabulary_level": "layman", "verbosity": "medium"},
        "emotional_state": {"baseline": "scared", "volatility": 6},
    }
}


@celery_app.task
def run_safety_input_guard(session_id: str, user_input: str) -> dict[str, Any]:
    guard = InputGuard()
    result = guard.run(user_input)

    return {
        "session_id": session_id,
        "user_input": user_input,
        "sanitized_input": result.sanitized_text,
        "input_passed": result.passed,
        "intent": result.metadata.get("intent"),
    }


@celery_app.task
def update_persona_state(input_data: dict[str, Any]) -> dict[str, Any]:
    session_id = input_data["session_id"]
    intent = input_data["intent"]

    # Load persona and session state
    persona_def = PERSONA_STORE.get("default")  # Placeholder
    session_state = SESSION_STORE.get(
        session_id,
        {
            "session_id": session_id,
            "current_state": "presentation",
            "variables": {
                "pain_level": 7,
                "anxiety_level": 4,
                "disclosed_symptoms": [],
                "hidden_symptoms": ["radiating_pain_to_arm"],
            },
            "history": [],
        },
    )

    sm = PersonaStateMachine.from_dict(session_state, persona_def)
    sm.transition(intent)

    # Save back
    new_state = sm.to_dict()
    SESSION_STORE[session_id] = new_state

    input_data["current_state"] = new_state
    input_data["persona_definition"] = persona_def
    return input_data


@celery_app.task(
    bind=True,
    max_retries=1,
    default_retry_delay=5,
    acks_late=True,
)
def generate_llm_response(self, data: dict[str, Any]) -> dict[str, Any]:
    if not data.get("input_passed"):
        data["llm_output"] = "I'm sorry, I cannot process that request."
        data["llm_failure"] = False
        return data

    try:
        provider = LLMProviderFactory.create()
    except ValueError as exc:
        logger.error(
            "llm_provider_creation_failed",
            error=str(exc),
            session_id=data.get("session_id"),
        )
        data["llm_output"] = (
            "I'm sorry, the language model provider is not configured correctly. Please contact your administrator."
        )
        data["llm_failure"] = True
        data["llm_failure_reason"] = str(exc)
        return data

    engine = InferenceEngine(provider)

    try:
        response = engine.generate_response(
            data["sanitized_input"],
            data["persona_definition"],
            data["current_state"],
        )
    except Exception as exc:
        logger.error(
            "llm_generation_failed",
            error=str(exc),
            session_id=data.get("session_id"),
        )
        data["llm_output"] = "I'm sorry, I encountered an issue while generating a response. Please try again."
        data["llm_failure"] = True
        data["llm_failure_reason"] = str(exc)
        return data

    data["llm_output"] = response
    data["llm_failure"] = False
    return data


@celery_app.task
def run_safety_output_guard(data: dict[str, Any]) -> dict[str, Any]:
    guard = OutputGuard(data["persona_definition"])
    result = guard.run(data["llm_output"], data["current_state"]["current_state"])

    data["output_passed"] = result.passed
    data["sanitized_output"] = result.sanitized_text
    data["guard_message"] = result.message

    return data


@celery_app.task
def broadcast_response(data: dict[str, Any]) -> dict[str, Any]:
    # In a real app, this would send to a WebSocket or similar
    return data
