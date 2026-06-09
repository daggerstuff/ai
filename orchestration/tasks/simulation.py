from orchestration.celery_app import celery_app
from orchestration.safety.guards import InputGuard, OutputGuard
from orchestration.core.state_machine import PersonaStateMachine
from orchestration.core.inference import InferenceEngine, MockLLMProvider
from typing import Dict, Any
import json

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
            "medications": ["Lisinopril"]
        },
        "communication_style": {"tone": "anxious", "vocabulary_level": "layman", "verbosity": "medium"},
        "emotional_state": {"baseline": "scared", "volatility": 6}
    }
}

@celery_app.task
def run_safety_input_guard(session_id: str, user_input: str) -> Dict[str, Any]:
    guard = InputGuard()
    result = guard.run(user_input)
    
    return {
        "session_id": session_id,
        "user_input": user_input,
        "sanitized_input": result.sanitized_text,
        "input_passed": result.passed,
        "intent": result.metadata.get("intent")
    }

@celery_app.task
def update_persona_state(input_data: Dict[str, Any]) -> Dict[str, Any]:
    session_id = input_data["session_id"]
    intent = input_data["intent"]
    
    # Load persona and session state
    persona_def = PERSONA_STORE.get("default") # Placeholder
    session_state = SESSION_STORE.get(session_id, {
        "session_id": session_id,
        "current_state": "presentation",
        "variables": {
            "pain_level": 7,
            "anxiety_level": 4,
            "disclosed_symptoms": [],
            "hidden_symptoms": ["radiating_pain_to_arm"]
        },
        "history": []
    })
    
    sm = PersonaStateMachine.from_dict(session_state, persona_def)
    sm.transition(intent)
    
    # Save back
    new_state = sm.to_dict()
    SESSION_STORE[session_id] = new_state
    
    input_data["current_state"] = new_state
    input_data["persona_definition"] = persona_def
    return input_data

@celery_app.task
def generate_llm_response(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data.get("input_passed"):
        data["llm_output"] = "I'm sorry, I cannot process that request."
        return data

    engine = InferenceEngine(MockLLMProvider())
    response = engine.generate_response(
        data["sanitized_input"],
        data["persona_definition"],
        data["current_state"]
    )
    
    data["llm_output"] = response
    return data

@celery_app.task
def run_safety_output_guard(data: Dict[str, Any]) -> Dict[str, Any]:
    guard = OutputGuard(data["persona_definition"])
    result = guard.run(data["llm_output"], data["current_state"]["current_state"])
    
    data["output_passed"] = result.passed
    data["sanitized_output"] = result.sanitized_text
    data["guard_message"] = result.message
    
    return data

@celery_app.task
def broadcast_response(data: Dict[str, Any]) -> Dict[str, Any]:
    # In a real app, this would send to a WebSocket or similar
    print(f"BROADCASTING to {data['session_id']}: {data['sanitized_output']}")
    return data
