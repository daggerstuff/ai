from ai.pipelines.model_training.core.state_machine import PersonaStateMachine, State
from ai.pipelines.model_training.safety.guards import InputGuard, OutputGuard


def test_state_machine_transitions():
    persona_def = {
        "clinical_profile": {
            "symptoms": [{"name": "pain", "severity": 7}],
            "medical_history": "None"
        },
        "emotional_state": {"volatility": 5}
    }
    sm = PersonaStateMachine("test_session", persona_def)

    assert sm.current_state == State.PRESENTATION

    # Test transition to History Revealed
    sm.transition("ask_history")
    assert sm.current_state == State.HISTORY_REVEALED
    assert "medical_history" in sm.variables["disclosed_symptoms"]

    # Test transition to Assessment
    sm.transition("perform_intervention")
    assert sm.current_state == State.ASSESSMENT

def test_state_machine_escalation():
    persona_def = {
        "clinical_profile": {
            "symptoms": [{"name": "pain", "severity": 7}]
        },
        "emotional_state": {"volatility": 5}
    }
    sm = PersonaStateMachine("test_session", persona_def)

    # Neglect 3 times
    sm.transition("neglect")
    sm.transition("neglect")
    sm.transition("neglect")

    assert sm.current_state == State.ESCALATION
    assert sm.variables["pain_level"] == 9

    # Address pain
    sm.transition("address_pain")
    assert sm.current_state == State.PRESENTATION
    assert sm.neglect_count == 0

def test_input_guard_phi_sanitization():
    guard = InputGuard()
    user_input = "My SSN is 123-45-6789 and my email is test@example.com"
    result = guard.run(user_input)

    assert "[SSN]" in result.sanitized_text
    assert "[EMAIL]" in result.sanitized_text
    assert "123-45-6789" not in result.sanitized_text

def test_input_guard_intent_detection():
    guard = InputGuard()

    assert guard.run("What is your medical history?").metadata["intent"] == "ask_history"
    assert guard.run("I will give you some medicine").metadata["intent"] == "perform_intervention"
    assert guard.run("Are you in pain?").metadata["intent"] == "address_pain"

def test_output_guard_medical_accuracy():
    persona_def = {"metadata": {"name": "John"}}
    guard = OutputGuard(persona_def)

    # Escalation state should not allow "no pain"
    result = guard.run("I have no pain at all", "escalation")
    assert result.passed is False
    assert "accuracy" in result.message.lower()

    # Presentation state should allow it
    result = guard.run("I have no pain at all", "presentation")
    assert result.passed is True
