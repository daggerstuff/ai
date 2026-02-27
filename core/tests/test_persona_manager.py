import pytest

from ai.core.persona_manager import PersonaManager


@pytest.fixture
def manager():
    return PersonaManager()


def test_persona_manager_initialization(manager):
    assert len(manager.get_available_archetypes()) == 10


def test_get_specific_persona(manager):
    persona = manager.get_persona("bpd")
    assert persona is not None
    assert persona.name == "Borderline / Emotionally Dysregulated"
    assert persona.default_defense == "Splitting"


def test_system_prompt_generation(manager):
    persona = manager.get_persona("avoidant")
    prompt = persona.generate_system_prompt()
    assert "Avoidant / Dismissive" in prompt
    assert "Disavowal" in prompt


def test_human_likeness_validation_pass(manager):
    human_text = (
        "I just don't know what to do anymore. It feels like my chest is "
        "tight and I can't breathe."
    )
    assert manager.validate_human_likeness(human_text) is True


def test_human_likeness_validation_reject_robotic(manager):
    robot_text = (
        "I understand that you are feeling anxious. As an AI, I cannot feel "
        "emotions, but I'm here to help."
    )
    assert manager.validate_human_likeness(robot_text) is False


def test_human_likeness_validation_reject_lists(manager):
    list_text = "I feel bad today.\n1. I'm sad.\n2. I'm anxious.\n3. I'm tired."
    assert manager.validate_human_likeness(list_text) is False
