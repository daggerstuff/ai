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


def test_human_likeness_validation_allows_therapeutic_lists(manager):
    """Lists in therapeutic context (exercises, homework) should be allowed."""
    therapeutic_text = (
        "Here's your homework:\n1. Practice deep breathing\n2. "
        "Write in your journal\n3. Try the grounding exercise"
    )
    assert manager.validate_human_likeness(therapeutic_text) is True


def test_human_likeness_validation_reject_ai_self_identification(manager):
    """AI self-identification should be rejected."""
    ai_text = "As an AI, I cannot feel emotions, but I can help you."
    assert manager.validate_human_likeness(ai_text) is False


def test_human_likeness_validation_reject_lecture_phrases(manager):
    """Pedantic/lecture phrases should be rejected."""
    lecture_text = "It is important to remember that therapy takes time."
    assert manager.validate_human_likeness(lecture_text) is False


def test_human_likeness_validation_reject_apology_patterns(manager):
    """Over-apologizing patterns should be rejected."""
    apology_text = "I apologize if my response wasn't helpful."
    assert manager.validate_human_likeness(apology_text) is False


def test_human_likeness_validation_reject_excessive_bullets(manager):
    """More than 4 bullet points should be rejected."""
    bullet_text = (
        "Things to consider:\n- First point\n- Second point\n"
        "- Third point\n- Fourth point\n- Fifth point"
    )
    assert manager.validate_human_likeness(bullet_text) is False


def test_human_likeness_validation_allows_fewer_bullets(manager):
    """4 or fewer bullet points should be allowed."""
    reasonable_text = "I'm feeling:\n- Sad\n- Anxious\n- Tired"
    assert manager.validate_human_likeness(reasonable_text) is True


def test_human_likeness_validation_soft_penalty_accumulation(manager):
    """Multiple soft penalties should trigger rejection."""
    soft_penalty_text = (
        "I hope this helps. Please let me know if you need anything. "
        "Feel free to reach out. There is hope for you."
    )
    assert manager.validate_human_likeness(soft_penalty_text) is False
