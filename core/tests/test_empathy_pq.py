import pytest
from ai.core.empathy_pq import EmpathyPQCalculator
from ai.core.gestalt_engine import CrisisLevel, GestaltState


@pytest.fixture
def calculator():
    return EmpathyPQCalculator()


def make_mock_state(maturity: float) -> GestaltState:
    return GestaltState(
        defense_label=1,
        defense_label_name="Mock",
        defense_confidence=0.9,
        defense_maturity=maturity,
        defense_probabilities={},
        plutchik_scores={},
        dominant_emotion="mock",
        dominant_emotion_intensity=0.5,
        ocean_scores={},
        crisis_level=CrisisLevel.NONE,
        behavioral_prediction="mock",
        persona_directive="mock",
        breakthrough_score=0.0,
    )


def test_initial_pq_is_50(calculator):
    state = make_mock_state(3.5)
    score = calculator.calculate_pq_increment(state, previous_maturity=3.5)
    assert score.overall_pq == 50.0


def test_invalidation_penalty(calculator):
    # Start at 3.5
    calculator.calculate_pq_increment(make_mock_state(3.5))
    # Drop to 1.5 (Delta -2.0)
    # Penalty is 4.0x
    # 50.0 - (2.0 * 4.0) = 42.0
    score = calculator.calculate_pq_increment(make_mock_state(1.5))
    assert score.overall_pq == 42.0
    assert score.invalidation_detected is True


def test_recovery_multiplier(calculator):
    calculator.calculate_pq_increment(make_mock_state(3.0))
    # Rise to 5.0 (Delta +2.0)
    # Multiplier is 2.5x
    # 50.0 + (2.0 * 2.5) = 55.0
    score = calculator.calculate_pq_increment(make_mock_state(5.0))
    assert score.overall_pq == 55.0


def test_breakthrough_bonus(calculator):
    calculator.calculate_pq_increment(make_mock_state(6.0))
    # Rise to 7.0 (Delta +1.0)
    # 50.0 + (1.0 * 2.5) + 15.0 (Bonus) = 67.5
    score = calculator.calculate_pq_increment(make_mock_state(7.0))
    assert score.overall_pq == 67.5
    assert score.breakthrough_detected is True


def test_performance_category(calculator):
    # Huge breakthrough
    calculator.calculate_pq_increment(make_mock_state(1.0))
    calculator.calculate_pq_increment(make_mock_state(7.5))
    summary = calculator.get_session_summary()
    assert summary["performance_category"] in ["Elite", "Clinical"]
