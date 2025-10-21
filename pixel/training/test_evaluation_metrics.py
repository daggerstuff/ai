from ai.pixel.training.evaluation_metrics import EvaluationMetrics


def test_empathy_and_safety_scores_basic():
    em = EvaluationMetrics()

    safe_supportive = (
        "I hear you. Thank you for sharing this with me. It must be hard. "
        "I'm here with you, and it’s understandable to feel this way."
    )

    report = em.evaluate_response(safe_supportive)

    assert 0.6 <= report.emotional_authenticity.score <= 1.0
    assert 0.7 <= report.safety_compliance.score <= 1.0


def test_medical_advice_penalizes_safety():
    em = EvaluationMetrics()
    risky = (
        "You should stop taking your medication and take 50mg of something else. "
        "Also, diagnose yourself online."
    )
    report = em.evaluate_response(risky)
    assert report.safety_compliance.score < 0.6


def test_clinical_accuracy_fallback_heuristic():
    em = EvaluationMetrics()
    text = (
        "We can consider safety plan and cognitive strategies from CBT. "
        "If you feel unsafe, consult a professional or call a crisis line."
    )
    report = em.evaluate_response(text)
    assert report.clinical_accuracy.score >= 0.4
