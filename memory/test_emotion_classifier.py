"""Unit tests for ai/memory/emotion_classifier.py — PIX-510 Task 3."""

from __future__ import annotations

from ai.memory.emotion_classifier import (
    ALL_EMOTION_CATEGORIES,
    EMOTION_MULTIPLIER,
    EmotionClassificationResult,
    EmotionClassifier,
    VADScorer,
    emotion_multiplier,
)

# ─── VAD Scorer ───────────────────────────────────────────────────────────────


class TestVADScorer:
    def test_positive_valence(self) -> None:
        vad = VADScorer()
        v, _a, _d = vad.score("I am so happy and grateful today")
        assert v > 0.6

    def test_negative_valence(self) -> None:
        vad = VADScorer()
        v, _a, _d = vad.score("I feel devastated and hopeless about everything")
        assert v < 0.4

    def test_neutral_text(self) -> None:
        vad = VADScorer()
        v, _a, _d = vad.score("The meeting is scheduled for 3pm tomorrow")
        assert 0.3 < v < 0.7

    def test_all_components_in_range(self) -> None:
        vad = VADScorer()
        for text in [
            "I love you so much!",
            "This is terrible and awful",
            "The file is on the desk",
        ]:
            v, a, d = vad.score(text)
            assert 0.0 <= v <= 1.0
            assert 0.0 <= a <= 1.0
            assert 0.0 <= d <= 1.0


# ─── Emotion multiplier ──────────────────────────────────────────────────────


class TestEmotionMultiplier:
    def test_crisis_multipliers(self) -> None:
        for cat in ["suicide", "self-harm"]:
            assert emotion_multiplier([cat]) == 5.0

    def test_high_multipliers(self) -> None:
        # anxiety/fear/anger = 2.0, grief/trauma/despair/hopelessness = 2.5
        assert emotion_multiplier(["anxiety"]) == 2.0
        assert emotion_multiplier(["fear"]) == 2.0
        assert emotion_multiplier(["anger"]) == 2.0
        assert emotion_multiplier(["grief"]) == 2.5

    def test_highest_wins(self) -> None:
        # grief (2.5) beats anxiety (2.0)
        assert emotion_multiplier(["joy", "grief"]) == 2.5
        assert emotion_multiplier(["anxiety", "suicide"]) == 5.0

    def test_normal_multipliers(self) -> None:
        for cat in ["joy", "trust", "anticipation"]:
            assert emotion_multiplier([cat]) == 1.0

    def test_empty_returns_normal(self) -> None:
        assert emotion_multiplier([]) == 1.0
        assert emotion_multiplier(["anxiety", "suicide"]) == 5.0


# ─── Emotion classifier ───────────────────────────────────────────────────────


class TestEmotionClassifier:
    def test_classifies_fear(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I feel really anxious and scared about the interview")
        assert result.top_category == "fear"
        assert result.multiplier >= 2.0
        assert "fear" in result.categories

    def test_classifies_joy(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I am so happy and grateful for your support")
        assert result.top_category == "joy"
        assert result.multiplier == 1.0

    def test_classifies_crisis_selfharm(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I want to hurt myself and end everything")
        assert result.top_category == "self-harm"
        assert result.multiplier == 5.0

    def test_classifies_crisis_suicide(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I have no reason to live and want to end my life")
        assert result.top_category == "suicide"
        assert result.multiplier == 5.0

    def test_empty_input(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("")
        assert result.categories == []
        assert result.top_category is None
        assert result.multiplier == 1.0
        assert result.valence == 0.5

    def test_whitespace_input(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("   ")
        assert result.categories == []

    def test_multi_label_true(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I feel sad and anxious today", multi_label=True)
        assert len(result.categories) >= 2
        assert result.multiplier >= 2.0

    def test_vad_scores_populated(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        result = clf.classify("I am feeling really sad and overwhelmed")
        assert 0.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.dominance <= 1.0

    def test_batch_classification(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        texts = [
            "I feel happy today",
            "I feel anxious about the test",
            "I feel nothing in particular",
        ]
        results = clf.classify_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, EmotionClassificationResult) for r in results)

    def test_latency_under_50ms(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        text = "I feel anxious about my upcoming medical appointment"
        ms = clf.benchmark_latency(text, n=500)
        assert ms < 50.0, f"Latency {ms:.3f}ms exceeds 50ms threshold"


# ─── Session trajectory ───────────────────────────────────────────────────────


class TestSessionTrajectory:
    def test_stable_trajectory(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        results = [
            clf.classify("I am feeling okay today"),
            clf.classify("I am feeling okay today"),
        ]
        traj = clf.session_trajectory(results)
        assert traj.trend == "stable"
        assert traj.crisis_indicators == []

    def test_empty_results(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        traj = clf.session_trajectory([])
        assert traj.trend == "stable"
        assert traj.max_intensity == 0.0
        assert traj.crisis_indicators == []

    def test_crisis_detection(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        results = [
            clf.classify("I am feeling okay today"),
            clf.classify("I feel panicked and scared"),
            clf.classify("I want to hurt myself"),
        ]
        traj = clf.session_trajectory(results)
        assert len(traj.crisis_indicators) >= 1
        assert traj.crisis_indicators[0] in ("suicide", "self-harm")

    def test_trajectory_returns_all_fields(self) -> None:
        clf = EmotionClassifier(mode="lexicon")
        results = [clf.classify("I feel happy"), clf.classify("I feel sad")]
        traj = clf.session_trajectory(results)
        assert 0.0 <= traj.start_valence <= 1.0
        assert 0.0 <= traj.end_valence <= 1.0
        assert traj.trend in ("escalating", "de-escalating", "stable", "volatile")
        assert traj.max_intensity >= 0.0
        assert isinstance(traj.trajectory_scores, list)


# ─── Plutchik categories ─────────────────────────────────────────────────────


class TestPlutchikCategories:
    def test_all_primary_present(self) -> None:
        expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"}
        assert expected.issubset(ALL_EMOTION_CATEGORIES)

    def test_crisis_categories_defined(self) -> None:
        for cat in ["suicide", "self-harm"]:
            assert cat in EMOTION_MULTIPLIER
            assert EMOTION_MULTIPLIER[cat] == 5.0
