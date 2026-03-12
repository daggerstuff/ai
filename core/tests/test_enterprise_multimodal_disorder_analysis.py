"""
Targeted tests for enterprise multi-modal feature gating and fusion weighting.
"""

import pytest

from ai.core.pipelines.processing.enterprise_multimodal_disorder_analysis import (
    EnterpriseAnalysisResult,
    EnterpriseModalityFeatures,
    EnterpriseMultiModalDisorderAnalyzer,
    ModalityType,
)


def _analysis_config() -> dict:
    """Build a stable analyzer config for focused unit tests."""
    analyzer = EnterpriseMultiModalDisorderAnalyzer()
    return analyzer.config


def _base_feature_vector() -> dict:
    """Build reusable feature vector used by unit tests."""
    return {
        "sentiment_score": 0.8,
        "emotional_intensity": 0.8,
        "linguistic_complexity": 0.8,
    }


def test_fusion_gates_low_confidence_modalities():
    """Low-confidence modalities should be de-prioritized in final disorder scoring."""
    config = _analysis_config()
    config["disorder_categories"] = ["depression"]
    config["modality_weights"] = {
        ModalityType.TEXT: 0.7,
        ModalityType.AUDIO: 0.3,
    }
    analyzer = EnterpriseMultiModalDisorderAnalyzer(config=config)

    extracted_features = {
        ModalityType.TEXT: EnterpriseModalityFeatures(
            modality_type=ModalityType.TEXT,
            feature_vector=_base_feature_vector(),
            confidence_score=0.95,
            quality_metrics={"reliability": 0.9, "completeness": 1.0},
        ),
        ModalityType.AUDIO: EnterpriseModalityFeatures(
            modality_type=ModalityType.AUDIO,
            feature_vector={"audio_signal_confidence": 0.1},
            confidence_score=0.02,
            quality_metrics={"reliability": 0.9, "completeness": 0.2},
        ),
    }
    predictions = analyzer._perform_fusion_analysis(extracted_features)
    contributions = analyzer._calculate_modality_contributions(extracted_features)

    assert predictions["depression"] == pytest.approx(0.9)
    assert ModalityType.TEXT in contributions
    assert ModalityType.AUDIO not in contributions


def test_extracted_features_capture_visual_micro_leakage_signal():
    """Behavioral extraction should expose micro-leakage risk for downstream alignment."""
    analyzer = EnterpriseMultiModalDisorderAnalyzer()
    modality_output = analyzer._extract_all_modality_features(
        {
            "modalities": {
                "text": {"content": "I am trying to keep control and stay grounded."},
                "audio": {"features": {"speech_rate": 175, "pause_ratio": 0.28}},
                "behavioral": {
                    "visual_modality": {
                        "facial_expressions": {"sadness": 0.44, "anger": 0.12},
                        "body_language": {
                            "eye_contact_ratio": 0.58,
                            "gesture_frequency": 0.31,
                            "movement_energy": 0.44,
                            "posture": "tense",
                        },
                        "micro_expressions": {
                            "detected_count": 9,
                            "authenticity_score": 0.36,
                        },
                    }
                },
            }
        }
    )

    behavioral = modality_output[ModalityType.BEHAVIORAL]
    assert "micro_leakage_risk" in behavioral.feature_vector
    assert 0.0 <= behavioral.confidence_score <= 1.0


def test_full_analyzer_returns_result_with_contributions():
    """Analyzer output should include confidence-aware modality contribution entries."""
    analyzer = EnterpriseMultiModalDisorderAnalyzer()
    result = analyzer.analyze_conversation(
        {
            "conversation_id": "test-enterprise-pix001",
            "modalities": {
                "text": {"content": "I'm feeling better today after breathing exercise."},
                "audio": {"features": {"speech_rate": 128, "pause_ratio": 0.3}},
            },
        }
    )

    assert isinstance(result, EnterpriseAnalysisResult)
    assert result.modality_contributions
    assert set(result.modality_contributions).issubset({ModalityType.TEXT, ModalityType.AUDIO})
    assert result.quality_score >= 0.0


def test_analysis_telemetry_tracks_confidence_floor_hits():
    """Telemetry should report how many modalities passed confidence gating."""
    analyzer = EnterpriseMultiModalDisorderAnalyzer()
    features = {
        ModalityType.TEXT: EnterpriseModalityFeatures(
            modality_type=ModalityType.TEXT,
            feature_vector=_base_feature_vector(),
            confidence_score=0.95,
            quality_metrics={"reliability": 0.9},
        ),
        ModalityType.AUDIO: EnterpriseModalityFeatures(
            modality_type=ModalityType.AUDIO,
            feature_vector={"audio_signal_confidence": 0.1},
            confidence_score=0.02,
            quality_metrics={"reliability": 0.9},
        ),
        ModalityType.BEHAVIORAL: EnterpriseModalityFeatures(
            modality_type=ModalityType.BEHAVIORAL,
            feature_vector={"micro_leakage_risk": 0.55},
            confidence_score=0.41,
            quality_metrics={"reliability": 0.88},
        ),
    }

    telemetry = analyzer._build_analysis_telemetry(features)

    assert telemetry["confidence_floor_hits"] == 2
    assert telemetry["confidence_floor_hit_ratio"] == pytest.approx(2 / 3)


def test_micro_leakage_trend_is_reported_in_consecutive_analyses():
    """Consecutive analyses should include micro-leakage trend direction."""
    analyzer = EnterpriseMultiModalDisorderAnalyzer()

    low_leakage = {
        "conversation_id": "pix-micro-1",
        "modalities": {
            "text": {"content": "I stayed calm and grounded today"},
            "behavioral": {
                "visual_modality": {
                    "facial_expressions": {"sadness": 0.1, "anger": 0.05},
                    "body_language": {
                        "eye_contact_ratio": 0.75,
                        "gesture_frequency": 0.2,
                        "movement_energy": 0.2,
                        "posture": "open",
                    },
                    "micro_expressions": {"detected_count": 4, "authenticity_score": 0.9},
                }
            },
        },
    }
    high_leakage = {
        "conversation_id": "pix-micro-2",
        "modalities": {
            "text": {"content": "I'm spiraling and panic is coming back hard"},
            "behavioral": {
                "visual_modality": {
                    "facial_expressions": {"sadness": 0.72, "anger": 0.5},
                    "body_language": {
                        "eye_contact_ratio": 0.3,
                        "gesture_frequency": 0.55,
                        "movement_energy": 0.62,
                        "posture": "tense",
                    },
                    "micro_expressions": {"detected_count": 16, "authenticity_score": 0.1},
                }
            },
        },
    }

    first = analyzer.analyze_conversation(low_leakage)
    second = analyzer.analyze_conversation(high_leakage)

    first_telemetry = first.metadata["analysis_telemetry"]
    second_telemetry = second.metadata["analysis_telemetry"]

    assert first_telemetry["micro_leakage_signal"] >= 0.0
    assert second_telemetry["micro_leakage_signal"] > first_telemetry["micro_leakage_signal"]
    assert second_telemetry["micro_leakage_delta"] > 0
    assert second_telemetry["micro_leakage_trend"] == "increasing"
