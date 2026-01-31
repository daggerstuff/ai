"""
Voice-Derived Authenticity Scorer

Implements authenticity scoring for voice-derived conversations (Task 3.6).
Specialized authenticity assessment for conversations extracted from voice data.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class VoiceAuthenticityMetrics:
    """Voice-derived authenticity metrics."""
    vocal_authenticity_score: float = 0.0
    conversational_authenticity_score: float = 0.0
    emotional_authenticity_score: float = 0.0
    overall_authenticity_score: float = 0.0
    authenticity_indicators: list[str] = field(default_factory=list)
    authenticity_concerns: list[str] = field(default_factory=list)

@dataclass
class VoiceAuthenticityAssessment:
    """Complete voice-derived authenticity assessment."""
    conversation_id: str
    voice_source_path: str
    assessment_timestamp: str
    authenticity_metrics: VoiceAuthenticityMetrics
    confidence_level: str  # "high", "medium", "low"
    recommendations: list[str] = field(default_factory=list)

class VoiceDerivedAuthenticityScorer:
    """Implements authenticity scoring for voice-derived conversations."""

    def __init__(self, voice_data_path: str = "./voice_data", output_dir: str = "./authenticity_scores"):
        self.voice_data_path = Path(voice_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Authenticity indicators for voice-derived conversations
        self.authenticity_indicators = {
            "vocal_authenticity": {
                "natural_speech_patterns": ["hesitations", "natural_pauses", "speech_variations"],
                "emotional_vocal_cues": ["tone_changes", "pitch_variations", "volume_modulation"],
                "spontaneous_elements": ["interruptions", "overlapping_speech", "natural_corrections"],
                "prosodic_features": ["rhythm_variations", "stress_patterns", "intonation_changes"]
            },
            "conversational_authenticity": {
                "natural_dialogue_flow": ["turn_taking", "response_timing", "conversation_rhythm"],
                "authentic_interactions": ["genuine_questions", "natural_responses", "organic_development"],
                "contextual_coherence": ["topic_continuity", "reference_consistency", "logical_progression"],
                "spontaneous_content": ["unscripted_moments", "natural_tangents", "authentic_reactions"]
            },
            "emotional_authenticity": {
                "genuine_emotions": ["authentic_emotional_expression", "consistent_emotional_tone", "natural_emotional_progression"],
                "emotional_congruence": ["voice_emotion_alignment", "content_emotion_match", "contextual_appropriateness"],
                "emotional_depth": ["emotional_complexity", "nuanced_expressions", "emotional_development"],
                "emotional_spontaneity": ["unguarded_moments", "natural_emotional_responses", "authentic_vulnerability"]
            }
        }

        # Authenticity concerns (red flags)
        self.authenticity_concerns = {
            "artificial_patterns": ["robotic_speech", "unnatural_timing", "mechanical_responses"],
            "scripted_indicators": ["overly_polished_speech", "perfect_grammar", "rehearsed_responses"],
            "emotional_inconsistencies": ["emotional_mismatch", "inappropriate_emotions", "flat_emotional_range"],
            "technical_artifacts": ["audio_splicing", "voice_synthesis_indicators", "editing_artifacts"]
        }

        # Scoring weights
        self.scoring_weights = {
            "vocal_authenticity": 0.35,
            "conversational_authenticity": 0.35,
            "emotional_authenticity": 0.30
        }

        logger.info("VoiceDerivedAuthenticityScorer initialized")

    def score_voice_derived_authenticity(self, voice_conversation_data: dict[str, Any]) -> VoiceAuthenticityAssessment:
        """Score authenticity of voice-derived conversation."""
        datetime.now()

        try:
            # Extract conversation ID
            conversation_id = voice_conversation_data.get("conversation_id", f"voice_auth_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Load voice and conversation data
            voice_features = self._extract_voice_features(voice_conversation_data)
            conversation_features = self._extract_conversation_features(voice_conversation_data)

            # Score vocal authenticity
            vocal_authenticity = self._score_vocal_authenticity(voice_features)

            # Score conversational authenticity
            conversational_authenticity = self._score_conversational_authenticity(conversation_features)

            # Score emotional authenticity
            emotional_authenticity = self._score_emotional_authenticity(voice_features, conversation_features)

            # Calculate overall authenticity
            overall_authenticity = self._calculate_overall_authenticity(
                vocal_authenticity, conversational_authenticity, emotional_authenticity
            )

            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_authenticity)

            # Generate recommendations
            recommendations = self._generate_authenticity_recommendations(
                vocal_authenticity, conversational_authenticity, emotional_authenticity
            )

            # Create authenticity metrics
            authenticity_metrics = VoiceAuthenticityMetrics(
                vocal_authenticity_score=vocal_authenticity["score"],
                conversational_authenticity_score=conversational_authenticity["score"],
                emotional_authenticity_score=emotional_authenticity["score"],
                overall_authenticity_score=overall_authenticity["score"],
                authenticity_indicators=overall_authenticity["indicators"],
                authenticity_concerns=overall_authenticity["concerns"]
            )

            # Create assessment
            assessment = VoiceAuthenticityAssessment(
                conversation_id=conversation_id,
                voice_source_path=voice_conversation_data.get("voice_source_path", "unknown"),
                assessment_timestamp=datetime.now().isoformat(),
                authenticity_metrics=authenticity_metrics,
                confidence_level=confidence_level,
                recommendations=recommendations
            )

            logger.info(f"Voice-derived authenticity scoring completed: {conversation_id} - {overall_authenticity['score']:.2f}")
            return assessment

        except Exception as e:
            logger.error(f"Voice-derived authenticity scoring failed: {e}")
            # Return failed assessment
            return VoiceAuthenticityAssessment(
                conversation_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                voice_source_path="unknown",
                assessment_timestamp=datetime.now().isoformat(),
                authenticity_metrics=VoiceAuthenticityMetrics(),
                confidence_level="low",
                recommendations=[f"Assessment failed: {e!s}"]
            )

    def batch_score_authenticity(self, voice_conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Score authenticity for multiple voice-derived conversations."""
        start_time = datetime.now()

        assessments = []
        authenticity_stats = {"high": 0, "medium": 0, "low": 0}

        for conversation_data in voice_conversations:
            assessment = self.score_voice_derived_authenticity(conversation_data)
            assessments.append(assessment)

            # Update stats based on overall score
            if assessment.authenticity_metrics.overall_authenticity_score >= 0.8:
                authenticity_stats["high"] += 1
            elif assessment.authenticity_metrics.overall_authenticity_score >= 0.6:
                authenticity_stats["medium"] += 1
            else:
                authenticity_stats["low"] += 1

        # Calculate batch statistics
        batch_stats = self._calculate_batch_authenticity_statistics(assessments)

        # Save batch results
        output_path = self._save_batch_authenticity_results(assessments, batch_stats)

        return {
            "success": True,
            "conversations_scored": len(voice_conversations),
            "authenticity_distribution": authenticity_stats,
            "batch_statistics": batch_stats,
            "assessments": assessments,
            "output_path": str(output_path),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

    def _extract_voice_features(self, voice_conversation_data: dict[str, Any]) -> dict[str, Any]:
        """Extract voice-specific features for authenticity assessment."""
        # Mock voice features extraction
        return {
            "speech_patterns": {
                "natural_hesitations": 0.75,
                "speech_rate_variation": 0.68,
                "pause_naturalness": 0.82,
                "pronunciation_consistency": 0.91
            },
            "prosodic_features": {
                "pitch_variation": 0.73,
                "volume_modulation": 0.67,
                "rhythm_naturalness": 0.79,
                "stress_patterns": 0.84
            },
            "emotional_vocal_cues": {
                "tone_authenticity": 0.76,
                "emotional_expression": 0.71,
                "vocal_emotion_consistency": 0.83,
                "spontaneous_reactions": 0.69
            },
            "technical_quality": {
                "audio_naturalness": 0.88,
                "editing_artifacts": 0.05,  # Lower is better
                "synthesis_indicators": 0.03,  # Lower is better
                "splicing_evidence": 0.02  # Lower is better
            }
        }

    def _extract_conversation_features(self, voice_conversation_data: dict[str, Any]) -> dict[str, Any]:
        """Extract conversation-specific features for authenticity assessment."""
        # Mock conversation features extraction
        return {
            "dialogue_flow": {
                "turn_taking_naturalness": 0.81,
                "response_timing": 0.74,
                "conversation_rhythm": 0.77,
                "interruption_patterns": 0.69
            },
            "content_authenticity": {
                "spontaneous_content": 0.72,
                "natural_language_use": 0.85,
                "contextual_coherence": 0.79,
                "topic_development": 0.76
            },
            "interaction_quality": {
                "genuine_questions": 0.83,
                "natural_responses": 0.78,
                "organic_development": 0.71,
                "authentic_reactions": 0.74
            },
            "linguistic_patterns": {
                "natural_grammar": 0.68,  # Some imperfection is natural
                "vocabulary_authenticity": 0.82,
                "speech_patterns": 0.75,
                "colloquial_usage": 0.73
            }
        }

    def _score_vocal_authenticity(self, voice_features: dict[str, Any]) -> dict[str, Any]:
        """Score vocal authenticity based on voice features."""

        # Weight different vocal aspects
        weights = {
            "speech_patterns": 0.3,
            "prosodic_features": 0.25,
            "emotional_vocal_cues": 0.25,
            "technical_quality": 0.2
        }

        component_scores = {}
        indicators = []
        concerns = []

        for category, features in voice_features.items():
            if category == "technical_quality":
                # For technical quality, lower artifact scores are better
                artifacts = ["editing_artifacts", "synthesis_indicators", "splicing_evidence"]
                artifact_penalty = sum(features.get(artifact, 0) for artifact in artifacts) / len(artifacts)
                naturalness = features.get("audio_naturalness", 0.5)
                component_scores[category] = (naturalness + (1 - artifact_penalty)) / 2

                if artifact_penalty > 0.1:
                    concerns.append("Technical artifacts detected")
                if naturalness > 0.8:
                    indicators.append("High audio naturalness")
            else:
                # For other categories, higher scores are better
                category_score = sum(features.values()) / len(features)
                component_scores[category] = category_score

                if category_score > 0.8:
                    indicators.append(f"Strong {category.replace('_', ' ')}")
                elif category_score < 0.6:
                    concerns.append(f"Weak {category.replace('_', ' ')}")

        # Calculate weighted overall score
        overall_score = sum(component_scores[cat] * weights[cat] for cat in weights)

        return {
            "score": overall_score,
            "components": component_scores,
            "indicators": indicators,
            "concerns": concerns
        }

    def _score_conversational_authenticity(self, conversation_features: dict[str, Any]) -> dict[str, Any]:
        """Score conversational authenticity based on conversation features."""

        # Weight different conversational aspects
        weights = {
            "dialogue_flow": 0.3,
            "content_authenticity": 0.3,
            "interaction_quality": 0.25,
            "linguistic_patterns": 0.15
        }

        component_scores = {}
        indicators = []
        concerns = []

        for category, features in conversation_features.items():
            category_score = sum(features.values()) / len(features)
            component_scores[category] = category_score

            if category_score > 0.8:
                indicators.append(f"Excellent {category.replace('_', ' ')}")
            elif category_score < 0.6:
                concerns.append(f"Poor {category.replace('_', ' ')}")

        # Calculate weighted overall score
        overall_score = sum(component_scores[cat] * weights[cat] for cat in weights)

        # Special checks for authenticity
        if component_scores.get("linguistic_patterns", 0) > 0.9:
            concerns.append("Suspiciously perfect language patterns")

        return {
            "score": overall_score,
            "components": component_scores,
            "indicators": indicators,
            "concerns": concerns
        }

    def _score_emotional_authenticity(self, voice_features: dict[str, Any],
                                    conversation_features: dict[str, Any]) -> dict[str, Any]:
        """Score emotional authenticity based on voice and conversation features."""

        # Extract emotional components
        vocal_emotions = voice_features.get("emotional_vocal_cues", {})
        interaction_emotions = conversation_features.get("interaction_quality", {})

        # Calculate emotional consistency
        vocal_emotion_score = sum(vocal_emotions.values()) / len(vocal_emotions) if vocal_emotions else 0.5
        interaction_emotion_score = sum(interaction_emotions.values()) / len(interaction_emotions) if interaction_emotions else 0.5

        # Emotional authenticity is the harmony between vocal and conversational emotions
        emotional_consistency = 1.0 - abs(vocal_emotion_score - interaction_emotion_score)

        # Overall emotional authenticity
        emotional_authenticity_score = (vocal_emotion_score + interaction_emotion_score + emotional_consistency) / 3

        indicators = []
        concerns = []

        if emotional_consistency > 0.8:
            indicators.append("Strong emotional consistency")
        elif emotional_consistency < 0.6:
            concerns.append("Emotional inconsistency detected")

        if vocal_emotion_score > 0.8:
            indicators.append("Authentic vocal emotional expression")
        elif vocal_emotion_score < 0.5:
            concerns.append("Limited vocal emotional range")

        return {
            "score": emotional_authenticity_score,
            "components": {
                "vocal_emotions": vocal_emotion_score,
                "interaction_emotions": interaction_emotion_score,
                "emotional_consistency": emotional_consistency
            },
            "indicators": indicators,
            "concerns": concerns
        }

    def _calculate_overall_authenticity(self, vocal_auth: dict[str, Any],
                                      conv_auth: dict[str, Any],
                                      emotional_auth: dict[str, Any]) -> dict[str, Any]:
        """Calculate overall authenticity score."""

        # Weighted combination
        overall_score = (
            vocal_auth["score"] * self.scoring_weights["vocal_authenticity"] +
            conv_auth["score"] * self.scoring_weights["conversational_authenticity"] +
            emotional_auth["score"] * self.scoring_weights["emotional_authenticity"]
        )

        # Combine indicators and concerns
        all_indicators = vocal_auth["indicators"] + conv_auth["indicators"] + emotional_auth["indicators"]
        all_concerns = vocal_auth["concerns"] + conv_auth["concerns"] + emotional_auth["concerns"]

        return {
            "score": overall_score,
            "indicators": all_indicators,
            "concerns": all_concerns
        }

    def _determine_confidence_level(self, overall_authenticity: dict[str, Any]) -> str:
        """Determine confidence level in authenticity assessment."""
        score = overall_authenticity["score"]
        concerns_count = len(overall_authenticity["concerns"])

        if score >= 0.8 and concerns_count <= 1:
            return "high"
        if score >= 0.6 and concerns_count <= 3:
            return "medium"
        return "low"

    def _generate_authenticity_recommendations(self, vocal_auth: dict[str, Any],
                                             conv_auth: dict[str, Any],
                                             emotional_auth: dict[str, Any]) -> list[str]:
        """Generate recommendations for improving authenticity."""
        recommendations = []

        if vocal_auth["score"] < 0.7:
            recommendations.append("Improve vocal naturalness and prosodic variation")

        if conv_auth["score"] < 0.7:
            recommendations.append("Enhance conversational flow and spontaneity")

        if emotional_auth["score"] < 0.7:
            recommendations.append("Strengthen emotional authenticity and consistency")

        # Check for specific concerns
        all_concerns = vocal_auth["concerns"] + conv_auth["concerns"] + emotional_auth["concerns"]
        if "Technical artifacts detected" in all_concerns:
            recommendations.append("Address technical artifacts in audio processing")

        if not recommendations:
            recommendations.append("Maintain current high authenticity standards")

        return recommendations

    def _calculate_batch_authenticity_statistics(self, assessments: list[VoiceAuthenticityAssessment]) -> dict[str, Any]:
        """Calculate statistics for batch authenticity assessment."""
        if not assessments:
            return {}

        authenticity_scores = [a.authenticity_metrics.overall_authenticity_score for a in assessments]

        return {
            "total_conversations": len(assessments),
            "average_authenticity": sum(authenticity_scores) / len(authenticity_scores),
            "authenticity_std": np.std(authenticity_scores),
            "min_authenticity": min(authenticity_scores),
            "max_authenticity": max(authenticity_scores),
            "high_authenticity_count": sum(1 for score in authenticity_scores if score >= 0.8),
            "medium_authenticity_count": sum(1 for score in authenticity_scores if 0.6 <= score < 0.8),
            "low_authenticity_count": sum(1 for score in authenticity_scores if score < 0.6),
            "confidence_distribution": {
                "high": sum(1 for a in assessments if a.confidence_level == "high"),
                "medium": sum(1 for a in assessments if a.confidence_level == "medium"),
                "low": sum(1 for a in assessments if a.confidence_level == "low")
            }
        }

    def _save_batch_authenticity_results(self, assessments: list[VoiceAuthenticityAssessment],
                                       batch_stats: dict[str, Any]) -> Path:
        """Save batch authenticity assessment results."""
        output_file = self.output_dir / f"voice_authenticity_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert assessments to serializable format
        assessments_data = []
        for assessment in assessments:
            assessment_dict = {
                "conversation_id": assessment.conversation_id,
                "voice_source_path": assessment.voice_source_path,
                "assessment_timestamp": assessment.assessment_timestamp,
                "authenticity_metrics": {
                    "vocal_authenticity_score": assessment.authenticity_metrics.vocal_authenticity_score,
                    "conversational_authenticity_score": assessment.authenticity_metrics.conversational_authenticity_score,
                    "emotional_authenticity_score": assessment.authenticity_metrics.emotional_authenticity_score,
                    "overall_authenticity_score": assessment.authenticity_metrics.overall_authenticity_score,
                    "authenticity_indicators": assessment.authenticity_metrics.authenticity_indicators,
                    "authenticity_concerns": assessment.authenticity_metrics.authenticity_concerns
                },
                "confidence_level": assessment.confidence_level,
                "recommendations": assessment.recommendations
            }
            assessments_data.append(assessment_dict)

        output_data = {
            "batch_info": {
                "assessment_type": "voice_derived_authenticity_batch",
                "processed_at": datetime.now().isoformat(),
                "scorer_version": "1.0"
            },
            "batch_statistics": batch_stats,
            "scoring_weights": self.scoring_weights,
            "authenticity_indicators": self.authenticity_indicators,
            "assessments": assessments_data
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Batch authenticity results saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize scorer
    scorer = VoiceDerivedAuthenticityScorer()

    # Mock voice conversation data
    mock_conversation = {
        "conversation_id": "test_voice_conv_001",
        "voice_source_path": "test_audio.wav",
        "transcription": "This is a sample therapeutic conversation...",
        "speaker_segments": [
            {"speaker": "client", "text": "I've been feeling anxious lately."},
            {"speaker": "therapist", "text": "Can you tell me more about that?"}
        ]
    }

    # Score authenticity
    assessment = scorer.score_voice_derived_authenticity(mock_conversation)


    # Batch scoring example
    mock_conversations = [mock_conversation] * 3
    batch_result = scorer.batch_score_authenticity(mock_conversations)

