#!/usr/bin/env python3
"""
Task 6.13: Audio Emotion Recognition Integration (IEMOCAP)

This module integrates audio emotion recognition with text-based therapeutic
conversations to create a comprehensive multi-modal analysis system.

Strategic Goal: Enhance therapeutic conversation analysis with audio emotional
cues for more accurate assessment and intervention recommendations.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
from condition_pattern_recognition import MentalHealthConditionRecognizer
from outcome_prediction import TherapeuticOutcomePredictor

# Import our existing components
from therapeutic_intelligence import TherapeuticApproachClassifier


class AudioEmotion(Enum):
    """Audio emotion categories from IEMOCAP dataset."""
    ANGER = "anger"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"


class EmotionIntensity(Enum):
    """Emotion intensity levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class AudioEmotionFeatures:
    """Audio emotion features extracted from speech."""
    dominant_emotion: AudioEmotion
    emotion_confidence: float
    intensity_level: EmotionIntensity
    emotion_distribution: dict[str, float]
    prosodic_features: dict[str, float]
    voice_quality_indicators: dict[str, float]
    emotional_stability: float  # 0.0 = highly variable, 1.0 = stable
    speech_rate: float
    pause_patterns: dict[str, float]


@dataclass
class MultiModalAnalysis:
    """Combined text and audio analysis result."""
    conversation_id: str
    text_analysis: dict[str, Any]
    audio_analysis: AudioEmotionFeatures
    emotion_text_alignment: float  # How well audio and text emotions align
    therapeutic_insights: list[str]
    risk_indicators: list[str]
    intervention_recommendations: list[str]
    overall_confidence: float
    analysis_timestamp: str


class AudioEmotionRecognizer:
    """Simulated audio emotion recognition system based on IEMOCAP patterns."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize emotion patterns based on IEMOCAP research
        self.emotion_patterns = self._initialize_emotion_patterns()
        self.prosodic_analyzers = self._initialize_prosodic_analyzers()

        # Performance tracking
        self.recognition_stats = {
            "total_analyzed": 0,
            "emotion_distribution": defaultdict(int),
            "intensity_distribution": defaultdict(int),
            "confidence_scores": []
        }

    def _initialize_emotion_patterns(self) -> dict[AudioEmotion, dict[str, Any]]:
        """Initialize emotion recognition patterns based on IEMOCAP research."""
        return {
            AudioEmotion.ANGER: {
                "pitch_range": (150, 400),  # Hz
                "intensity_range": (0.7, 1.0),
                "speech_rate_multiplier": 1.2,
                "voice_quality": {"harsh": 0.8, "tense": 0.9},
                "typical_words": ["angry", "frustrated", "mad", "furious", "irritated"]
            },
            AudioEmotion.SADNESS: {
                "pitch_range": (80, 200),
                "intensity_range": (0.2, 0.6),
                "speech_rate_multiplier": 0.7,
                "voice_quality": {"breathy": 0.7, "low_energy": 0.8},
                "typical_words": ["sad", "depressed", "hopeless", "empty", "down"]
            },
            AudioEmotion.FEAR: {
                "pitch_range": (200, 500),
                "intensity_range": (0.6, 0.9),
                "speech_rate_multiplier": 1.3,
                "voice_quality": {"tremulous": 0.8, "tense": 0.7},
                "typical_words": ["scared", "afraid", "terrified", "anxious", "worried"]
            },
            AudioEmotion.HAPPINESS: {
                "pitch_range": (180, 350),
                "intensity_range": (0.6, 0.9),
                "speech_rate_multiplier": 1.1,
                "voice_quality": {"bright": 0.8, "energetic": 0.9},
                "typical_words": ["happy", "good", "great", "wonderful", "excited"]
            },
            AudioEmotion.NEUTRAL: {
                "pitch_range": (120, 250),
                "intensity_range": (0.4, 0.7),
                "speech_rate_multiplier": 1.0,
                "voice_quality": {"balanced": 0.8, "steady": 0.7},
                "typical_words": ["okay", "fine", "normal", "usual", "regular"]
            },
            AudioEmotion.FRUSTRATION: {
                "pitch_range": (140, 380),
                "intensity_range": (0.6, 0.8),
                "speech_rate_multiplier": 1.1,
                "voice_quality": {"strained": 0.7, "effortful": 0.8},
                "typical_words": ["frustrated", "stuck", "difficult", "hard", "struggle"]
            }
        }

    def _initialize_prosodic_analyzers(self) -> dict[str, Any]:
        """Initialize prosodic feature analyzers."""
        return {
            "pitch_analyzer": {
                "fundamental_frequency": True,
                "pitch_contour": True,
                "pitch_variability": True
            },
            "intensity_analyzer": {
                "rms_energy": True,
                "peak_intensity": True,
                "intensity_variability": True
            },
            "temporal_analyzer": {
                "speech_rate": True,
                "pause_duration": True,
                "rhythm_patterns": True
            },
            "voice_quality_analyzer": {
                "jitter": True,
                "shimmer": True,
                "harmonics_to_noise_ratio": True
            }
        }

    def analyze_audio_emotion(self, conversation: dict[str, Any],
                            audio_data: dict[str, Any] | None = None) -> AudioEmotionFeatures:
        """Analyze audio emotion features from conversation (simulated for demo)."""
        conversation.get("id", "unknown")

        # Extract text for emotion correlation
        text_content = self._extract_conversation_text(conversation)

        # Simulate audio emotion recognition based on text patterns
        # In production, this would process actual audio files
        emotion_scores = self._simulate_emotion_recognition(text_content)

        # Determine dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        confidence = emotion_scores[dominant_emotion]

        # Determine intensity based on text indicators
        intensity = self._determine_emotion_intensity(text_content, dominant_emotion)

        # Generate prosodic features
        prosodic_features = self._generate_prosodic_features(dominant_emotion, intensity)

        # Generate voice quality indicators
        voice_quality = self._generate_voice_quality_indicators(dominant_emotion)

        # Calculate emotional stability
        emotional_stability = self._calculate_emotional_stability(text_content)

        # Generate speech patterns
        speech_rate = prosodic_features.get("speech_rate", 1.0)
        pause_patterns = self._generate_pause_patterns(dominant_emotion, intensity)

        features = AudioEmotionFeatures(
            dominant_emotion=dominant_emotion,
            emotion_confidence=confidence,
            intensity_level=intensity,
            emotion_distribution=emotion_scores,
            prosodic_features=prosodic_features,
            voice_quality_indicators=voice_quality,
            emotional_stability=emotional_stability,
            speech_rate=speech_rate,
            pause_patterns=pause_patterns
        )

        # Update statistics
        self._update_recognition_stats(features)

        return features

    def _extract_conversation_text(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation structure."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "input" in conversation and "output" in conversation:
            text_parts.extend([conversation["input"], conversation["output"]])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts).lower()

    def _simulate_emotion_recognition(self, text: str) -> dict[AudioEmotion, float]:
        """Simulate audio emotion recognition based on text patterns."""
        emotion_scores = dict.fromkeys(AudioEmotion, 0.0)

        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0

            # Check for typical words
            for word in patterns["typical_words"]:
                if word in text:
                    score += 0.3

            # Check for emotional intensity indicators
            intensity_indicators = {
                "very": 0.2, "extremely": 0.3, "really": 0.15,
                "so": 0.1, "completely": 0.25, "totally": 0.2
            }

            for indicator, weight in intensity_indicators.items():
                if indicator in text:
                    score += weight

            # Normalize and add baseline
            emotion_scores[emotion] = min(score + 0.1, 1.0)

        # Ensure scores sum to reasonable distribution
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            emotion_scores = {
                emotion: score / total_score
                for emotion, score in emotion_scores.items()
            }

        return emotion_scores

    def _determine_emotion_intensity(self, text: str, emotion: AudioEmotion) -> EmotionIntensity:
        """Determine emotion intensity based on text indicators."""
        high_intensity_indicators = [
            "extremely", "completely", "totally", "absolutely", "incredibly",
            "overwhelmingly", "devastatingly", "unbearably"
        ]

        moderate_intensity_indicators = [
            "very", "really", "quite", "pretty", "fairly", "rather"
        ]

        # Check for intensity indicators
        if any(indicator in text for indicator in high_intensity_indicators):
            return EmotionIntensity.HIGH
        if any(indicator in text for indicator in moderate_intensity_indicators):
            return EmotionIntensity.MODERATE
        return EmotionIntensity.LOW

    def _generate_prosodic_features(self, emotion: AudioEmotion,
                                  intensity: EmotionIntensity) -> dict[str, float]:
        """Generate prosodic features based on emotion and intensity."""
        base_patterns = self.emotion_patterns.get(emotion, {})

        # Base prosodic values
        pitch_range = base_patterns.get("pitch_range", (120, 250))
        intensity_range = base_patterns.get("intensity_range", (0.4, 0.7))
        speech_rate_mult = base_patterns.get("speech_rate_multiplier", 1.0)

        # Adjust for intensity
        intensity_multiplier = {
            EmotionIntensity.LOW: 0.8,
            EmotionIntensity.MODERATE: 1.0,
            EmotionIntensity.HIGH: 1.3,
            EmotionIntensity.EXTREME: 1.6
        }[intensity]

        return {
            "fundamental_frequency": (pitch_range[0] + pitch_range[1]) / 2,
            "pitch_variability": (pitch_range[1] - pitch_range[0]) * intensity_multiplier,
            "rms_energy": (intensity_range[0] + intensity_range[1]) / 2 * intensity_multiplier,
            "speech_rate": speech_rate_mult * intensity_multiplier,
            "pitch_contour_slope": np.random.normal(0, 0.1),  # Simulated
            "intensity_variability": intensity_multiplier * 0.2
        }

    def _generate_voice_quality_indicators(self, emotion: AudioEmotion) -> dict[str, float]:
        """Generate voice quality indicators based on emotion."""
        base_quality = self.emotion_patterns.get(emotion, {}).get("voice_quality", {})

        # Default voice quality features
        quality_features = {
            "jitter": 0.01,  # Voice stability
            "shimmer": 0.03,  # Amplitude stability
            "harmonics_to_noise_ratio": 15.0,  # Voice clarity
            "breathiness": 0.2,
            "roughness": 0.1,
            "strain": 0.1
        }

        # Adjust based on emotion-specific patterns
        for quality_type, weight in base_quality.items():
            if quality_type in quality_features:
                quality_features[quality_type] *= weight

        return quality_features

    def _calculate_emotional_stability(self, text: str) -> float:
        """Calculate emotional stability based on text patterns."""
        # Look for emotional variability indicators
        variability_indicators = [
            "sometimes", "other times", "one minute", "then suddenly",
            "back and forth", "up and down", "all over the place"
        ]

        stability_indicators = [
            "consistently", "always", "constantly", "steady", "stable"
        ]

        variability_count = sum(1 for indicator in variability_indicators if indicator in text)
        stability_count = sum(1 for indicator in stability_indicators if indicator in text)

        # Calculate stability score (0.0 = highly variable, 1.0 = stable)
        if variability_count > stability_count:
            return max(0.2, 1.0 - (variability_count * 0.2))
        return min(0.8 + (stability_count * 0.1), 1.0)

    def _generate_pause_patterns(self, emotion: AudioEmotion,
                               intensity: EmotionIntensity) -> dict[str, float]:
        """Generate pause patterns based on emotion and intensity."""
        base_patterns = {
            AudioEmotion.SADNESS: {"long_pauses": 0.8, "frequent_pauses": 0.6},
            AudioEmotion.ANGER: {"short_pauses": 0.9, "infrequent_pauses": 0.7},
            AudioEmotion.FEAR: {"irregular_pauses": 0.8, "hesitation_pauses": 0.9},
            AudioEmotion.HAPPINESS: {"rhythmic_pauses": 0.7, "brief_pauses": 0.8},
            AudioEmotion.NEUTRAL: {"regular_pauses": 0.8, "moderate_pauses": 0.7}
        }

        patterns = base_patterns.get(emotion, {"regular_pauses": 0.5})

        # Adjust for intensity
        intensity_factor = {
            EmotionIntensity.LOW: 0.8,
            EmotionIntensity.MODERATE: 1.0,
            EmotionIntensity.HIGH: 1.2,
            EmotionIntensity.EXTREME: 1.4
        }[intensity]

        return {
            pattern_type: value * intensity_factor
            for pattern_type, value in patterns.items()
        }

    def _update_recognition_stats(self, features: AudioEmotionFeatures):
        """Update recognition statistics."""
        self.recognition_stats["total_analyzed"] += 1
        self.recognition_stats["emotion_distribution"][features.dominant_emotion.value] += 1
        self.recognition_stats["intensity_distribution"][features.intensity_level.value] += 1
        self.recognition_stats["confidence_scores"].append(features.emotion_confidence)

    def get_recognition_statistics(self) -> dict[str, Any]:
        """Get audio emotion recognition statistics."""
        total = self.recognition_stats["total_analyzed"]
        if total == 0:
            return {}

        return {
            "total_analyzed": total,
            "emotion_distribution": dict(self.recognition_stats["emotion_distribution"]),
            "intensity_distribution": dict(self.recognition_stats["intensity_distribution"]),
            "average_confidence": statistics.mean(self.recognition_stats["confidence_scores"]) if self.recognition_stats["confidence_scores"] else 0
        }


class MultiModalTherapeuticAnalyzer:
    """Integrates text and audio analysis for comprehensive therapeutic assessment."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize component analyzers
        self.text_classifier = TherapeuticApproachClassifier()
        self.condition_recognizer = MentalHealthConditionRecognizer()
        self.outcome_predictor = TherapeuticOutcomePredictor()
        self.audio_recognizer = AudioEmotionRecognizer()

        # Analysis tracking
        self.analysis_stats = {
            "total_analyses": 0,
            "alignment_scores": [],
            "multi_modal_insights": 0
        }

    def analyze_conversation(self, conversation: dict[str, Any],
                           audio_data: dict[str, Any] | None = None) -> MultiModalAnalysis:
        """Perform comprehensive multi-modal analysis of a therapeutic conversation."""
        conversation_id = conversation.get("id", "unknown")

        # Step 1: Text-based analysis
        text_analysis = self._perform_text_analysis(conversation)

        # Step 2: Audio emotion analysis
        audio_analysis = self.audio_recognizer.analyze_audio_emotion(conversation, audio_data)

        # Step 3: Calculate emotion-text alignment
        alignment_score = self._calculate_emotion_text_alignment(text_analysis, audio_analysis)

        # Step 4: Generate therapeutic insights
        insights = self._generate_therapeutic_insights(text_analysis, audio_analysis, alignment_score)

        # Step 5: Identify risk indicators
        risk_indicators = self._identify_multimodal_risks(text_analysis, audio_analysis)

        # Step 6: Generate intervention recommendations
        interventions = self._generate_multimodal_interventions(text_analysis, audio_analysis, insights)

        # Step 7: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(text_analysis, audio_analysis, alignment_score)

        analysis = MultiModalAnalysis(
            conversation_id=conversation_id,
            text_analysis=text_analysis,
            audio_analysis=audio_analysis,
            emotion_text_alignment=alignment_score,
            therapeutic_insights=insights,
            risk_indicators=risk_indicators,
            intervention_recommendations=interventions,
            overall_confidence=overall_confidence,
            analysis_timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Update statistics
        self._update_analysis_stats(analysis)

        return analysis

    def _perform_text_analysis(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """Perform comprehensive text-based analysis."""
        # Therapeutic approach classification
        approach_classification = self.text_classifier.classify_conversation(conversation)

        # Mental health condition recognition
        condition_recognition = self.condition_recognizer.recognize_condition(conversation)

        # Outcome prediction
        outcome_prediction = self.outcome_predictor.predict_outcome(
            conversation, condition_recognition.primary_condition, "short_term"
        )

        return {
            "approach_classification": {
                "primary_approach": approach_classification.primary_approach.value,
                "quality_score": approach_classification.quality_score,
                "secondary_approaches": [app.value for app in approach_classification.secondary_approaches]
            },
            "condition_recognition": {
                "primary_condition": condition_recognition.primary_condition.value,
                "severity": condition_recognition.severity_assessment,
                "quality_score": condition_recognition.recognition_quality,
                "risk_indicators": condition_recognition.risk_indicators
            },
            "outcome_prediction": {
                "predicted_outcome": outcome_prediction.predicted_outcome.value,
                "confidence": outcome_prediction.confidence_score,
                "protective_factors": outcome_prediction.protective_factors,
                "risk_factors": outcome_prediction.risk_factors
            }
        }

    def _calculate_emotion_text_alignment(self, text_analysis: dict[str, Any],
                                        audio_analysis: AudioEmotionFeatures) -> float:
        """Calculate how well audio emotions align with text-based analysis."""
        # Map text-based indicators to expected audio emotions
        condition = text_analysis["condition_recognition"]["primary_condition"]
        severity = text_analysis["condition_recognition"]["severity"]

        expected_emotions = {
            "major_depressive_disorder": [AudioEmotion.SADNESS, AudioEmotion.NEUTRAL],
            "generalized_anxiety_disorder": [AudioEmotion.FEAR, AudioEmotion.FRUSTRATION],
            "post_traumatic_stress_disorder": [AudioEmotion.FEAR, AudioEmotion.SADNESS],
            "bipolar_disorder": [AudioEmotion.ANGER, AudioEmotion.HAPPINESS, AudioEmotion.SADNESS],
            "borderline_personality_disorder": [AudioEmotion.ANGER, AudioEmotion.SADNESS]
        }

        expected = expected_emotions.get(condition, [AudioEmotion.NEUTRAL])

        # Calculate alignment score
        base_alignment = 0.8 if audio_analysis.dominant_emotion in expected else 0.4

        # Adjust for severity
        severity_multiplier = {
            "mild": 0.9,
            "moderate": 1.0,
            "severe": 1.1
        }.get(severity, 1.0)

        # Adjust for emotion confidence
        confidence_factor = audio_analysis.emotion_confidence

        return min(base_alignment * severity_multiplier * confidence_factor, 1.0)

    def _generate_therapeutic_insights(self, text_analysis: dict[str, Any],
                                     audio_analysis: AudioEmotionFeatures,
                                     alignment_score: float) -> list[str]:
        """Generate therapeutic insights from multi-modal analysis."""
        insights = []

        # Emotion-text alignment insights
        if alignment_score > 0.8:
            insights.append("Strong congruence between expressed emotions and audio cues")
        elif alignment_score < 0.5:
            insights.append("Potential emotional incongruence - client may be masking feelings")

        # Audio-specific insights
        if audio_analysis.emotional_stability < 0.4:
            insights.append("High emotional variability detected in speech patterns")

        if audio_analysis.intensity_level == EmotionIntensity.HIGH:
            insights.append("High emotional intensity requires immediate attention")

        # Voice quality insights
        voice_quality = audio_analysis.voice_quality_indicators
        if voice_quality.get("strain", 0) > 0.7:
            insights.append("Voice strain indicates significant emotional distress")

        if voice_quality.get("breathiness", 0) > 0.6:
            insights.append("Breathy voice quality may indicate anxiety or sadness")

        return insights

    def _identify_multimodal_risks(self, text_analysis: dict[str, Any],
                                 audio_analysis: AudioEmotionFeatures) -> list[str]:
        """Identify risk indicators from multi-modal analysis."""
        risks = []

        # Include text-based risks
        text_risks = text_analysis["condition_recognition"]["risk_indicators"]
        risks.extend(text_risks)

        # Add audio-based risks
        if audio_analysis.dominant_emotion == AudioEmotion.ANGER and audio_analysis.intensity_level == EmotionIntensity.HIGH:
            risks.append("high_anger_intensity")

        if audio_analysis.emotional_stability < 0.3:
            risks.append("emotional_dysregulation")

        if audio_analysis.voice_quality_indicators.get("strain", 0) > 0.8:
            risks.append("severe_emotional_distress")

        return list(set(risks))  # Remove duplicates

    def _generate_multimodal_interventions(self, text_analysis: dict[str, Any],
                                         audio_analysis: AudioEmotionFeatures,
                                         insights: list[str]) -> list[str]:
        """Generate intervention recommendations based on multi-modal analysis."""
        interventions = []

        # Audio-informed interventions
        if audio_analysis.dominant_emotion == AudioEmotion.ANGER:
            interventions.append("Anger management techniques")
            interventions.append("Voice modulation exercises")

        if audio_analysis.emotional_stability < 0.5:
            interventions.append("Emotional regulation skills training")
            interventions.append("Mindfulness-based interventions")

        if "emotional incongruence" in " ".join(insights):
            interventions.append("Explore discrepancy between expressed and felt emotions")

        # Voice quality interventions
        voice_quality = audio_analysis.voice_quality_indicators
        if voice_quality.get("strain", 0) > 0.6:
            interventions.append("Relaxation and breathing exercises")

        return interventions[:5]  # Limit to top 5

    def _calculate_overall_confidence(self, text_analysis: dict[str, Any],
                                    audio_analysis: AudioEmotionFeatures,
                                    alignment_score: float) -> float:
        """Calculate overall confidence in the multi-modal analysis."""
        # Component confidences
        text_confidence = (
            text_analysis["approach_classification"]["quality_score"] +
            text_analysis["condition_recognition"]["quality_score"] +
            text_analysis["outcome_prediction"]["confidence"]
        ) / 3

        audio_confidence = audio_analysis.emotion_confidence

        # Weighted combination
        overall_confidence = (
            text_confidence * 0.5 +
            audio_confidence * 0.3 +
            alignment_score * 0.2
        )

        return min(overall_confidence, 1.0)

    def _update_analysis_stats(self, analysis: MultiModalAnalysis):
        """Update analysis statistics."""
        self.analysis_stats["total_analyses"] += 1
        self.analysis_stats["alignment_scores"].append(analysis.emotion_text_alignment)
        self.analysis_stats["multi_modal_insights"] += len(analysis.therapeutic_insights)

    def get_analysis_statistics(self) -> dict[str, Any]:
        """Get multi-modal analysis statistics."""
        total = self.analysis_stats["total_analyses"]
        if total == 0:
            return {}

        return {
            "total_analyses": total,
            "average_alignment_score": statistics.mean(self.analysis_stats["alignment_scores"]) if self.analysis_stats["alignment_scores"] else 0,
            "average_insights_per_analysis": self.analysis_stats["multi_modal_insights"] / total,
            "audio_emotion_stats": self.audio_recognizer.get_recognition_statistics()
        }


# Example usage and testing
def main():
    """Example usage of the audio emotion integration system."""

    # Create multi-modal analyzer
    analyzer = MultiModalTherapeuticAnalyzer()

    # Example conversation with emotional content
    test_conversation = {
        "id": "multimodal_test",
        "messages": [
            {
                "role": "client",
                "content": "I feel so angry and frustrated! Nothing is working and I just want to scream. My voice is shaking and I can barely control myself."
            },
            {
                "role": "therapist",
                "content": "I can hear the intensity in your voice and see how much distress you're experiencing right now. Let's work on some breathing techniques to help you feel more grounded."
            }
        ]
    }


    # Perform multi-modal analysis
    analysis = analyzer.analyze_conversation(test_conversation)

    # Display results



    for _insight in analysis.therapeutic_insights:
        pass

    for _risk in analysis.risk_indicators:
        pass

    for _intervention in analysis.intervention_recommendations:
        pass

    # Show statistics
    analyzer.get_analysis_statistics()


if __name__ == "__main__":
    main()
