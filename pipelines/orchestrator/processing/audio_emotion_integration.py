#!/usr/bin/env python3
"""
Audio Emotion Recognition Integration for Task 6.13
Integrates audio emotion recognition with conversation analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"


class AudioFeature(Enum):
    """Audio features for emotion recognition."""
    PITCH = "pitch"
    TONE = "tone"
    VOLUME = "volume"
    PACE = "pace"
    PAUSES = "pauses"
    VOICE_QUALITY = "voice_quality"
    PROSODY = "prosody"


@dataclass
class AudioEmotionData:
    """Audio emotion recognition data."""
    timestamp: float
    emotion: EmotionCategory
    confidence: float
    intensity: float
    audio_features: dict[AudioFeature, float]
    duration: float


@dataclass
class EmotionAnalysis:
    """Complete emotion analysis result."""
    conversation_id: str
    audio_emotions: list[AudioEmotionData]
    text_emotions: list[dict[str, Any]]
    emotion_alignment: float
    dominant_emotion: EmotionCategory
    emotion_trajectory: list[tuple[float, EmotionCategory]]
    emotional_coherence: float
    analysis_confidence: float
    insights: list[str]


class AudioEmotionIntegration:
    """
    Audio emotion recognition integration system.
    """

    def __init__(self):
        """Initialize audio emotion integration."""
        self.emotion_keywords = self._load_emotion_keywords()
        self.audio_emotion_patterns = self._load_audio_patterns()
        self.emotion_transitions = self._load_emotion_transitions()

        logger.info("AudioEmotionIntegration initialized")

    def _load_emotion_keywords(self) -> dict[EmotionCategory, list[str]]:
        """Load emotion keywords for text analysis."""
        return {
            EmotionCategory.JOY: [
                "happy", "joy", "excited", "pleased", "delighted", "cheerful",
                "elated", "euphoric", "content", "satisfied", "glad", "thrilled"
            ],
            EmotionCategory.SADNESS: [
                "sad", "depressed", "down", "blue", "melancholy", "sorrowful",
                "dejected", "despondent", "gloomy", "mournful", "grief", "despair"
            ],
            EmotionCategory.ANGER: [
                "angry", "mad", "furious", "irritated", "annoyed", "rage",
                "frustrated", "livid", "outraged", "hostile", "resentful", "bitter"
            ],
            EmotionCategory.FEAR: [
                "afraid", "scared", "fearful", "terrified", "anxious", "worried",
                "nervous", "apprehensive", "panicked", "frightened", "alarmed"
            ],
            EmotionCategory.SURPRISE: [
                "surprised", "shocked", "amazed", "astonished", "stunned",
                "bewildered", "startled", "taken aback", "unexpected"
            ],
            EmotionCategory.ANXIETY: [
                "anxious", "worried", "nervous", "stressed", "tense", "uneasy",
                "restless", "agitated", "overwhelmed", "panic", "dread"
            ],
            EmotionCategory.EXCITEMENT: [
                "excited", "thrilled", "enthusiastic", "energetic", "pumped",
                "eager", "animated", "exhilarated", "stimulated"
            ],
            EmotionCategory.FRUSTRATION: [
                "frustrated", "annoyed", "irritated", "exasperated", "fed up",
                "aggravated", "vexed", "bothered", "irked"
            ]
        }

    def _load_audio_patterns(self) -> dict[EmotionCategory, dict[AudioFeature, tuple[float, float]]]:
        """Load audio feature patterns for emotions (mock data)."""
        return {
            EmotionCategory.JOY: {
                AudioFeature.PITCH: (0.6, 0.8),  # Higher pitch
                AudioFeature.VOLUME: (0.7, 0.9),  # Louder
                AudioFeature.PACE: (0.6, 0.8),    # Faster pace
                AudioFeature.TONE: (0.7, 0.9)     # Brighter tone
            },
            EmotionCategory.SADNESS: {
                AudioFeature.PITCH: (0.2, 0.4),   # Lower pitch
                AudioFeature.VOLUME: (0.3, 0.5),  # Quieter
                AudioFeature.PACE: (0.2, 0.4),    # Slower pace
                AudioFeature.TONE: (0.2, 0.4)     # Duller tone
            },
            EmotionCategory.ANGER: {
                AudioFeature.PITCH: (0.7, 0.9),   # Higher pitch
                AudioFeature.VOLUME: (0.8, 1.0),  # Loudest
                AudioFeature.PACE: (0.7, 0.9),    # Faster pace
                AudioFeature.TONE: (0.6, 0.8)     # Harsh tone
            },
            EmotionCategory.FEAR: {
                AudioFeature.PITCH: (0.6, 0.8),   # Higher pitch
                AudioFeature.VOLUME: (0.4, 0.6),  # Variable volume
                AudioFeature.PACE: (0.7, 0.9),    # Faster pace
                AudioFeature.TONE: (0.3, 0.5)     # Tense tone
            },
            EmotionCategory.ANXIETY: {
                AudioFeature.PITCH: (0.5, 0.7),   # Slightly higher
                AudioFeature.VOLUME: (0.4, 0.6),  # Variable
                AudioFeature.PACE: (0.6, 0.8),    # Faster
                AudioFeature.PAUSES: (0.3, 0.5)   # More pauses
            }
        }

    def _load_emotion_transitions(self) -> dict[tuple[EmotionCategory, EmotionCategory], float]:
        """Load emotion transition probabilities."""
        return {
            (EmotionCategory.SADNESS, EmotionCategory.JOY): 0.3,
            (EmotionCategory.ANGER, EmotionCategory.FRUSTRATION): 0.8,
            (EmotionCategory.FEAR, EmotionCategory.ANXIETY): 0.9,
            (EmotionCategory.ANXIETY, EmotionCategory.FEAR): 0.7,
            (EmotionCategory.NEUTRAL, EmotionCategory.JOY): 0.5,
            (EmotionCategory.NEUTRAL, EmotionCategory.SADNESS): 0.4,
            (EmotionCategory.FRUSTRATION, EmotionCategory.ANGER): 0.6,
            (EmotionCategory.EXCITEMENT, EmotionCategory.JOY): 0.8
        }

    def analyze_conversation_emotions(self, conversation: dict[str, Any]) -> EmotionAnalysis:
        """Analyze emotions in conversation with audio integration."""
        conversation_id = conversation.get("id", "unknown")

        # Extract text content
        text_content = self._extract_text_content(conversation)

        # Analyze text emotions
        text_emotions = self._analyze_text_emotions(text_content)

        # Simulate audio emotion analysis (in real implementation, this would use actual audio)
        audio_emotions = self._simulate_audio_emotion_analysis(conversation, text_emotions)

        # Calculate emotion alignment between text and audio
        emotion_alignment = self._calculate_emotion_alignment(text_emotions, audio_emotions)

        # Determine dominant emotion
        dominant_emotion = self._determine_dominant_emotion(text_emotions, audio_emotions)

        # Create emotion trajectory
        emotion_trajectory = self._create_emotion_trajectory(audio_emotions)

        # Calculate emotional coherence
        emotional_coherence = self._calculate_emotional_coherence(emotion_trajectory)

        # Calculate analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(
            text_emotions, audio_emotions, emotion_alignment
        )

        # Generate insights
        insights = self._generate_emotion_insights(
            text_emotions, audio_emotions, dominant_emotion, emotion_alignment
        )

        analysis = EmotionAnalysis(
            conversation_id=conversation_id,
            audio_emotions=audio_emotions,
            text_emotions=text_emotions,
            emotion_alignment=emotion_alignment,
            dominant_emotion=dominant_emotion,
            emotion_trajectory=emotion_trajectory,
            emotional_coherence=emotional_coherence,
            analysis_confidence=analysis_confidence,
            insights=insights
        )

        logger.info(f"Emotion analysis completed for {conversation_id}: dominant emotion {dominant_emotion.value}")
        return analysis

    def _extract_text_content(self, conversation: dict[str, Any]) -> str:
        """Extract text content from conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)
        elif isinstance(content, dict) and "turns" in conversation:
            turns = conversation["turns"]
            content = " ".join(turn.get("content", "") for turn in turns)
        return content

    def _analyze_text_emotions(self, text_content: str) -> list[dict[str, Any]]:
        """Analyze emotions in text content."""
        text_emotions = []

        if not text_content:
            return text_emotions

        # Split into sentences for granular analysis
        sentences = [s.strip() for s in text_content.split(".") if s.strip()]

        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            emotion_scores = {}

            # Score each emotion category
            for emotion, keywords in self.emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in sentence_lower)
                if score > 0:
                    emotion_scores[emotion] = min(1.0, score / len(keywords))

            # Determine primary emotion for sentence
            if emotion_scores:
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[primary_emotion]

                text_emotions.append({
                    "sentence_index": i,
                    "sentence": sentence,
                    "emotion": primary_emotion,
                    "confidence": confidence,
                    "all_scores": emotion_scores
                })

        return text_emotions

    def _simulate_audio_emotion_analysis(self, conversation: dict[str, Any],
                                       text_emotions: list[dict[str, Any]]) -> list[AudioEmotionData]:
        """Simulate audio emotion analysis (mock implementation)."""
        audio_emotions = []

        # In real implementation, this would process actual audio files
        # For simulation, we'll generate plausible audio emotion data

        duration_per_segment = 3.0  # 3 seconds per segment

        for i, text_emotion in enumerate(text_emotions):
            timestamp = i * duration_per_segment

            # Use text emotion as base, add some variation for audio
            base_emotion = text_emotion["emotion"]
            base_confidence = text_emotion["confidence"]

            # Simulate audio features based on emotion
            audio_features = self._generate_audio_features(base_emotion)

            # Add some noise/variation to confidence
            import random
            audio_confidence = min(1.0, max(0.1, base_confidence + random.uniform(-0.2, 0.2)))
            intensity = random.uniform(0.3, 0.9)

            audio_emotion = AudioEmotionData(
                timestamp=timestamp,
                emotion=base_emotion,
                confidence=audio_confidence,
                intensity=intensity,
                audio_features=audio_features,
                duration=duration_per_segment
            )

            audio_emotions.append(audio_emotion)

        return audio_emotions

    def _generate_audio_features(self, emotion: EmotionCategory) -> dict[AudioFeature, float]:
        """Generate audio features for given emotion."""
        import random

        features = {}

        if emotion in self.audio_emotion_patterns:
            patterns = self.audio_emotion_patterns[emotion]

            for feature, (min_val, max_val) in patterns.items():
                # Generate value within expected range with some noise
                value = random.uniform(min_val, max_val)
                features[feature] = value

        # Fill in missing features with neutral values
        for feature in AudioFeature:
            if feature not in features:
                features[feature] = random.uniform(0.4, 0.6)  # Neutral range

        return features

    def _calculate_emotion_alignment(self, text_emotions: list[dict[str, Any]],
                                   audio_emotions: list[AudioEmotionData]) -> float:
        """Calculate alignment between text and audio emotions."""
        if not text_emotions or not audio_emotions:
            return 0.0

        alignments = []

        # Compare emotions at similar time points
        min_length = min(len(text_emotions), len(audio_emotions))

        for i in range(min_length):
            text_emotion = text_emotions[i]["emotion"]
            audio_emotion = audio_emotions[i].emotion

            if text_emotion == audio_emotion:
                alignments.append(1.0)
            elif self._are_emotions_related(text_emotion, audio_emotion):
                alignments.append(0.7)  # Partial alignment
            else:
                alignments.append(0.0)

        return sum(alignments) / len(alignments) if alignments else 0.0

    def _are_emotions_related(self, emotion1: EmotionCategory, emotion2: EmotionCategory) -> bool:
        """Check if two emotions are related."""
        related_groups = [
            {EmotionCategory.SADNESS, EmotionCategory.FEAR, EmotionCategory.ANXIETY},
            {EmotionCategory.ANGER, EmotionCategory.FRUSTRATION},
            {EmotionCategory.JOY, EmotionCategory.EXCITEMENT},
            {EmotionCategory.FEAR, EmotionCategory.ANXIETY}
        ]

        return any(emotion1 in group and emotion2 in group for group in related_groups)

    def _determine_dominant_emotion(self, text_emotions: list[dict[str, Any]],
                                  audio_emotions: list[AudioEmotionData]) -> EmotionCategory:
        """Determine dominant emotion across conversation."""
        emotion_counts = {}

        # Count text emotions
        for text_emotion in text_emotions:
            emotion = text_emotion["emotion"]
            confidence = text_emotion["confidence"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + confidence

        # Count audio emotions (weighted by confidence)
        for audio_emotion in audio_emotions:
            emotion = audio_emotion.emotion
            confidence = audio_emotion.confidence
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + confidence

        if emotion_counts:
            return max(emotion_counts, key=emotion_counts.get)
        return EmotionCategory.NEUTRAL

    def _create_emotion_trajectory(self, audio_emotions: list[AudioEmotionData]) -> list[tuple[float, EmotionCategory]]:
        """Create emotion trajectory over time."""
        trajectory = []

        for audio_emotion in audio_emotions:
            trajectory.append((audio_emotion.timestamp, audio_emotion.emotion))

        return trajectory

    def _calculate_emotional_coherence(self, trajectory: list[tuple[float, EmotionCategory]]) -> float:
        """Calculate emotional coherence of trajectory."""
        if len(trajectory) < 2:
            return 1.0

        coherence_scores = []

        for i in range(len(trajectory) - 1):
            current_emotion = trajectory[i][1]
            next_emotion = trajectory[i + 1][1]

            if current_emotion == next_emotion:
                coherence_scores.append(1.0)  # Same emotion = high coherence
            else:
                # Check transition probability
                transition_key = (current_emotion, next_emotion)
                if transition_key in self.emotion_transitions:
                    coherence_scores.append(self.emotion_transitions[transition_key])
                else:
                    coherence_scores.append(0.3)  # Low coherence for unexpected transitions

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0

    def _calculate_analysis_confidence(self, text_emotions: list[dict[str, Any]],
                                     audio_emotions: list[AudioEmotionData],
                                     emotion_alignment: float) -> float:
        """Calculate overall analysis confidence."""
        confidence_factors = []

        # Text emotion confidence
        if text_emotions:
            avg_text_confidence = sum(e["confidence"] for e in text_emotions) / len(text_emotions)
            confidence_factors.append(avg_text_confidence)

        # Audio emotion confidence
        if audio_emotions:
            avg_audio_confidence = sum(e.confidence for e in audio_emotions) / len(audio_emotions)
            confidence_factors.append(avg_audio_confidence)

        # Alignment factor
        confidence_factors.append(emotion_alignment)

        # Data availability factor
        data_factor = min(1.0, (len(text_emotions) + len(audio_emotions)) / 10)
        confidence_factors.append(data_factor)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

    def _generate_emotion_insights(self, text_emotions: list[dict[str, Any]],
                                 audio_emotions: list[AudioEmotionData],
                                 dominant_emotion: EmotionCategory,
                                 emotion_alignment: float) -> list[str]:
        """Generate insights about emotional content."""
        insights = []

        # Dominant emotion insight
        insights.append(f"Dominant emotion throughout conversation: {dominant_emotion.value}")

        # Alignment insight
        if emotion_alignment > 0.8:
            insights.append("High alignment between text and audio emotions - consistent emotional expression")
        elif emotion_alignment > 0.5:
            insights.append("Moderate alignment between text and audio emotions - some emotional complexity")
        else:
            insights.append("Low alignment between text and audio emotions - potential emotional masking or complexity")

        # Emotion variety insight
        unique_emotions = set()
        for text_emotion in text_emotions:
            unique_emotions.add(text_emotion["emotion"])
        for audio_emotion in audio_emotions:
            unique_emotions.add(audio_emotion.emotion)

        if len(unique_emotions) > 4:
            insights.append("High emotional variety - complex emotional landscape")
        elif len(unique_emotions) > 2:
            insights.append("Moderate emotional variety - some emotional shifts")
        else:
            insights.append("Low emotional variety - consistent emotional state")

        # Intensity insights
        if audio_emotions:
            avg_intensity = sum(e.intensity for e in audio_emotions) / len(audio_emotions)
            if avg_intensity > 0.7:
                insights.append("High emotional intensity detected in audio")
            elif avg_intensity < 0.4:
                insights.append("Low emotional intensity detected in audio")

        return insights

    def get_emotion_summary(self, analyses: list[EmotionAnalysis]) -> dict[str, Any]:
        """Get summary of emotion analyses."""
        if not analyses:
            return {"status": "empty"}

        # Emotion distribution
        emotion_counts = {}
        for analysis in analyses:
            emotion = analysis.dominant_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Average metrics
        avg_alignment = sum(a.emotion_alignment for a in analyses) / len(analyses)
        avg_coherence = sum(a.emotional_coherence for a in analyses) / len(analyses)
        avg_confidence = sum(a.analysis_confidence for a in analyses) / len(analyses)

        # Common insights
        all_insights = []
        for analysis in analyses:
            all_insights.extend(analysis.insights)

        insight_counts = {}
        for insight in all_insights:
            insight_counts[insight] = insight_counts.get(insight, 0) + 1

        common_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_analyses": len(analyses),
            "emotion_distribution": emotion_counts,
            "average_alignment": avg_alignment,
            "average_coherence": avg_coherence,
            "average_confidence": avg_confidence,
            "common_insights": common_insights,
            "high_alignment_rate": len([a for a in analyses if a.emotion_alignment > 0.7]) / len(analyses)
        }


def main():
    """Test the audio emotion integration system."""
    integration = AudioEmotionIntegration()

    # Test conversations
    test_conversations = [
        {
            "id": "conv_1",
            "content": "I'm feeling really anxious about my presentation tomorrow. I'm worried I'll mess up and everyone will judge me. This fear is overwhelming me."
        },
        {
            "id": "conv_2",
            "content": "I'm so excited about my new job! I can't wait to start. This is such a happy moment for me. I feel thrilled and energetic."
        },
        {
            "id": "conv_3",
            "content": "I've been feeling sad lately. Everything seems difficult and I don't have much energy. I feel down and discouraged most days."
        }
    ]


    # Analyze conversations
    analyses = []
    for conv in test_conversations:
        analysis = integration.analyze_conversation_emotions(conv)
        analyses.append(analysis)


        if analysis.insights:
            for _insight in analysis.insights[:3]:
                pass

    # Generate summary
    integration.get_emotion_summary(analyses)



if __name__ == "__main__":
    main()


class MultiModalTherapeuticAnalyzer:
    """
    Multi-modal therapeutic analyzer for comprehensive emotion analysis.
    """

    def __init__(self):
        """Initialize the multi-modal analyzer."""
        self.audio_integration = AudioEmotionIntegration()
        logger.info("MultiModalTherapeuticAnalyzer initialized")

    def analyze_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        """Analyze conversation using multi-modal approach."""
        try:
            # Use the existing audio emotion integration
            analysis = self.audio_integration.analyze_conversation(conversation)

            return {
                "conversation_id": conversation.get("id", "unknown"),
                "emotion_analysis": analysis,
                "multimodal_score": analysis.alignment_score,
                "therapeutic_indicators": self._extract_therapeutic_indicators(analysis),
                "recommendations": self._generate_recommendations(analysis)
            }

        except Exception as e:
            logger.error(f"Error in multi-modal analysis: {e}")
            return {
                "conversation_id": conversation.get("id", "unknown"),
                "error": str(e),
                "multimodal_score": 0.0
            }

    def _extract_therapeutic_indicators(self, analysis) -> list[str]:
        """Extract therapeutic indicators from analysis."""
        indicators = []

        if analysis.alignment_score > 0.8:
            indicators.append("High emotional alignment")
        if analysis.coherence_score > 0.7:
            indicators.append("Good emotional coherence")
        if len(analysis.insights) > 3:
            indicators.append("Rich emotional insights")

        return indicators

    def _generate_recommendations(self, analysis) -> list[str]:
        """Generate therapeutic recommendations."""
        recommendations = []

        if analysis.alignment_score < 0.5:
            recommendations.append("Focus on emotional validation")
        if analysis.coherence_score < 0.6:
            recommendations.append("Improve emotional consistency")
        if not analysis.insights:
            recommendations.append("Enhance emotional depth")

        return recommendations


# Alias for compatibility
AudioEmotionFeatures = AudioEmotionData
