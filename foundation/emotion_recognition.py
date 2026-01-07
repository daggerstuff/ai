"""
Emotion Recognition Foundation for Therapeutic Conversations.

Capabilities:
- Detect emotional states from text (valence, arousal)
- Therapeutic emotion tracking
- Crisis signal detection
- Emotion trajectory analysis
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class EmotionValence(str, Enum):
    """Emotional valence (positive to negative)."""

    VERY_POSITIVE = "very_positive"  # 0.8-1.0
    POSITIVE = "positive"  # 0.6-0.8
    NEUTRAL = "neutral"  # 0.4-0.6
    NEGATIVE = "negative"  # 0.2-0.4
    VERY_NEGATIVE = "very_negative"  # 0.0-0.2


class EmotionArousal(str, Enum):
    """Emotional arousal (calm to intense)."""

    VERY_CALM = "very_calm"  # 0.0-0.2
    CALM = "calm"  # 0.2-0.4
    MODERATE = "moderate"  # 0.4-0.6
    ACTIVATED = "activated"  # 0.6-0.8
    HIGHLY_ACTIVATED = "highly_activated"  # 0.8-1.0


@dataclass
class EmotionState:
    """Complete emotional state measurement."""

    valence: float  # 0.0 (very negative) to 1.0 (very positive)
    arousal: float  # 0.0 (calm) to 1.0 (activated/intense)
    primary_emotion: Optional[str] = None  # e.g., "anxiety", "sadness", "hope"
    confidence: float = 1.0

    @property
    def valence_category(self) -> EmotionValence:
        """Classify valence into category."""
        if self.valence >= 0.8:
            return EmotionValence.VERY_POSITIVE
        elif self.valence >= 0.6:
            return EmotionValence.POSITIVE
        elif self.valence >= 0.4:
            return EmotionValence.NEUTRAL
        elif self.valence >= 0.2:
            return EmotionValence.NEGATIVE
        else:
            return EmotionValence.VERY_NEGATIVE

    @property
    def arousal_category(self) -> EmotionArousal:
        """Classify arousal into category."""
        if self.arousal >= 0.8:
            return EmotionArousal.HIGHLY_ACTIVATED
        elif self.arousal >= 0.6:
            return EmotionArousal.ACTIVATED
        elif self.arousal >= 0.4:
            return EmotionArousal.MODERATE
        elif self.arousal >= 0.2:
            return EmotionArousal.CALM
        else:
            return EmotionArousal.VERY_CALM


class CrisisSignalDetector:
    """Detect crisis indicators in therapeutic conversations."""

    CRISIS_KEYWORDS = [
        # Suicidal ideation
        "suicide",
        "kill myself",
        "end it all",
        "not worth living",
        "hopeless",
        "can't go on",
        # Self-harm
        "cut myself",
        "hurt myself",
        "harm",
        # Acute distress
        "overwhelming",
        "can't breathe",
        "panicking",
        "losing control",
    ]

    @classmethod
    def detect_crisis_signals(cls, text: str) -> Tuple[bool, List[str], float]:
        """
        Detect crisis signals in text.

        Returns:
            (has_crisis_signal, matched_keywords, confidence)
        """
        if matched := [kw for kw in cls.CRISIS_KEYWORDS if kw in text.lower()]:
            # Higher confidence with more keywords
            return True, matched, min(len(matched) / 3, 1.0)

        return False, [], 0.0


class EmotionRecognizer:
    """Recognizes emotional states in therapeutic dialogue."""

    def __init__(self):
        # Simple heuristics for demo; real implementation uses transformers
        self.positive_words = [
            "better",
            "hopeful",
            "improving",
            "progress",
            "grateful",
            "happy",
            "relieved",
            "confident",
        ]
        self.negative_words = [
            "worse",
            "anxious",
            "depressed",
            "scared",
            "angry",
            "frustrated",
            "overwhelmed",
            "stuck",
        ]
        self.arousal_words = {
            "calm": ["calm", "peaceful", "relaxed", "serene"],
            "activated": ["energetic", "alert", "motivated", "driven"],
            "intense": ["frantic", "panicked", "rushed", "desperate"],
        }

    def recognize_emotion(self, text: str) -> EmotionState:
        """Recognize emotional state from text."""
        text_lower = text.lower()

        # Simple valence scoring
        positive_count = sum(word in text_lower for word in self.positive_words)
        negative_count = sum(word in text_lower for word in self.negative_words)

        # Calculate valence (0-1)
        total = positive_count + negative_count
        valence = (
            0.5 if total == 0 else positive_count / total
        )  # Neutral if no emotion words

        # Simple arousal scoring
        calm_count = sum(word in text_lower for word in self.arousal_words["calm"])
        activated_count = sum(
            word in text_lower for word in self.arousal_words["activated"]
        )
        intense_count = sum(
            word in text_lower for word in self.arousal_words["intense"]
        )

        arousal_total = calm_count + activated_count + intense_count
        arousal = (
            0.5
            if arousal_total == 0
            else min((activated_count + 2 * intense_count) / (2 * arousal_total), 1.0)
        )  # Heavy weight on intense

        # Detect crisis signals
        has_crisis, matched_keywords, crisis_confidence = (
            CrisisSignalDetector.detect_crisis_signals(text)
        )

        primary_emotion = "crisis" if has_crisis and crisis_confidence > 0.5 else None
        confidence = min(
            0.4 + (total * 0.1), 1.0
        )  # Low confidence with simple heuristics

        return EmotionState(
            valence=valence,
            arousal=arousal,
            primary_emotion=primary_emotion,
            confidence=confidence,
        )

    def track_emotion_trajectory(self, texts: List[str]) -> List[EmotionState]:
        """Track how emotions change across conversation turns."""
        trajectory = []
        for text in texts:
            emotion = self.recognize_emotion(text)
            trajectory.append(emotion)
        return trajectory


if __name__ == "__main__":
    recognizer = EmotionRecognizer()

    # Example patient statements
    examples = [
        "I'm feeling really anxious and overwhelmed right now.",
        "Things are getting a bit better. I'm cautiously hopeful.",
        "I can't go on like this anymore. I want to end it all.",
    ]

    print("=== Emotion Recognition Examples ===\n")
    for text in examples:
        emotion = recognizer.recognize_emotion(text)
        has_crisis, keywords, conf = CrisisSignalDetector.detect_crisis_signals(text)

        print(f"Text: {text}")
        print(f"Valence: {emotion.valence:.2f} ({emotion.valence_category.value})")
        print(f"Arousal: {emotion.arousal:.2f} ({emotion.arousal_category.value})")
        if has_crisis:
            print(f"⚠️  CRISIS SIGNAL (confidence: {conf:.2%})")
            print(f"   Keywords: {keywords}")
        print()
