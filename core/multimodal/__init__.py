"""
Multimodal Processing for Pixel Therapeutic Conversations.

Enables real-time processing of audio and text modalities with emotion fusion.

Modules:
  - speech_recognition: Audio-to-text conversion (Whisper, Wav2Vec2)
  - audio_emotion_recognition: Emotion detection from speech (valence/arousal)
  - multimodal_fusion: Combine text and audio signals for enhanced understanding
  - text_to_speech: Generate speech with emotional prosody (future)

Usage:
    >>> from ai.multimodal import SpeechRecognizer, AudioEmotionRecognizer
    >>>
    >>> # Speech recognition
    >>> recognizer = SpeechRecognizer(model_name="base")
    >>> transcript = await recognizer.transcribe_audio("session_001", "audio.wav")
    >>>
    >>> # Audio emotion
    >>> emotion_recognizer = AudioEmotionRecognizer()
    >>> emotions = await emotion_recognizer.detect_emotions("session_001", "audio.wav")
    >>>
    >>> # Multimodal fusion
    >>> from ai.multimodal import MultimodalFusion
    >>> fusion = MultimodalFusion(text_weight=0.6, audio_weight=0.4)
    >>> fused = fusion.fuse_emotions(
    ...     text_emotion=pixel_output,
    ...     audio_emotion=emotions.overall_emotion.__dict__
    ... )
"""

from .audio_emotion_recognition import (
    AudioEmotionRecognizer,
    AudioEmotionResult,
    AudioPreprocessor,
    EmotionalState,
    EmotionTrajectory,
)
from .multimodal_fusion import (
    FusedEmotionalState,
    ModalityWeights,
    MultimodalFusion,
    MultimodalResponseGenerator,
    TextToSpeechGenerator,
)

try:
    from .speech_recognition import (
        AudioPreprocessor as SpeechAudioPreprocessor,
    )
    from .speech_recognition import (
        SpeechRecognizer,
        TranscriptionResult,
        TranscriptionSegment,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    class SpeechAudioPreprocessor:
        """Fallback audio preprocessor when speech dependencies are unavailable."""

        def __init__(self, sample_rate: int = 16000):
            self.sample_rate = sample_rate

        def preprocess(self, audio_input):
            return audio_input

    class SpeechRecognizer:
        """Fallback speech recognizer when runtime speech dependencies are unavailable."""

        def __init__(self, *args, **kwargs):
            pass

    class TranscriptionResult:
        """Fallback transcription result placeholder."""

        text: str = ""

    class TranscriptionSegment:
        """Fallback transcription segment placeholder."""

        text: str = ""

__all__ = [
    # Speech recognition
    "SpeechRecognizer",
    "TranscriptionResult",
    "TranscriptionSegment",
    "SpeechAudioPreprocessor",
    # Audio emotion recognition
    "AudioEmotionRecognizer",
    "AudioPreprocessor",
    "EmotionTrajectory",
    "EmotionalState",
    "AudioEmotionResult",
    # Multimodal fusion
    "MultimodalFusion",
    "ModalityWeights",
    "FusedEmotionalState",
    "TextToSpeechGenerator",
    "MultimodalResponseGenerator",
]

__version__ = "0.1.0"
__author__ = "Pixelated Empathy"
