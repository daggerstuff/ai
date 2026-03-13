"""
Multimodal Processing for Pixel Therapeutic Conversations.

Enables real-time processing of audio, video, and text modalities with emotion fusion.

Modules:
  - speech_recognition: Audio-to-text conversion (Whisper, Wav2Vec2)
  - audio_emotion_recognition: Emotion detection from speech (valence/arousal)
  - video_emotion_recognition: Facial expression emotion detection (edge processing)
  - multimodal_fusion: Combine text, audio, and video signals for enhanced understanding
  - text_to_speech: Generate speech with emotional prosody (future)

Usage:
    >>> from ai.multimodal import SpeechRecognizer, AudioEmotionRecognizer, VideoEmotionRecognizer
    >>>
    >>> # Speech recognition
    >>> recognizer = SpeechRecognizer(model_name="base")
    >>> transcript = await recognizer.transcribe_audio("session_001", "audio.wav")
    >>>
    >>> # Audio emotion
    >>> emotion_recognizer = AudioEmotionRecognizer()
    >>> emotions = await emotion_recognizer.detect_emotions("session_001", "audio.wav")
    >>>
    >>> # Video emotion (edge processing for privacy)
    >>> video_recognizer = VideoEmotionRecognizer(edge_processing=True)
    >>> video_emotions = await video_recognizer.detect_emotions("session_001", "video.mp4")
    >>>
    >>> # Multimodal fusion
    >>> from ai.multimodal import MultimodalFusion
    >>> fusion = MultimodalFusion(text_weight=0.5, audio_weight=0.3, video_weight=0.2)
    >>> fused = fusion.fuse_emotions(
    ...     text_emotion=pixel_output,
    ...     audio_emotion=emotions.overall_emotion.__dict__,
    ...     video_emotion=video_emotions.overall_emotion.__dict__
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
    from .video_emotion_recognition import (
        FaceDetection,
        FacialLandmark,
        FrameEmotion,
        VideoEmotionRecognizer,
        VideoEmotionResult,
        VideoEmotionTrajectory,
    )
except Exception:  # pragma: no cover - optional dependency fallback

    class VideoEmotionRecognizer:
        def __init__(self, *args, **kwargs):
            pass

    class VideoEmotionResult:
        session_id: str = ""
        error: str = "Video dependencies unavailable"

    class FaceDetection:
        pass

    class FacialLandmark:
        pass

    class FrameEmotion:
        pass

    class VideoEmotionTrajectory:
        pass


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
        def __init__(self, sample_rate: int = 16000):
            self.sample_rate = sample_rate

        def preprocess(self, audio_input):
            return audio_input

    class SpeechRecognizer:
        def __init__(self, *args, **kwargs):
            pass

    class TranscriptionResult:
        text: str = ""

    class TranscriptionSegment:
        text: str = ""


__all__ = [
    "SpeechRecognizer",
    "TranscriptionResult",
    "TranscriptionSegment",
    "SpeechAudioPreprocessor",
    "AudioEmotionRecognizer",
    "AudioPreprocessor",
    "EmotionTrajectory",
    "EmotionalState",
    "AudioEmotionResult",
    "MultimodalFusion",
    "ModalityWeights",
    "FusedEmotionalState",
    "TextToSpeechGenerator",
    "MultimodalResponseGenerator",
    "VideoEmotionRecognizer",
    "VideoEmotionResult",
    "FaceDetection",
    "FacialLandmark",
    "FrameEmotion",
    "VideoEmotionTrajectory",
]

__version__ = "0.2.0"
__author__ = "Pixelated Empathy"
