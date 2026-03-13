"""Tests for Video Emotion Recognition module."""

import pytest

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import pytest_asyncio

    HAS_ASYNCIO = True
except ImportError:
    HAS_ASYNCIO = False


class TestVideoEmotionRecognizer:
    """Tests for VideoEmotionRecognizer class."""

    def test_initialization_default_params(self):
        """Test default initialization."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionRecognizer

        recognizer = VideoEmotionRecognizer()
        assert recognizer.model_type == "edge"
        assert recognizer.edge_processing is True
        assert recognizer.frame_sample_rate == 5

    def test_initialization_custom_params(self):
        """Test custom parameter initialization."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionRecognizer

        recognizer = VideoEmotionRecognizer(
            model_type="accurate",
            device="cuda",
            edge_processing=False,
            frame_sample_rate=10,
        )
        assert recognizer.model_type == "accurate"
        assert recognizer.edge_processing is False
        assert recognizer.frame_sample_rate == 10

    @pytest.mark.skipif(not HAS_ASYNCIO, reason="pytest-asyncio not installed")
    @pytest.mark.asyncio
    async def test_detect_emotions_file_not_found(self):
        """Test handling of non-existent video file."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionRecognizer

        recognizer = VideoEmotionRecognizer()
        result = await recognizer.detect_emotions(
            "test_session", "/nonexistent/video.mp4"
        )

        assert result.error is not None
        assert "not found" in result.error.lower()
        assert result.session_id == "test_session"

    @pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")
    def test_rule_based_emotion(self):
        """Test rule-based emotion classification."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionRecognizer
        import numpy as np

        recognizer = VideoEmotionRecognizer()
        face_region = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        emotion, probabilities, confidence = recognizer._rule_based_emotion(face_region)

        assert isinstance(emotion, str)
        assert emotion in [
            "neutral",
            "happy",
            "sad",
            "angry",
            "fearful",
            "surprised",
            "disgusted",
            "contempt",
        ]
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 8
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_aggregate_frame_emotions_empty(self):
        """Test aggregation with no frames."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionRecognizer

        recognizer = VideoEmotionRecognizer()
        result = recognizer._aggregate_frame_emotions([])

        assert result.valence == 0.0
        assert result.arousal == 0.0
        assert result.dominance == 0.0
        assert result.confidence == 0.0
        assert result.primary_emotion == "neutral"


class TestVideoEmotionTrajectory:
    """Tests for VideoEmotionTrajectory class."""

    def test_initialization(self):
        """Test trajectory initialization."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionTrajectory

        trajectory = VideoEmotionTrajectory(window_size=30)
        assert trajectory.window_size == 30
        assert len(trajectory.frame_emotions) == 0

    def test_get_trend_insufficient_data(self):
        """Test trend calculation with insufficient data."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionTrajectory

        trajectory = VideoEmotionTrajectory()
        trend = trajectory.get_trend()

        assert trend["valence_trend"] == 0.0
        assert trend["arousal_trend"] == 0.0
        assert trend["dominance_trend"] == 0.0

    def test_get_stats_empty(self):
        """Test stats with no data."""
        from ai.core.multimodal.video_emotion_recognition import VideoEmotionTrajectory

        trajectory = VideoEmotionTrajectory()
        stats = trajectory.get_stats()

        assert stats == {}


class TestFusedEmotionalStateWithVideo:
    """Tests for FusedEmotionalState with video support."""

    def test_fused_state_with_video(self):
        """Test FusedEmotionalState includes video contribution."""
        from ai.core.multimodal.multimodal_fusion import FusedEmotionalState

        state = FusedEmotionalState(
            eq_scores=[0.8, 0.7, 0.6, 0.9, 0.75],
            overall_eq=0.75,
            valence=0.6,
            arousal=0.5,
            dominance=0.7,
            text_contribution=0.5,
            audio_contribution=0.3,
            video_contribution=0.2,
            conflict_score=0.1,
            confidence=0.85,
            visual_micro_leakage=False,
        )

        assert state.video_contribution == 0.2
        assert state.visual_micro_leakage is False

    def test_fused_state_micro_leakage_detected(self):
        """Test FusedEmotionalState with micro-leakage detected."""
        from ai.core.multimodal.multimodal_fusion import FusedEmotionalState

        state = FusedEmotionalState(
            eq_scores=[0.5, 0.5, 0.5, 0.5, 0.5],
            overall_eq=0.5,
            valence=-0.5,
            arousal=0.5,
            dominance=0.0,
            text_contribution=0.5,
            audio_contribution=0.3,
            video_contribution=0.2,
            conflict_score=0.6,
            confidence=0.7,
            visual_micro_leakage=True,
        )

        assert state.visual_micro_leakage is True


class TestMultimodalFusionWithVideo:
    """Tests for MultimodalFusion with video modality."""

    def test_initialization_with_video_weight(self):
        """Test initialization with video weight."""
        from ai.core.multimodal.multimodal_fusion import MultimodalFusion

        fusion = MultimodalFusion(text_weight=0.5, audio_weight=0.3, video_weight=0.2)

        assert abs(fusion.text_weight - 0.5) < 0.01
        assert abs(fusion.audio_weight - 0.3) < 0.01
        assert abs(fusion.video_weight - 0.2) < 0.01

    def test_fuse_emotions_all_modalities(self):
        """Test fusion with all three modalities."""
        from ai.core.multimodal.multimodal_fusion import MultimodalFusion

        fusion = MultimodalFusion(text_weight=0.5, audio_weight=0.3, video_weight=0.2)

        result = fusion.fuse_emotions(
            text_emotion={"eq_scores": [0.8, 0.7, 0.6, 0.9, 0.75], "confidence": 0.9},
            audio_emotion={
                "valence": 0.6,
                "arousal": 0.7,
                "dominance": 0.5,
                "confidence": 0.8,
            },
            video_emotion={
                "valence": 0.5,
                "arousal": 0.4,
                "dominance": 0.6,
                "confidence": 0.7,
            },
        )

        assert result.video_contribution == 0.2
        assert result.confidence > 0
        assert result.video_emotion is not None

    def test_fuse_emotions_video_only(self):
        """Test fusion with only video modality."""
        from ai.core.multimodal.multimodal_fusion import MultimodalFusion

        fusion = MultimodalFusion(text_weight=0.5, audio_weight=0.3, video_weight=0.2)

        result = fusion.fuse_emotions(
            video_emotion={
                "valence": 0.5,
                "arousal": 0.5,
                "dominance": 0.5,
                "confidence": 0.8,
            }
        )

        assert result.video_emotion is not None
        assert result.text_emotion is not None
        assert result.audio_emotion is not None

    def test_detect_visual_micro_leakage_aligned(self):
        """Test micro-leakage detection when emotions are aligned."""
        from ai.core.multimodal.multimodal_fusion import MultimodalFusion

        fusion = MultimodalFusion()

        leakage = fusion._detect_visual_micro_leakage(
            text_emotion={"eq_scores": [0.8, 0.7, 0.6, 0.9, 0.85]},
            video_emotion={
                "valence": 0.6,
                "arousal": 0.5,
                "dominance": 0.5,
                "confidence": 0.8,
            },
        )

        assert leakage is False or leakage is None

    def test_detect_visual_micro_leakage_conflict(self):
        """Test micro-leakage detection when emotions conflict."""
        from ai.core.multimodal.multimodal_fusion import MultimodalFusion

        fusion = MultimodalFusion()

        leakage = fusion._detect_visual_micro_leakage(
            text_emotion={"eq_scores": [0.2, 0.2, 0.2, 0.2, 0.2]},
            video_emotion={
                "valence": 0.8,
                "arousal": 0.5,
                "dominance": 0.5,
                "confidence": 0.9,
            },
        )

        assert leakage is True


class TestModalityWeightsWithVideo:
    """Tests for ModalityWeights with video support."""

    def test_modality_weights_with_video(self):
        """Test ModalityWeights includes video_weight."""
        from ai.core.multimodal.multimodal_fusion import ModalityWeights

        weights = ModalityWeights(text_weight=0.5, audio_weight=0.3, video_weight=0.2)

        assert weights.text_weight == 0.5
        assert weights.audio_weight == 0.3
        assert weights.video_weight == 0.2

    def test_modality_weights_defaults(self):
        """Test default weights include video."""
        from ai.core.multimodal.multimodal_fusion import ModalityWeights

        weights = ModalityWeights()

        assert weights.text_weight == 0.5
        assert weights.audio_weight == 0.3
        assert weights.video_weight == 0.2
