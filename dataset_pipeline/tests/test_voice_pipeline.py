"""
Comprehensive tests for the voice training data pipeline.

Tests all components of the voice processing pipeline including YouTube processing,
audio preprocessing, transcription, personality extraction, and conversation conversion.
"""

import unittest
from unittest.mock import Mock, patch

from .audio_processor import AudioProcessor, AudioQualityMetrics
from .personality_extractor import PersonalityDimension, PersonalityExtractor, PersonalityProfile
from .voice_conversation_converter import VoiceConversationConverter
from .voice_pipeline_integration import VoicePipelineConfig, VoiceTrainingPipeline
from .voice_transcriber import TranscriptionResult, TranscriptionSegment, VoiceTranscriber


class TestAudioProcessor(unittest.TestCase):
    """Test audio processing functionality."""

    def setUp(self):
        self.processor = AudioProcessor()

    def test_audio_quality_assessment(self):
        """Test audio quality assessment with mock data."""
        # Create a mock audio file path
        mock_file = "test_audio.wav"

        with patch.object(self.processor, "_get_audio_info_ffprobe") as mock_ffprobe:
            mock_ffprobe.return_value = {
                "duration": 10.0,
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le"
            }

            metrics = self.processor.assess_audio_quality(mock_file)

            assert metrics.file_path == mock_file
            assert metrics.duration == 10.0
            assert metrics.sample_rate == 16000
            assert metrics.channels == 1
            assert isinstance(metrics.quality_score, float)
            assert metrics.quality_score >= 0.0
            assert metrics.quality_score <= 1.0

    def test_quality_score_calculation(self):
        """Test quality score calculation logic."""
        # Test high quality metrics
        high_quality_metrics = AudioQualityMetrics(
            file_path="test.wav",
            duration=15.0,
            sample_rate=16000,
            channels=1,
            snr_db=25.0,
            silence_ratio=0.1,
            clipping_ratio=0.0
        )

        score = self.processor._calculate_quality_score(high_quality_metrics)
        assert score > 0.7

        # Test low quality metrics
        low_quality_metrics = AudioQualityMetrics(
            file_path="test.wav",
            duration=0.5,  # Too short
            sample_rate=8000,  # Low sample rate
            channels=1,
            snr_db=5.0,  # Low SNR
            silence_ratio=0.9,  # Mostly silence
            clipping_ratio=0.05  # Clipping present
        )

        score = self.processor._calculate_quality_score(low_quality_metrics)
        assert score < 0.3


class TestVoiceTranscriber(unittest.TestCase):
    """Test voice transcription functionality."""

    def setUp(self):
        # Mock the model initialization to avoid loading actual models
        with patch("ai.dataset_pipeline.voice_transcriber.FASTER_WHISPER_AVAILABLE", False):
            with patch("ai.dataset_pipeline.voice_transcriber.WHISPER_AVAILABLE", False):
                self.transcriber = VoiceTranscriber()
                self.transcriber.model = Mock()
                self.transcriber.model_type = "mock"

    def test_transcription_result_creation(self):
        """Test creation of transcription results."""
        segments = [
            TranscriptionSegment(
                start_time=0.0,
                end_time=5.0,
                text="Hello, how are you?",
                confidence=0.9
            ),
            TranscriptionSegment(
                start_time=5.0,
                end_time=10.0,
                text="I'm doing well, thank you.",
                confidence=0.85
            )
        ]

        result = TranscriptionResult(
            file_path="test.wav",
            success=True,
            segments=segments,
            full_text="Hello, how are you? I'm doing well, thank you.",
            confidence_score=0.875,
            model_used="mock-whisper"
        )

        assert result.success
        assert len(result.segments) == 2
        assert result.confidence_score == 0.875
        assert "Hello" in result.full_text

    def test_confidence_calculation(self):
        """Test confidence score calculation from log probabilities."""
        # Test high confidence (low negative log prob)
        high_conf = self.transcriber._logprob_to_confidence(-0.5)
        assert high_conf > 0.9

        # Test low confidence (high negative log prob)
        low_conf = self.transcriber._logprob_to_confidence(-3.5)
        assert low_conf < 0.4

        # Test medium confidence
        med_conf = self.transcriber._logprob_to_confidence(-1.5)
        assert med_conf > 0.6
        assert med_conf < 0.8


class TestPersonalityExtractor(unittest.TestCase):
    """Test personality extraction functionality."""

    def setUp(self):
        self.extractor = PersonalityExtractor()

    def test_personality_extraction(self):
        """Test personality profile extraction from text."""
        test_text = """
        I'm really excited about this creative project! I love exploring new ideas
        and thinking outside the box. I'm very organized and always plan everything
        carefully. I enjoy meeting new people and talking about innovative solutions.
        I try to be kind and helpful to everyone I meet.
        """

        profile = self.extractor.extract_personality_profile(test_text)

        assert isinstance(profile, PersonalityProfile)
        assert len(profile.personality_scores) == 5  # Big Five dimensions
        assert profile.confidence_score > 0.0
        assert profile.confidence_score <= 1.0

        # Check that we found some personality indicators
        openness_score = next(
            (score for score in profile.personality_scores
             if score.dimension == PersonalityDimension.OPENNESS),
            None
        )
        assert openness_score is not None
        assert openness_score.score > 0.5  # Should detect high openness

    def test_communication_style_detection(self):
        """Test communication style pattern detection."""
        formal_text = "Therefore, I believe we should carefully analyze the situation."
        informal_text = "Yeah, that's totally awesome! I mean, like, it's really cool."

        formal_profile = self.extractor.extract_personality_profile(formal_text)
        informal_profile = self.extractor.extract_personality_profile(informal_text)

        # Check that different styles are detected
        assert len(formal_profile.communication_patterns) != len(informal_profile.communication_patterns)

    def test_emotional_analysis(self):
        """Test emotional pattern analysis."""
        emotional_text = """
        I feel so happy and excited about this opportunity! It brings me great joy
        to help others and show compassion. I'm grateful for all the support.
        """

        profile = self.extractor.extract_personality_profile(emotional_text)
        emotional_profile = profile.emotional_profile

        assert "positive" in emotional_profile.dominant_emotions
        assert emotional_profile.emotional_vocabulary_richness > 0.0
        assert len(emotional_profile.empathy_indicators) > 0


class TestVoiceConversationConverter(unittest.TestCase):
    """Test voice-to-conversation conversion functionality."""

    def setUp(self):
        self.converter = VoiceConversationConverter()

    def test_conversation_conversion(self):
        """Test conversion of transcription to conversation format."""
        # Create mock transcription result
        segments = [
            TranscriptionSegment(
                start_time=0.0,
                end_time=3.0,
                text="Hello, how can I help you today?",
                confidence=0.9
            ),
            TranscriptionSegment(
                start_time=5.0,
                end_time=8.0,
                text="I've been feeling anxious lately.",
                confidence=0.85
            ),
            TranscriptionSegment(
                start_time=10.0,
                end_time=15.0,
                text="Can you tell me more about what's been causing this anxiety?",
                confidence=0.88
            )
        ]

        transcription = TranscriptionResult(
            file_path="therapy_session.wav",
            success=True,
            segments=segments,
            full_text=" ".join(seg.text for seg in segments),
            confidence_score=0.88,
            model_used="test-whisper"
        )

        result = self.converter.convert_transcription_to_conversation(transcription)

        assert result.success
        assert result.conversation is not None
        assert len(result.conversation.messages) > 0
        assert result.quality_score > 0.0

    def test_speaker_change_detection(self):
        """Test detection of speaker changes in conversation."""
        prev_text = "How are you feeling today?"
        current_text = "I'm feeling much better, thank you."

        speaker_change = self.converter._detect_speaker_change_by_content(prev_text, current_text)
        assert speaker_change  # Question followed by answer

        # Test non-speaker change
        prev_text = "I think we should consider"
        current_text = "all the available options carefully."

        speaker_change = self.converter._detect_speaker_change_by_content(prev_text, current_text)
        assert not speaker_change  # Continuation of same thought

    def test_message_text_cleaning(self):
        """Test message text cleaning functionality."""
        dirty_text = "  Um,  like,  I  think  that  uh  we  should  go  .  "
        clean_text = self.converter._clean_message_text(dirty_text)

        assert "um" not in clean_text.lower()
        assert "uh" not in clean_text.lower()
        assert "  " not in clean_text  # No double spaces
        assert clean_text[0].isupper()  # Capitalized


class TestVoicePipelineIntegration(unittest.TestCase):
    """Test complete voice pipeline integration."""

    def setUp(self):
        self.config = VoicePipelineConfig(
            youtube_output_dir="test_youtube",
            audio_output_dir="test_audio",
            transcription_output_dir="test_transcriptions",
            conversation_output_dir="test_conversations"
        )

    @patch("ai.dataset_pipeline.voice_pipeline_integration.YouTubePlaylistProcessor")
    @patch("ai.dataset_pipeline.voice_pipeline_integration.AudioProcessor")
    @patch("ai.dataset_pipeline.voice_pipeline_integration.VoiceTranscriber")
    @patch("ai.dataset_pipeline.voice_pipeline_integration.VoiceConversationConverter")
    def test_pipeline_initialization(self, mock_converter, mock_transcriber, mock_audio, mock_youtube):
        """Test pipeline component initialization."""
        pipeline = VoiceTrainingPipeline(self.config)

        # Verify all components are initialized
        assert pipeline.youtube_processor is not None
        assert pipeline.audio_processor is not None
        assert pipeline.voice_transcriber is not None
        assert pipeline.conversation_converter is not None

    def test_quality_distribution_analysis(self):
        """Test quality distribution analysis."""
        from .voice_conversation_converter import ConversionResult

        # Create mock conversion results with different quality scores
        results = [
            ConversionResult(success=True, quality_score=0.9),  # excellent
            ConversionResult(success=True, quality_score=0.7),  # good
            ConversionResult(success=True, quality_score=0.5),  # acceptable
            ConversionResult(success=True, quality_score=0.3),  # poor
            ConversionResult(success=False, quality_score=0.0)  # failed
        ]

        pipeline = VoiceTrainingPipeline(self.config)
        distribution = pipeline._analyze_quality_distribution(results)

        assert distribution["excellent"] == 1
        assert distribution["good"] == 1
        assert distribution["acceptable"] == 1
        assert distribution["poor"] == 1


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end pipeline functionality with mocked components."""

    def test_simple_pipeline_flow(self):
        """Test a simplified end-to-end pipeline flow."""
        # This test verifies that all components can work together
        # without actually processing real audio/video files

        # Create mock data that flows through the pipeline
        mock_audio_file = "test_audio.wav"
        mock_transcription_text = "Hello, how are you feeling today? I'm doing well, thank you for asking."

        # Test audio quality assessment
        processor = AudioProcessor()
        with patch.object(processor, "_get_audio_info_ffprobe") as mock_ffprobe:
            mock_ffprobe.return_value = {
                "duration": 10.0,
                "sample_rate": 16000,
                "channels": 1
            }
            quality_metrics = processor.assess_audio_quality(mock_audio_file)
            assert quality_metrics.quality_score > 0.0

        # Test personality extraction
        extractor = PersonalityExtractor()
        personality_profile = extractor.extract_personality_profile(mock_transcription_text)
        assert personality_profile.confidence_score > 0.0

        # Test conversation conversion
        converter = VoiceConversationConverter()

        # Create mock transcription segments
        segments = [
            TranscriptionSegment(0.0, 3.0, "Hello, how are you feeling today?", 0.9),
            TranscriptionSegment(4.0, 7.0, "I'm doing well, thank you for asking.", 0.85)
        ]

        transcription = TranscriptionResult(
            file_path=mock_audio_file,
            success=True,
            segments=segments,
            full_text=mock_transcription_text,
            confidence_score=0.875
        )

        conversion_result = converter.convert_transcription_to_conversation(
            transcription, personality_profile
        )

        assert conversion_result.success
        assert conversion_result.conversation is not None
        assert len(conversion_result.conversation.messages) > 0


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
