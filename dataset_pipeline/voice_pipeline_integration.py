"""
Voice Training Data Pipeline Integration.

This module provides a comprehensive integration of all voice processing components:
YouTube processing, audio preprocessing, transcription, personality extraction,
and conversation format conversion for complete voice training data pipeline.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .audio_processor import AudioProcessor
from .logger import setup_logger
from .personality_extractor import PersonalityExtractor
from .voice_conversation_converter import ConversionResult, VoiceConversationConverter
from .voice_transcriber import VoiceTranscriber
from .youtube_processor import (
    AntiDetectionConfig,
    ProxyConfig,
    RateLimitConfig,
    YouTubePlaylistProcessor,
)


@dataclass
class VoicePipelineConfig:
    """Configuration for the voice processing pipeline."""
    # YouTube processing
    youtube_output_dir: str = "voice_data/youtube"
    audio_format: str = "wav"
    max_concurrent_downloads: int = 3

    # Audio processing
    audio_output_dir: str = "voice_data/processed"
    target_sample_rate: int = 16000
    min_segment_duration: float = 1.0
    max_segment_duration: float = 30.0
    audio_quality_threshold: float = 0.6

    # Transcription
    transcription_output_dir: str = "voice_data/transcriptions"
    whisper_model: str = "base"
    transcription_language: str | None = None
    min_transcription_confidence: float = 0.6
    use_faster_whisper: bool = True

    # Personality extraction
    personality_output_dir: str = "voice_data/personalities"
    personality_language: str = "en"

    # Conversation conversion
    conversation_output_dir: str = "voice_data/conversations"
    min_conversation_length: int = 3
    max_speaker_gap: float = 30.0

    # Quality filtering
    overall_quality_threshold: float = 0.5
    save_intermediate_results: bool = True

    # Enhanced YouTube processing configurations
    rate_limit_config: RateLimitConfig | None = None
    proxy_config: ProxyConfig | None = None
    anti_detection_config: AntiDetectionConfig | None = None


@dataclass
class VoicePipelineResult:
    """Result of complete voice pipeline processing."""
    success: bool
    total_playlists: int = 0
    total_audio_files: int = 0
    total_transcriptions: int = 0
    total_conversations: int = 0
    processing_time: float = 0.0
    quality_distribution: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    output_files: dict[str, list[str]] = field(default_factory=dict)


class VoiceTrainingPipeline:
    """
    Comprehensive voice training data processing pipeline.

    Integrates all voice processing components into a single, cohesive pipeline
    that can process YouTube playlists from start to finish, producing
    high-quality conversation datasets for voice training.
    """

    def __init__(self, config: VoicePipelineConfig):
        self.config = config
        self.logger = setup_logger("voice_training_pipeline")

        # Initialize all components
        self.youtube_processor = YouTubePlaylistProcessor(
            output_dir=config.youtube_output_dir,
            audio_format=config.audio_format,
            max_concurrent=config.max_concurrent_downloads,
            rate_limit_config=config.rate_limit_config,
            proxy_config=config.proxy_config,
            anti_detection_config=config.anti_detection_config
        )

        self.audio_processor = AudioProcessor(
            target_sample_rate=config.target_sample_rate,
            min_segment_duration=config.min_segment_duration,
            max_segment_duration=config.max_segment_duration,
            quality_threshold=config.audio_quality_threshold
        )

        self.voice_transcriber = VoiceTranscriber(
            model_name=config.whisper_model,
            language=config.transcription_language,
            min_confidence=config.min_transcription_confidence,
            use_faster_whisper=config.use_faster_whisper
        )

        self.personality_extractor = PersonalityExtractor(
            language=config.personality_language
        )

        self.conversation_converter = VoiceConversationConverter(
            min_conversation_length=config.min_conversation_length,
            max_gap_seconds=config.max_speaker_gap
        )

        # Create output directories
        self._create_output_directories()

    def _create_output_directories(self):
        """Create all necessary output directories."""
        directories = [
            self.config.youtube_output_dir,
            self.config.audio_output_dir,
            self.config.transcription_output_dir,
            self.config.personality_output_dir,
            self.config.conversation_output_dir
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def process_youtube_playlists(self, playlist_urls: list[str]) -> VoicePipelineResult:
        """Process YouTube playlists through the complete pipeline."""
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting voice pipeline processing for {len(playlist_urls)} playlists")

            # Step 1: Download YouTube playlists
            self.logger.info("Step 1: Downloading YouTube playlists...")
            youtube_result = await self.youtube_processor.process_playlists_batch(playlist_urls)

            if youtube_result.successful_playlists == 0:
                return VoicePipelineResult(
                    success=False,
                    errors=["No playlists downloaded successfully", *youtube_result.errors]
                )

            # Collect all audio files
            all_audio_files = []
            for result in youtube_result.results:
                if result.success:
                    all_audio_files.extend([str(f) for f in result.audio_files])

            self.logger.info(f"Downloaded {len(all_audio_files)} audio files")

            # Step 2: Process audio files
            self.logger.info("Step 2: Processing audio files...")
            audio_results = self.audio_processor.process_batch(
                all_audio_files,
                self.config.audio_output_dir
            )

            # Collect processed audio segments
            processed_segments = []
            for result in audio_results:
                if result.success:
                    processed_segments.extend([seg.file_path for seg in result.segments])

            self.logger.info(f"Processed {len(processed_segments)} audio segments")

            # Step 3: Transcribe audio segments
            self.logger.info("Step 3: Transcribing audio segments...")
            transcription_result = self.voice_transcriber.transcribe_batch(
                processed_segments,
                self.config.transcription_output_dir if self.config.save_intermediate_results else None
            )

            successful_transcriptions = [r for r in transcription_result.results if r.success]
            self.logger.info(f"Transcribed {len(successful_transcriptions)} segments")

            # Step 4: Extract personality profiles and convert to conversations
            self.logger.info("Step 4: Converting to conversation format...")
            conversion_results = self.conversation_converter.convert_batch(
                successful_transcriptions,
                self.config.conversation_output_dir
            )

            successful_conversations = [r for r in conversion_results if r.success]
            self.logger.info(f"Created {len(successful_conversations)} conversations")

            # Step 5: Quality filtering and final processing
            self.logger.info("Step 5: Quality filtering...")
            high_quality_conversations = self._filter_by_quality(successful_conversations)

            # Generate comprehensive results
            processing_time = (datetime.now() - start_time).total_seconds()

            result = VoicePipelineResult(
                success=True,
                total_playlists=youtube_result.total_playlists,
                total_audio_files=len(all_audio_files),
                total_transcriptions=len(successful_transcriptions),
                total_conversations=len(high_quality_conversations),
                processing_time=processing_time,
                quality_distribution=self._analyze_quality_distribution(successful_conversations),
                errors=youtube_result.errors + transcription_result.errors,
                output_files={
                    "conversations": [r.conversation for r in high_quality_conversations],
                    "transcriptions": [r.file_path for r in successful_transcriptions],
                    "audio_segments": processed_segments
                }
            )

            # Save pipeline summary
            self._save_pipeline_summary(result)

            self.logger.info(f"Pipeline complete: {len(high_quality_conversations)} high-quality conversations in {processing_time:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Pipeline processing failed: {e}"
            self.logger.error(error_msg)

            return VoicePipelineResult(
                success=False,
                processing_time=(datetime.now() - start_time).total_seconds(),
                errors=[error_msg]
            )

    def _filter_by_quality(self, conversion_results: list[ConversionResult]) -> list[ConversionResult]:
        """Filter conversations by overall quality threshold."""
        high_quality = []

        for result in conversion_results:
            if result.success and result.quality_score >= self.config.overall_quality_threshold:
                high_quality.append(result)

        self.logger.info(f"Quality filtering: {len(high_quality)}/{len(conversion_results)} conversations passed")
        return high_quality

    def _analyze_quality_distribution(self, results: list[ConversionResult]) -> dict[str, int]:
        """Analyze quality score distribution."""
        distribution = {
            "excellent": 0,    # 0.8+
            "good": 0,         # 0.6-0.8
            "acceptable": 0,   # 0.4-0.6
            "poor": 0          # <0.4
        }

        for result in results:
            if result.success:
                score = result.quality_score
                if score >= 0.8:
                    distribution["excellent"] += 1
                elif score >= 0.6:
                    distribution["good"] += 1
                elif score >= 0.4:
                    distribution["acceptable"] += 1
                else:
                    distribution["poor"] += 1

        return distribution

    def _save_pipeline_summary(self, result: VoicePipelineResult):
        """Save comprehensive pipeline processing summary."""
        try:
            summary_path = Path(self.config.conversation_output_dir) / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            summary_data = {
                "pipeline_config": {
                    "youtube_output_dir": self.config.youtube_output_dir,
                    "whisper_model": self.config.whisper_model,
                    "quality_threshold": self.config.overall_quality_threshold,
                    "target_sample_rate": self.config.target_sample_rate
                },
                "results": {
                    "success": result.success,
                    "total_playlists": result.total_playlists,
                    "total_audio_files": result.total_audio_files,
                    "total_transcriptions": result.total_transcriptions,
                    "total_conversations": result.total_conversations,
                    "processing_time": result.processing_time,
                    "quality_distribution": result.quality_distribution
                },
                "errors": result.errors,
                "timestamp": datetime.now().isoformat()
            }

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Pipeline summary saved: {summary_path}")

        except Exception as e:
            self.logger.error(f"Failed to save pipeline summary: {e}")

    def generate_pipeline_report(self, result: VoicePipelineResult) -> str:
        """Generate comprehensive pipeline processing report."""
        report = []
        report.append("=" * 80)
        report.append("VOICE TRAINING DATA PIPELINE REPORT")
        report.append("=" * 80)
        report.append(f"Processing Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        report.append(f"Total Processing Time: {result.processing_time:.2f} seconds")
        report.append("")

        report.append("PROCESSING STATISTICS:")
        report.append("-" * 40)
        report.append(f"YouTube Playlists Processed: {result.total_playlists}")
        report.append(f"Audio Files Downloaded: {result.total_audio_files}")
        report.append(f"Successful Transcriptions: {result.total_transcriptions}")
        report.append(f"High-Quality Conversations: {result.total_conversations}")

        if result.total_playlists > 0:
            conversion_rate = (result.total_conversations / result.total_playlists) * 100
            report.append(f"Playlist-to-Conversation Rate: {conversion_rate:.1f}%")

        report.append("")

        if result.quality_distribution:
            report.append("QUALITY DISTRIBUTION:")
            report.append("-" * 40)
            for quality, count in result.quality_distribution.items():
                report.append(f"{quality.capitalize()}: {count}")

        if result.errors:
            report.append("")
            report.append("ERRORS ENCOUNTERED:")
            report.append("-" * 40)
            for error in result.errors[:10]:  # Show first 10 errors
                report.append(f"• {error[:150]}...")

        report.append("=" * 80)
        return "\n".join(report)


# Convenience function for simple pipeline execution
async def process_youtube_voice_data(
    playlist_urls: list[str],
    output_base_dir: str = "voice_training_data",
    whisper_model: str = "base",
    quality_threshold: float = 0.5
) -> VoicePipelineResult:
    """
    Process YouTube playlists through the complete voice training pipeline.

    This is a convenience function that sets up the pipeline with sensible
    defaults and processes the provided playlist URLs.
    """
    config = VoicePipelineConfig(
        youtube_output_dir=f"{output_base_dir}/youtube",
        audio_output_dir=f"{output_base_dir}/processed_audio",
        transcription_output_dir=f"{output_base_dir}/transcriptions",
        personality_output_dir=f"{output_base_dir}/personalities",
        conversation_output_dir=f"{output_base_dir}/conversations",
        whisper_model=whisper_model,
        overall_quality_threshold=quality_threshold
    )

    pipeline = VoiceTrainingPipeline(config)
    return await pipeline.process_youtube_playlists(playlist_urls)
