#!/usr/bin/env python3
"""
Voice Training Data Pipeline Usage Examples.

This file demonstrates how to use the voice processing pipeline
for converting YouTube content into training conversations.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.tools.utilities.pipelines.voice_pipeline_integration import (
    VoicePipelineConfig,
    VoiceTrainingPipeline,
    process_youtube_voice_data,
)


async def example_simple_processing():
    """Simple example using the convenience function."""

    # Example YouTube URLs (replace with actual URLs)
    playlist_urls = [
        "https://www.youtube.com/playlist?list=PLexample1",
        "https://www.youtube.com/watch?v=example_video_id",
    ]

    try:
        result = await process_youtube_voice_data(
            playlist_urls=playlist_urls,
            output_base_dir="voice_training_output",
            whisper_model="base",
            quality_threshold=0.6,
        )

        if result.success:
            pass
        else:
            for _error in result.errors[:3]:
                pass

    except Exception:
        pass


async def example_advanced_configuration():
    """Advanced example with custom configuration."""

    # Custom configuration
    config = VoicePipelineConfig(
        # Output directories
        youtube_output_dir="advanced_output/youtube",
        audio_output_dir="advanced_output/audio",
        transcription_output_dir="advanced_output/transcriptions",
        conversation_output_dir="advanced_output/conversations",
        # Processing settings
        whisper_model="large",  # Higher quality model
        transcription_language="en",  # Force English
        audio_format="wav",
        target_sample_rate=16000,
        # Quality thresholds
        overall_quality_threshold=0.7,  # Higher quality requirement
        min_transcription_confidence=0.7,
        audio_quality_threshold=0.6,
        # Performance settings
        max_concurrent_downloads=2,  # Conservative for large model
        # Conversation settings
        min_conversation_length=5,  # Longer conversations
        max_speaker_gap=20.0,  # Shorter gap for speaker changes
        # Options
        use_faster_whisper=True,  # Use Faster-Whisper for speed
        save_intermediate_results=True,  # Keep all intermediate files
    )

    # Initialize pipeline
    pipeline = VoiceTrainingPipeline(config)

    # Example URLs
    urls = [
        "https://www.youtube.com/playlist?list=PLexample_therapy_sessions",
        "https://www.youtube.com/playlist?list=PLexample_interviews",
    ]

    try:
        result = await pipeline.process_youtube_playlists(urls)

        # Generate detailed report
        report = pipeline.generate_pipeline_report(result)

        # Save report to file
        report_path = Path(config.conversation_output_dir) / "processing_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

    except Exception:
        pass


async def example_batch_processing():
    """Example of processing multiple batches with different settings."""

    # Different batches with different quality requirements
    batches = [
        {
            "name": "High Quality Therapy Sessions",
            "urls": ["https://www.youtube.com/playlist?list=PLtherapy_high_quality"],
            "config": VoicePipelineConfig(
                whisper_model="large",
                overall_quality_threshold=0.8,
                min_conversation_length=10,
                conversation_output_dir="output/high_quality",
            ),
        },
        {
            "name": "General Conversations",
            "urls": ["https://www.youtube.com/playlist?list=PLgeneral_conversations"],
            "config": VoicePipelineConfig(
                whisper_model="base",
                overall_quality_threshold=0.5,
                min_conversation_length=3,
                conversation_output_dir="output/general",
            ),
        },
        {
            "name": "Quick Processing",
            "urls": ["https://www.youtube.com/playlist?list=PLquick_content"],
            "config": VoicePipelineConfig(
                whisper_model="tiny",
                overall_quality_threshold=0.3,
                min_conversation_length=2,
                conversation_output_dir="output/quick",
                save_intermediate_results=False,
            ),
        },
    ]

    total_conversations = 0

    for batch in batches:
        pipeline = VoiceTrainingPipeline(batch["config"])
        result = await pipeline.process_youtube_playlists(batch["urls"])

        if result.success:
            total_conversations += result.total_conversations
        else:
            pass


def example_cli_usage():
    """Show CLI usage examples."""

    cli_examples = [
        "# Simple single playlist processing",
        'python scripts/run_voice_pipeline.py --url "https://youtube.com/playlist?list=..." --output-dir voice_data',
        "",
        "# Process multiple playlists from file",
        "python scripts/run_voice_pipeline.py --url-file playlists.txt --whisper-model large",
        "",
        "# High quality processing with custom settings",
        "python scripts/run_voice_pipeline.py --url-file therapy_playlists.txt \\",
        "  --whisper-model large --quality-threshold 0.8 --min-conversation-length 5",
        "",
        "# Quick processing for testing",
        'python scripts/run_voice_pipeline.py --url "..." --whisper-model tiny \\',
        "  --quality-threshold 0.3 --no-intermediate-files",
        "",
        "# Dry run to see what would be processed",
        "python scripts/run_voice_pipeline.py --url-file playlists.txt --dry-run",
    ]

    for _line in cli_examples:
        pass


async def main():
    """Run all examples."""

    # Note: These examples use placeholder URLs
    # Replace with actual YouTube URLs for real processing

    # Show CLI examples
    example_cli_usage()


if __name__ == "__main__":
    asyncio.run(main())
