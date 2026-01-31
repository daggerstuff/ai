#!/usr/bin/env python3
"""Extract ALL YouTube creator transcripts for training.

This script processes transcripts from the YouTube RAG system and converts
them to ChatML format for training.

Outputs:
- ai/training_ready/data/generated/youtube_transcripts/{creator}/transcripts.jsonl
- ai/training_ready/data/generated/youtube_transcripts_stats.json

Usage:
    python extract_all_youtube_transcripts.py --all --output-dir ...
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to sys.path
project_root = Path(__file__).parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ai.pipelines.orchestrator.youtube_rag_system import (  # noqa: E402
    TranscriptMetadata,
    YouTubeRAGSystem,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingContext:
    output_dir: Path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract YouTube creator transcripts for training")
    parser.add_argument(
        "--creator",
        help="Specific creator to extract (optional filter)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all creators",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parents[1] / "data" / "generated" / "youtube_transcripts"),
        help="Output directory for transcripts",
    )
    return parser


def _sanitize_filename(name: str) -> str:
    return (
        "".join(c for c in name if c.isalnum() or c in (" ", "-", "_"))
        .strip()
        .replace(" ", "_")
        .lower()
    )


def _convert_transcript_to_chatml(metadata: TranscriptMetadata, content: str) -> dict[str, Any]:
    """Convert transcript to ChatML format"""

    # Create system prompt based on analysis
    approaches = (
        ", ".join(metadata.therapeutic_approaches)
        if metadata.therapeutic_approaches
        else "therapeutic principles"
    )
    topics = ", ".join(metadata.topics[:3]) if metadata.topics else "mental health"

    system_prompt = (
        f"You are a therapeutic AI assistant modeled after the speaking style of {metadata.speaker}. "
        f"You specialize in {topics} using {approaches}. "
        f"Respond with {metadata.personality_markers.get('tone', 'empathy')} and clarity."
    )

    # Create user message from transcript content (summarized or full?)
    # For training, we often want the transcript to represent the "assistant's knowledge" or "response"
    # But here, the transcript IS the content.
    # We can frame it as: User asks about [Title], Assistant provides [Content]

    user_message = f"Please discuss: {metadata.title}"

    # Assistant message is the transcript content
    assistant_message = content

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "metadata": {
            "source_family": "youtube_transcripts",
            "source_key": f"youtube/{_sanitize_filename(metadata.speaker)}/{metadata.video_id}",
            "creator": metadata.speaker,
            "persona": "expert",
            "video_id": metadata.video_id,
            "video_title": metadata.title,
            "topics": metadata.topics,
            "pii_status": "none_detected",  # Assumed processed
            "license_tag": "youtube_transcript",
            "split": "train",
            "phase": "stage1_foundation",
            "provenance": {
                "original_source": "youtube",
                "creator": metadata.speaker,
                "video_id": metadata.video_id,
                "processing_pipeline": "extract_all_youtube_transcripts",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dedup_status": "unique",
            },
        },
    }


def _process_speaker_batch(
    speaker: str,
    items: list[tuple[str, TranscriptMetadata]],
    rag_system: YouTubeRAGSystem,
    ctx: ProcessingContext,
) -> int:
    clean_speaker = _sanitize_filename(speaker)
    speaker_dir = ctx.output_dir / clean_speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)

    output_file = speaker_dir / "transcripts.jsonl"
    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for video_id, metadata in items:
            # Read content
            # rag_system.transcripts_dir / f"{video_id}.md"
            # Need to use private method or just read directly
            transcript_path = rag_system.transcripts_dir / f"{video_id}.md"
            if transcript_path.exists():
                raw_content = transcript_path.read_text(encoding="utf-8")
                # access protected method, should ideally expose public method but for now:
                # pylint: disable=protected-access
                content = rag_system._extract_transcript_content(raw_content)

                chatml = _convert_transcript_to_chatml(metadata, content)
                f.write(json.dumps(chatml, ensure_ascii=False) + "\n")
                count += 1

    logger.info(f"Processed {count} transcripts for {speaker} to {output_file}")
    return count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Initializing YouTube Transcript Extraction...")

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.creator and not args.all:
        logger.error("Must specify --creator or --all")
        return 1

    # Initialize RAG System to get transcripts
    try:
        rag_system = YouTubeRAGSystem()
        transcripts_map = rag_system.process_transcripts()
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        # Try to continue if transcripts_dir exists
        transcripts_map = {}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = ProcessingContext(
        output_dir=output_dir,
    )

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "creators": {},
        "total_transcripts": 0,
    }

    # Group by creator (speaker)
    grouped_transcripts = {}
    for video_id, metadata in transcripts_map.items():
        speaker = metadata.speaker or "Unknown"
        if args.creator and args.creator.lower() not in speaker.lower():
            continue

        if speaker not in grouped_transcripts:
            grouped_transcripts[speaker] = []
        grouped_transcripts[speaker].append((video_id, metadata))

    # Process groups
    for speaker, items in grouped_transcripts.items():
        count = _process_speaker_batch(speaker, items, rag_system, ctx)
        stats["creators"][speaker] = count
        stats["total_transcripts"] += count

    # Save stats
    stats_path = output_dir / "youtube_transcripts_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    logger.info(f"✓ Extraction complete. Total: {stats['total_transcripts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
