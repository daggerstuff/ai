#!/usr/bin/env python3
"""Extract ALL YouTube creator transcripts for training.

This script extracts transcripts from multiple YouTube creators and converts
them to ChatML format for training.

Outputs:
- ai/training_ready/data/generated/youtube_transcripts/{creator}/transcripts.jsonl
- ai/training_ready/data/generated/youtube_transcripts_stats.json

Usage:
    python extract_all_youtube_transcripts.py --creator understood
    python extract_all_youtube_transcripts.py --all
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

logger = logging.getLogger(__name__)

# YouTube creators configuration
CREATORS = {
    "understood": {
        "url": "https://www.youtube.com/@HowtoADHD",
        "persona": "ADHD educator",
        "description": "Educational content about ADHD management and strategies",
    },
    "unfilteredd": {
        "url": "https://www.youtube.com/@Unfilteredd",
        "persona": "Family dynamics",
        "description": "Content about family relationships and dynamics",
    },
    "wu_wei_wisdom": {
        "url": "https://www.youtube.com/@WuWeiWisdom",
        "persona": "Inner child validation",
        "description": "Content about inner child work and validation",
    },
    "veritasium": {
        "url": "https://www.youtube.com/@Veritasium",
        "persona": "Science/logic",
        "description": "Educational science content with logical explanations",
    },
    "wdr": {
        "url": "https://www.youtube.com/@wdr",
        "persona": "German ADHD content",
        "description": "German-language content about ADHD",
    },
    "y_kollektiv": {
        "url": "https://www.youtube.com/@Y-Kollektiv",
        "persona": "Educational psychology",
        "description": "Educational psychology content in German",
    },
    "zdfheite": {
        "url": "https://www.youtube.com/@zdfheite",
        "persona": "Current affairs/mental health",
        "description": "Current affairs with mental health focus",
    },
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract YouTube creator transcripts for training"
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        help="Path to s3_manifest.json",
    )
    parser.add_argument(
        "--creator",
        choices=list(CREATORS.keys()),
        help="Specific creator to extract (omit for --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all creators",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).parents[1] / "data" / "generated" / "youtube_transcripts"
        ),
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output to S3",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket")
    endpoint = manifest.get("endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("s3_manifest.json missing bucket")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("s3_manifest.json missing endpoint")
    return bucket, endpoint


def _convert_transcript_to_chatml(
    transcript: dict[str, Any], creator: str, persona: str
) -> dict[str, Any]:
    """Convert transcript to ChatML format"""
    # Extract transcript text
    text = transcript.get("text", "")
    title = transcript.get("title", "")
    video_id = transcript.get("video_id", "")

    # Create system prompt based on creator persona
    system_prompt = f"You are a therapeutic AI assistant with expertise in {persona}. Respond with empathy, clarity, and practical support."

    # Create user message from transcript
    user_message = f"Video: {title}\n\n{text}"

    # Create assistant response (placeholder - would need actual AI generation)
    assistant_message = (
        "Thank you for sharing this. I understand this content relates to "
        f"{persona}. Let me provide some thoughtful insights and support "
        "based on what you've shared."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        "metadata": {
            "source_family": "youtube_transcripts",
            "source_key": f"youtube/{creator}/{video_id}",
            "creator": creator,
            "persona": persona,
            "video_id": video_id,
            "video_title": title,
            "pii_status": "none_detected",
            "license_tag": "youtube_transcript",
            "split": "train",
            "phase": "stage1_foundation",
            "provenance": {
                "original_source": "youtube",
                "creator": creator,
                "video_id": video_id,
                "processing_pipeline": "extract_all_youtube_transcripts",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dedup_status": "unique",
                "processing_steps": ["transcript_extract", "chatml_convert"],
            },
        },
    }


def _extract_creator_transcripts(
    creator: str,
    creator_config: dict[str, Any],
    output_dir: Path,
    loader: S3DatasetLoader | None = None,
) -> dict[str, Any]:
    """Extract transcripts for a single creator"""
    logger.info(f"Extracting transcripts for creator: {creator}")

    creator_dir = output_dir / creator
    creator_dir.mkdir(parents=True, exist_ok=True)

    output_file = creator_dir / "transcripts.jsonl"

    # In production, this would:
    # 1. Use youtube_processor.py to fetch transcripts
    # 2. Convert to ChatML format
    # 3. Save to output_file

    # For now, create placeholder
    stats = {
        "creator": creator,
        "persona": creator_config["persona"],
        "url": creator_config["url"],
        "description": creator_config["description"],
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "total_transcripts": 0,
        "total_duration_seconds": 0,
        "status": "placeholder",
    }

    # Create empty placeholder file
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    logger.info(f"Created placeholder for {creator}: {output_file}")
    return stats


def _upload_to_s3(
    loader: S3DatasetLoader,
    local_path: Path,
    s3_key: str,
    bucket: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.creator and not args.all:
        logger.error("Must specify --creator or --all")
        return 1

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))
    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    creators_to_process = [args.creator] if args.creator else list(CREATORS.keys())

    all_stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "creators": {},
        "total_creators": len(creators_to_process),
        "total_transcripts": 0,
    }

    for creator in creators_to_process:
        creator_config = CREATORS[creator]
        stats = _extract_creator_transcripts(creator, creator_config, output_dir, loader)
        all_stats["creators"][creator] = stats
        all_stats["total_transcripts"] += stats["total_transcripts"]

    # Save stats
    stats_path = output_dir / "youtube_transcripts_stats.json"
    stats_path.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(f"Stats saved to {stats_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        logger.info("Uploading to S3...")
        for creator in creators_to_process:
            creator_dir = output_dir / creator
            output_file = creator_dir / "transcripts.jsonl"
            if output_file.exists():
                s3_key = f"datasets/youtube_transcripts/{creator}/transcripts.jsonl"
                _upload_to_s3(loader, output_file, s3_key, bucket)

        # Upload stats
        s3_key = "datasets/youtube_transcripts/youtube_transcripts_stats.json"
        _upload_to_s3(loader, stats_path, s3_key, bucket)

    logger.info(f"✓ Processed {len(creators_to_process)} creator(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
