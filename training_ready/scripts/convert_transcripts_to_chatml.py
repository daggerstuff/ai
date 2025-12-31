#!/usr/bin/env python3
"""Convert all transcripts in .notes/transcripts/ to ChatML format for training.

This script processes ALL transcript files from multiple sources:
- Tim Fletcher (CPTSD, narcissism, trauma, shame, recovery)
- Understood (ADHD, emotional dysregulation)
- Unfilteredd (Narcissistic family dynamics)
- Wu Wei Wisdom (Attention seeking, validation, inner child)
- Veritasium (Human connection)
- WDR (ADHD diagnosis, mental health)
- Y-Kollektiv (Educational psychology)
- ZDFheute Nachrichten (Current affairs/mental health)
- Standalone files (Complex trauma characteristics)

Outputs:
- ai/training_ready/data/generated/all_transcripts_chatml.jsonl (main output)
- ai/training_ready/data/generated/all_transcripts_stats.json (statistics)
- Optionally uploads to S3 with --upload-s3

Features:
- Automatic source detection from directory structure
- PII redaction (emails, phones, URLs)
- Intelligent chunking for long transcripts
- Source-specific system prompts
- Content hash for deduplication
- Progress logging
- Nemotron3 Data Designer integration ready

Usage Examples:
    # Process all transcripts with default settings
    python convert_transcripts_to_chatml.py

    # Process with verbose output and upload to S3
    python convert_transcripts_to_chatml.py --verbose --upload-s3

    # Process only specific source directories
    python convert_transcripts_to_chatml.py --source tim_fletcher --source Understood

    # Limit chunks per file (for testing)
    python convert_transcripts_to_chatml.py --max-chunks-per-file 3 --verbose

Notes:
- Uses ai/training_ready/data/s3_manifest.json for S3 configuration
- Integrates with existing voice profile system when available
- Compatible with NeMo Data Designer pipeline

Author: Pixelated Empathy AI Team
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# PII redaction patterns
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. (]*)?(?:\d{3}[-. )]*)?\d{3}[-. ]*\d{4}\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# Progress logging interval
PROGRESS_LOG_INTERVAL = 25

# Default transcripts directory (relative to project root)
DEFAULT_TRANSCRIPTS_DIR = ".notes/transcripts"

# Source-specific configuration
SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "tim_fletcher": {
        "display_name": "Tim Fletcher",
        "topics": ["CPTSD", "complex trauma", "narcissism", "shame", "recovery", "boundaries"],
        "style": "educational therapeutic",
        "system_prompt": (
            "You are a trauma-informed therapeutic educator specializing in Complex PTSD (CPTSD) "
            "and complex trauma recovery. Use a grounded, practical tone similar to Tim Fletcher's approach. "
            "Explain trauma concepts clearly with compassion, validate experiences, and provide actionable "
            "steps for healing. Use relatable analogies and acknowledge the difficulty while offering hope."
        ),
    },
    "Understood": {
        "display_name": "Understood",
        "topics": ["ADHD", "emotional dysregulation", "neurodivergence", "learning differences"],
        "style": "educational supportive",
        "system_prompt": (
            "You are an educational mental health assistant specializing in ADHD and neurodivergent experiences. "
            "Provide clear, supportive explanations about emotional dysregulation, executive function, and "
            "practical strategies for managing ADHD symptoms. Be validating and solution-focused."
        ),
    },
    "Unfilteredd": {
        "display_name": "Unfilteredd",
        "topics": ["narcissistic families", "family dynamics", "golden child", "scapegoat"],
        "style": "educational therapeutic",
        "system_prompt": (
            "You are a therapeutic guide specializing in narcissistic family dynamics and recovery. "
            "Help people understand dysfunctional family roles (golden child, scapegoat), recognize "
            "unhealthy patterns, and develop healthier relationship dynamics. Be validating and informative."
        ),
    },
    "Wu Wei Wisdom": {
        "display_name": "Wu Wei Wisdom",
        "topics": ["attention seeking", "validation", "inner child", "approval seeking"],
        "style": "philosophical therapeutic",
        "system_prompt": (
            "You are a therapeutic guide combining Eastern philosophy with modern psychology. "
            "Help people understand attention-seeking behaviors, the need for validation, and inner child work. "
            "Use a calm, wise tone that encourages self-reflection and self-compassion."
        ),
    },
    "Veritasium": {
        "display_name": "Veritasium",
        "topics": ["human connection", "social psychology", "science of relationships"],
        "style": "educational scientific",
        "system_prompt": (
            "You are an educational assistant explaining the science of human connection and social psychology. "
            "Use clear, evidence-based explanations about how humans connect, social networks, and relationship science."
        ),
    },
    "WDR": {
        "display_name": "WDR",
        "topics": ["ADHD diagnosis", "mental health awareness", "personal stories"],
        "style": "documentary informative",
        "system_prompt": (
            "You are a mental health educator providing informative content about ADHD diagnosis, "
            "mental health awareness, and personal mental health journeys. Be empathetic and informative."
        ),
    },
    "Y-Kollektiv": {
        "display_name": "Y-Kollektiv",
        "topics": ["educational psychology", "youth mental health", "social issues"],
        "style": "documentary educational",
        "system_prompt": (
            "You are an educational guide discussing psychology, youth mental health, and social issues. "
            "Provide thoughtful, nuanced perspectives on complex topics affecting young people."
        ),
    },
    "ZDFheute Nachrichten": {
        "display_name": "ZDFheute",
        "topics": ["current affairs", "mental health in news", "societal issues"],
        "style": "journalistic informative",
        "system_prompt": (
            "You are an informative assistant discussing current affairs and their mental health implications. "
            "Provide balanced, thoughtful perspectives on news events and societal issues."
        ),
    },
    "10_ Happier": {
        "display_name": "Ten Percent Happier",
        "topics": ["meditation", "mindfulness", "somatic experiencing", "nervous system"],
        "style": "conversational therapeutic",
        "system_prompt": (
            "You are a mindfulness and meditation guide specializing in somatic experiencing and nervous system regulation. "
            "Help people understand their bodies' stress responses and provide practical techniques for regulation."
        ),
    },
    "_default": {
        "display_name": "Mental Health Education",
        "topics": ["complex trauma", "mental health", "psychology", "healing"],
        "style": "educational therapeutic",
        "system_prompt": (
            "You are a compassionate mental health educator specializing in trauma-informed care. "
            "Explain psychological concepts clearly, validate experiences, and provide practical guidance "
            "for healing and personal growth. Use a warm, supportive tone."
        ),
    },
}

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Clean and redact PII from transcript text."""
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = TIMESTAMP_RE.sub("", text)
    # Normalize whitespace
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _chunk_text(
    text: str,
    *,
    max_chars: int = 2000,
    min_chars: int = 400,
    overlap_chars: int = 100,
) -> list[str]:
    """Split text into chunks based on paragraphs with optional overlap."""
    # Split on paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for p in paras:
        para_len = len(p)

        # If adding this paragraph exceeds max and we have enough content, save chunk
        if cur_len + para_len + 2 > max_chars and cur_len >= min_chars:
            chunk_text = "\n\n".join(cur).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Start new chunk, optionally with overlap from last paragraph
            if overlap_chars > 0 and cur:
                last_para = cur[-1]
                if len(last_para) <= overlap_chars * 2:
                    cur = [last_para]
                    cur_len = len(last_para)
                else:
                    cur = []
                    cur_len = 0
            else:
                cur = []
                cur_len = 0

        cur.append(p)
        cur_len += para_len + 2

    # Add remaining content if substantial
    if cur_len >= min_chars:
        chunk_text = "\n\n".join(cur).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def _title_from_path(path: Path) -> str:
    """Extract a human-readable title from file path."""
    name = path.stem
    # Clean up common patterns
    name = name.replace("_", " ").replace("-", " ")
    # Remove leading numbers/prefixes like "10_" or "characteristics_of_"
    name = re.sub(r"^\d+\s*", "", name)
    # Capitalize first letter of each word
    name = " ".join(word.capitalize() for word in name.split())
    return name.strip() or "Mental Health Topic"


def _content_hash(content: str) -> str:
    """Generate SHA256 hash of content for deduplication."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"


def _detect_source(file_path: Path, transcripts_root: Path) -> str:
    """Detect source category from file path."""
    try:
        rel_path = file_path.relative_to(transcripts_root)
        parts = rel_path.parts

        # Check if file is in a subdirectory
        if len(parts) > 1:
            source_dir = parts[0]
            # Match against known sources
            for key in SOURCE_CONFIG:
                if key != "_default" and key.lower() in source_dir.lower():
                    return key
            # Return directory name if not a known source
            return source_dir

        # Standalone file in root - check filename for patterns
        filename = file_path.stem.lower()
        if "tim_fletcher" in filename or "tim fletcher" in filename:
            return "tim_fletcher"
        if "complex_trauma" in filename or "characteristics" in filename:
            return "tim_fletcher"  # These are typically Tim Fletcher content

        return "_default"
    except ValueError:
        return "_default"


def _get_source_config(source: str) -> dict[str, Any]:
    """Get configuration for a source, falling back to default."""
    return SOURCE_CONFIG.get(source, SOURCE_CONFIG["_default"])


def _create_user_prompt(title: str, source_config: dict[str, Any]) -> str:
    """Create a contextual user prompt based on the title and source."""
    topics = source_config.get("topics", ["mental health"])
    topic_hint = topics[0] if topics else "mental health"

    # Vary the prompt style
    prompts = [
        f"Can you explain this topic about {topic_hint}: {title}?",
        f"I'd like to understand more about {title}. Can you help?",
        f"Please teach me about {title} and how it relates to {topic_hint}.",
        f"What should I know about {title}?",
        f"Help me understand the concept of {title}.",
    ]

    # Use hash of title to consistently pick same prompt for same content
    idx = hash(title) % len(prompts)
    return prompts[idx]


def _find_transcript_files(transcripts_dir: Path) -> list[Path]:
    """Find all transcript files in the directory."""
    files: list[Path] = []

    if not transcripts_dir.exists():
        logger.error(f"Transcripts directory not found: {transcripts_dir}")
        return files

    # Find all .txt files recursively
    for f in transcripts_dir.rglob("*.txt"):
        # Skip backup files and scripts
        if f.suffix == ".orig" or f.name.endswith(".py"):
            continue
        # Skip hidden files
        if f.name.startswith("."):
            continue
        files.append(f)

    # Also find .md files (some transcripts might be markdown)
    for f in transcripts_dir.rglob("*.md"):
        if f.suffix == ".orig" or f.name.startswith("."):
            continue
        files.append(f)

    return sorted(files)


def _load_transcript(file_path: Path) -> str | None:
    """Load transcript text from file."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None


def _upload_to_s3(
    local_path: Path,
    s3_key: str,
    bucket: str,
    endpoint: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        # Import here to avoid dependency issues
        from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

        loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def _load_s3_config(manifest_path: Path) -> tuple[str, str]:
    """Load S3 configuration from manifest."""
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bucket = manifest.get("bucket", "pixel-data")
            endpoint = manifest.get("endpoint", "https://s3.us-east-va.io.cloud.ovh.us")
            return bucket, endpoint
        except (OSError, json.JSONDecodeError):
            pass
    return "pixel-data", "https://s3.us-east-va.io.cloud.ovh.us"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert all transcripts to ChatML format for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process all transcripts
  %(prog)s --verbose --upload-s3              # Process with S3 upload
  %(prog)s --source tim_fletcher --source Understood  # Process specific sources
  %(prog)s --max-chunks-per-file 3 --verbose  # Limit chunks (for testing)
""",
    )
    parser.add_argument(
        "--transcripts-dir",
        default=DEFAULT_TRANSCRIPTS_DIR,
        metavar="DIR",
        help=f"Directory containing transcripts (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.add_argument(
        "--output",
        default="ai/training_ready/data/generated/all_transcripts_chatml.jsonl",
        metavar="PATH",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        metavar="NAME",
        help="Process only specific source directories (repeatable)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        metavar="N",
        help="Maximum number of files to process (0 = unlimited)",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=8,
        metavar="N",
        help="Maximum chunks to extract per file (default: 8)",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=400,
        metavar="N",
        help="Minimum characters per chunk (default: 400)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=2000,
        metavar="N",
        help="Maximum characters per chunk (default: 2000)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output files to S3 after generation",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="gdrive/processed/transcripts",
        metavar="PREFIX",
        help="S3 prefix for uploaded output",
    )
    parser.add_argument(
        "--manifest",
        default="ai/training_ready/data/s3_manifest.json",
        metavar="PATH",
        help="Path to S3 manifest JSON for bucket/endpoint config",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress logging",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s" if args.verbose else "%(message)s",
    )

    # Resolve paths relative to project root
    project_root = Path(__file__).parents[3]  # ai/training_ready/scripts -> project root
    transcripts_dir = project_root / args.transcripts_dir
    output_path = project_root / args.output
    manifest_path = project_root / args.manifest

    logger.info("=" * 60)
    logger.info("TRANSCRIPT TO CHATML CONVERSION")
    logger.info("=" * 60)
    logger.info(f"Transcripts directory: {transcripts_dir}")
    logger.info(f"Output file: {output_path}")

    # Find all transcript files
    all_files = _find_transcript_files(transcripts_dir)

    if not all_files:
        logger.error("No transcript files found!")
        return 1

    logger.info(f"Found {len(all_files)} transcript files")

    # Filter by source if specified
    if args.sources:
        filtered_files = []
        for f in all_files:
            source = _detect_source(f, transcripts_dir)
            if any(s.lower() in source.lower() for s in args.sources):
                filtered_files.append(f)
        all_files = filtered_files
        logger.info(f"Filtered to {len(all_files)} files from sources: {args.sources}")

    # Limit files if specified
    if args.max_files > 0:
        all_files = all_files[: args.max_files]
        logger.info(f"Limited to {len(all_files)} files")

    if not all_files:
        logger.error("No files to process after filtering!")
        return 1

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    stats: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "transcripts_dir": str(transcripts_dir),
        "output_file": str(output_path),
        "sources_processed": Counter(),
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_written": 0,
        "total_chars": 0,
        "chunk_histogram": Counter(),
    }

    seen_hashes: set[str] = set()
    written = 0
    last_progress = 0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, file_path in enumerate(all_files, 1):
            # Progress logging
            if idx - last_progress >= PROGRESS_LOG_INTERVAL or args.verbose:
                logger.info(f"Processing {idx}/{len(all_files)}: {file_path.name}")
                last_progress = idx

            # Load transcript
            raw_text = _load_transcript(file_path)
            if not raw_text:
                stats["files_skipped"] += 1
                continue

            # Clean text
            cleaned = _clean_text(raw_text)
            if len(cleaned) < args.min_chunk_chars:
                if args.verbose:
                    logger.debug(f"  Skipped (too short): {len(cleaned)} chars")
                stats["files_skipped"] += 1
                continue

            # Detect source and get config
            source = _detect_source(file_path, transcripts_dir)
            source_config = _get_source_config(source)
            system_prompt = source_config["system_prompt"]

            # Chunk the text
            chunks = _chunk_text(
                cleaned,
                max_chars=args.max_chunk_chars,
                min_chars=args.min_chunk_chars,
            )

            if not chunks:
                stats["files_skipped"] += 1
                continue

            # Limit chunks per file
            chunks = chunks[: args.max_chunks_per_file]
            title = _title_from_path(file_path)

            stats["files_processed"] += 1
            stats["sources_processed"][source] += 1
            stats["chunk_histogram"][str(min(len(chunks), 10))] += 1

            file_written = 0
            for chunk_idx, chunk in enumerate(chunks):
                # Deduplication
                content_hash = _content_hash(chunk)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # Create user prompt
                user_prompt = _create_user_prompt(title, source_config)

                # Build ChatML record
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": chunk},
                    ],
                    "metadata": {
                        "source_family": "video_transcripts",
                        "source_category": source,
                        "source_display": source_config["display_name"],
                        "source_file": file_path.name,
                        "content_hash": content_hash,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "title": title,
                        "topics": source_config.get("topics", []),
                        "pii_status": "scrubbed",
                        "license_tag": "transcript_corpus",
                        "split": "train",
                        "phase": "stage4_voice_personas",
                        "provenance": {
                            "original_source": str(file_path.relative_to(project_root)),
                            "processing_pipeline": "convert_transcripts_to_chatml",
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "dedup_status": "unique",
                            "processing_steps": ["text_clean", "pii_redact", "chunk", "chatml_convert"],
                        },
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                file_written += 1
                stats["total_chars"] += len(chunk)

            stats["chunks_written"] += file_written

            if args.verbose and file_written > 0:
                logger.debug(f"  Wrote {file_written} chunks from {source}")

    # Finalize stats
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["examples_written"] = written
    stats["unique_hashes"] = len(seen_hashes)
    stats["sources_processed"] = dict(stats["sources_processed"])
    stats["chunk_histogram"] = dict(sorted(stats["chunk_histogram"].items(), key=lambda x: int(x[0])))

    # Save stats
    stats_path = output_path.with_name("all_transcripts_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONVERSION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Generated {written:,} ChatML training examples")
    logger.info(f"  Files processed: {stats['files_processed']:,}")
    logger.info(f"  Files skipped: {stats['files_skipped']:,}")
    logger.info(f"  Total characters: {stats['total_chars']:,}")
    logger.info("")
    logger.info("Sources breakdown:")
    for source, count in sorted(stats["sources_processed"].items(), key=lambda x: -x[1]):
        display = _get_source_config(source)["display_name"]
        logger.info(f"  {display}: {count} files")
    logger.info("")
    logger.info(f"Output: {output_path}")
    logger.info(f"Stats: {stats_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        logger.info("")
        logger.info("Uploading to S3...")
        bucket, endpoint = _load_s3_config(manifest_path)

        # Upload main output
        output_s3_key = f"{args.s3_output_prefix}/all_transcripts_chatml.jsonl"
        success1 = _upload_to_s3(output_path, output_s3_key, bucket, endpoint)

        # Upload stats
        stats_s3_key = f"{args.s3_output_prefix}/all_transcripts_stats.json"
        success2 = _upload_to_s3(stats_path, stats_s3_key, bucket, endpoint)

        if success1 and success2:
            logger.info(f"✓ Uploaded to s3://{bucket}/{args.s3_output_prefix}/")
        else:
            logger.error("Some uploads failed")
            return 1

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
