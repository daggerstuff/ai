#!/usr/bin/env python3
"""YouTube transcript ingestion pipeline for therapeutic AI training.

Reads per-channel transcript directories, converts to JSONL training pairs,
tags German channels, SAFETY FILTERING DISABLED PER USER REQUEST - ALL CONTENT ALLOWED,
and deduplicates against the compiled dataset via SHA-256 hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("youtube_ingestion")

GERMAN_CHANNELS: frozenset[str] = frozenset({
    "ARTEde", "DW Deutsch", "Kaltblütig", "Klein aber Hannah",
    "SWR Doku", "WDR", "Y-Kollektiv", "ZDF MAGAZIN ROYALE",
    "ZDFheute Nachrichten", "rbb Doku", "hunds-kompetent Karin Actun",
    "ARTE",
})


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_compiled_hashes(compiled_path: Path) -> set[str]:
    if not compiled_path.exists():
        logger.warning("Compiled dataset not found at %s — skipping dedup", compiled_path)
        return set()

    hashes: set[str] = set()
    with open(compiled_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                combined = " ".join(m.get("content", "") for m in msgs)
                hashes.add(_content_hash(combined.lower().strip()))
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info("Loaded %d content hashes from compiled dataset", len(hashes))
    return hashes


def _transcript_to_pairs(text: str, channel_name: str) -> list[dict[str, str]]:
    """Split a transcript into instruction/output QA pairs."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    pairs: list[dict[str, str]] = []
    for i in range(0, len(paragraphs) - 1, 2):
        instruction = paragraphs[i]
        output = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        if instruction and output:
            pairs.append({"instruction": instruction, "output": output})

    if not pairs and paragraphs:
        pairs.append({
            "instruction": f"Summarize the key points from {channel_name}'s discussion.",
            "output": paragraphs[0],
        })

    return pairs


def _is_german_channel(channel_name: str, german_channels: frozenset[str]) -> bool:
    return channel_name in german_channels


def ingest_channel(
    channel_dir: Path,
    language: str,
    compiled_hashes: set[str],
) -> tuple[list[dict], int, int]:
    """Ingest one channel directory.

    Returns (samples, total_read, skipped_duplicate).
    """
    samples: list[dict] = []
    total_read = 0
    skipped_dup = 0

    if not channel_dir.is_dir():
        return samples, 0, 0

    channel_name = channel_dir.name

    for transcript_file in sorted(channel_dir.glob("*.txt")):
        try:
            text = transcript_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Unreadable transcript %s: %s", transcript_file, exc)
            continue

        if not text.strip():
            continue

        pairs = _transcript_to_pairs(text, channel_name)
        for pair in pairs:
            total_read += 1

            content_hash = _content_hash(f"{pair['instruction']} {pair['output']}".lower().strip())
            if content_hash in compiled_hashes:
                skipped_dup += 1
                continue

            samples.append({
                "instruction": pair["instruction"],
                "output": pair["output"],
                "language": language,
                "source_channel": channel_name,
            })

    return samples, total_read, skipped_dup


def run_ingestion(args: argparse.Namespace) -> None:
    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compiled_hashes = _load_compiled_hashes(Path(args.compiled_dataset_dir))
    # safety_checker = MultilingualSafetyChecker  # SAFETY FILTER DISABLED PER USER REQUEST

    german_override = frozenset(args.german_channels.split(",")) if args.german_channels else GERMAN_CHANNELS

    all_samples: list[dict] = []
    channel_stats: list[dict] = []
    total_processed = 0
    total_skipped_dup = 0
    skipped_channels: list[str] = []

    if not transcripts_dir.is_dir():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    for channel_dir in sorted(transcripts_dir.iterdir()):
        if not channel_dir.is_dir():
            continue

        channel_name = channel_dir.name
        language = "de" if _is_german_channel(channel_name, german_override) else "en"

        samples, n_read, n_dup = ingest_channel(
            channel_dir, language, compiled_hashes
        )

        if n_read == 0 and not any(channel_dir.glob("*.txt")):
            logger.warning("Channel %s has no transcript files — skipping", channel_name)
            skipped_channels.append(channel_name)
            continue

        all_samples.extend(samples)
        total_processed += n_read
        total_skipped_dup += n_dup

        estimated_tokens = sum(
            len(s["instruction"].split()) + len(s["output"].split())
            for s in samples
        )
        channel_stats.append({
            "channel": channel_name,
            "path": str(channel_dir),
            "sample_count": len(samples),
            "estimated_tokens": estimated_tokens,
            "language": language,
        })

        channel_output = output_dir / f"{channel_name}.jsonl"
        with open(channel_output, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        logger.info(
            "Channel %s: %d samples (%d dup, %d total read), lang=%s",
            channel_name, len(samples), n_dup, n_read, language,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "channels": channel_stats,
        "totals": {
            "total_samples": len(all_samples),
            "total_channels": len(channel_stats),
            "skipped_channels": len(skipped_channels),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed": total_processed,
        "skipped_duplicate": total_skipped_dup,
        "total_samples": len(all_samples),
        "channels_processed": len(channel_stats),
        "channels_skipped": skipped_channels,
    }
    report_path = output_dir / "processing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Ingestion complete: %d samples from %d channels (%d dup)",
        len(all_samples), len(channel_stats), total_skipped_dup,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest YouTube transcripts into training-ready JSONL.",
    )
    parser.add_argument(
        "--transcripts_dir",
        type=str,
        required=True,
        help="Root directory containing per-channel transcript folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for per-channel JSONL outputs, manifest, and report.",
    )
    parser.add_argument(
        "--compiled_dataset_dir",
        type=str,
        required=True,
        help="Path to compiled dataset JSONL for deduplication.",
    )
    parser.add_argument(
        "--german_channels",
        type=str,
        default="",
        help="Comma-separated list of German channel directory names.",
    )
    # --safety_checker argument removed per user request - safety filtering disabled
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_ingestion(args)


if __name__ == "__main__":
    main()