#!/usr/bin/env python3
"""Stage-aware deduplication for training data ingestion.

Implements the Master Training Plan's deduplication requirements:
- Primary hash: sha256(lowercase(concat(messages.role + messages.content)))
- Secondary hash: sha1(conversation_id + stage + source + crisis_intensity)
- Conflict resolution: stage4 > stage3 > stage2 > stage1 > supplementary

This ensures higher-priority specialized data (voice/persona, edge cases) survives
collisions with lower-priority foundation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

STAGE_PRIORITY = {
    "stage4_voice_persona": 5,
    "stage3_edge_stress_test": 4,
    "stage2_therapeutic_expertise": 3,
    "stage1_foundation": 2,
    "supplementary": 1,
}


@dataclass
class DeduplicationStats:
    """Statistics from deduplication run."""

    total_records: int = 0
    unique_records: int = 0
    duplicates_removed: int = 0
    stage_conflicts_resolved: int = 0
    records_by_stage: dict[str, int] = field(default_factory=dict)


def compute_primary_hash(record: dict) -> str:
    """Compute primary content hash: sha256(lowercase(concat(messages.role + messages.content))).

    Args:
        record: Training data record with 'messages' field.

    Returns:
        Hex digest of SHA-256 hash.
    """
    messages = record.get("messages", [])
    if not messages:
        return hashlib.sha256(b"").hexdigest()

    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            parts.append(msg.get("role", ""))
            parts.append(msg.get("content", ""))
    concatenated = "".join(parts)
    normalized = concatenated.lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_secondary_hash(record: dict) -> str:
    """Compute secondary metadata hash: sha1(conversation_id + stage + source + crisis_intensity).

    Args:
        record: Training data record with metadata fields.

    Returns:
        Hex digest of SHA-1 hash.
    """
    metadata = record.get("metadata", {})
    conversation_id = str(metadata.get("conversation_id", ""))
    stage = str(metadata.get("stage", ""))
    source = str(metadata.get("source", ""))
    crisis_intensity = str(metadata.get("crisis_intensity", ""))

    concatenated = f"{conversation_id}{stage}{source}{crisis_intensity}"
    return hashlib.sha1(concatenated.encode("utf-8")).hexdigest()


def get_stage_priority(record: dict) -> int:
    """Get numeric priority for a record's stage.

    Args:
        record: Training data record with metadata.stage field.

    Returns:
        Priority value (higher = more important).
    """
    metadata = record.get("metadata", {})
    stage = metadata.get("stage", "supplementary")
    return STAGE_PRIORITY.get(stage, 1)


def deduplicate_records(
    records: list[dict],
    use_secondary_hash: bool = False,
) -> tuple[list[dict], DeduplicationStats]:
    """Deduplicate records with stage-aware conflict resolution.

    When duplicates are found, the record with higher stage priority is kept.

    Args:
        records: List of training data records.
        use_secondary_hash: If True, use secondary hash for deduplication.

    Returns:
        Tuple of (deduplicated records, statistics).
    """
    stats = DeduplicationStats()
    stats.total_records = len(records)

    hash_to_record: dict[str, dict] = {}
    hash_to_stage: dict[str, int] = {}

    for record in records:
        record_hash = compute_secondary_hash(record) if use_secondary_hash else compute_primary_hash(record)
        record_stage = get_stage_priority(record)
        metadata = record.get("metadata", {})
        stage_name = metadata.get("stage", "supplementary")

        stats.records_by_stage[stage_name] = stats.records_by_stage.get(stage_name, 0) + 1

        if record_hash not in hash_to_record:
            hash_to_record[record_hash] = record
            hash_to_stage[record_hash] = record_stage
        else:
            existing_stage = hash_to_stage[record_hash]
            if record_stage > existing_stage:
                hash_to_record[record_hash] = record
                hash_to_stage[record_hash] = record_stage
                stats.stage_conflicts_resolved += 1
            else:
                stats.duplicates_removed += 1

    stats.unique_records = len(hash_to_record)

    return list(hash_to_record.values()), stats


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    use_secondary_hash: bool = False,
) -> DeduplicationStats:
    """Process a single JSONL file with stage-aware deduplication.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        use_secondary_hash: If True, use secondary hash for deduplication.

    Returns:
        Deduplication statistics.
    """
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    deduped, stats = deduplicate_records(records, use_secondary_hash)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record) + "\n")

    logger.info(
        f"Deduplication: {stats.total_records} → {stats.unique_records} "
        f"({stats.duplicates_removed} removed, {stats.stage_conflicts_resolved} stage conflicts resolved)"
    )

    return stats


def main() -> None:
    """CLI entry point for stage-aware deduplication."""
    parser = argparse.ArgumentParser(
        description="Stage-aware deduplication for training data ingestion",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file",
    )
    parser.add_argument(
        "--use-secondary-hash",
        action="store_true",
        help="Use secondary hash (metadata-based) instead of primary hash (content-based)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    stats = process_jsonl_file(args.input, args.output, args.use_secondary_hash)

    print(f"\nDeduplication Summary:")
    print(f"  Total records: {stats.total_records}")
    print(f"  Unique records: {stats.unique_records}")
    print(f"  Duplicates removed: {stats.duplicates_removed}")
    print(f"  Stage conflicts resolved: {stats.stage_conflicts_resolved}")
    print(f"\nRecords by stage:")
    for stage, count in sorted(stats.records_by_stage.items()):
        print(f"  {stage}: {count}")


if __name__ == "__main__":
    main()
