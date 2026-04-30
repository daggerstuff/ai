#!/usr/bin/env python3
"""Run semantic deduplication on the CPTSD JSONL dataset.

Reads the generated cptsd_transcripts.jsonl, runs content-hash
and semantic (cosine similarity) dedup, writes a clean output.

Usage:
    uv run ai/training/ready_packages/scripts/run_cptsd_dedup.py
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "generated"
INPUT_FILE = DATA_DIR / "cptsd_transcripts.jsonl"
OUTPUT_FILE = DATA_DIR / "cptsd_transcripts_deduped.jsonl"

# Cosine similarity threshold for semantic dedup
SEMANTIC_THRESHOLD = 0.92


def _content_hash(record: dict) -> str:
    """Generate hash from message content for exact dedup."""
    msgs = record.get("messages", [])
    content_parts = []
    for msg in msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        content_parts.append(f"{role}:{content}")
    combined = "|".join(content_parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _extract_text(record: dict) -> str:
    """Extract all message content as plain text."""
    parts = []
    for msg in record.get("messages", []):
        content = msg.get("content", "")
        if content and msg.get("role") != "system":
            parts.append(content)
    return " ".join(parts)


def exact_dedup(records: list[dict]) -> list[dict]:
    """Remove exact duplicates by content hash."""
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        h = _content_hash(rec)
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    if removed := len(records) - len(unique):
        logger.info(
            "Exact dedup: removed %d/%d records",
            removed,
            len(records),
        )
    return unique


def semantic_dedup(
    records: list[dict],
    threshold: float = SEMANTIC_THRESHOLD,
) -> list[dict]:
    """Remove near-duplicates using TF-IDF cosine similarity.

    Falls back to exact-only dedup if sklearn is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import (
            TfidfVectorizer,
        )
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.warning("sklearn not available, skipping semantic dedup")
        return records

    texts = [_extract_text(r) for r in records]
    if not texts:
        return records

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Find pairs above threshold
    to_remove: set[int] = set()
    n = len(records)
    # Process in batches to avoid memory issues
    batch_size = 500
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sim = cosine_similarity(tfidf_matrix[start:end], tfidf_matrix)
        for i in range(batch_sim.shape[0]):
            abs_i = start + i
            if abs_i in to_remove:
                continue
            for j in range(abs_i + 1, n):
                if j in to_remove:
                    continue
                if batch_sim[i][j] >= threshold:
                    to_remove.add(j)

    if to_remove:
        logger.info(
            "Semantic dedup: removing %d/%d near-duplicates (threshold=%.2f)",
            len(to_remove),
            len(records),
            threshold,
        )

    return [r for idx, r in enumerate(records) if idx not in to_remove]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not INPUT_FILE.exists():
        logger.error("Input not found: %s", INPUT_FILE)
        return 1

    with open(INPUT_FILE) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    logger.info("Loaded %d records from %s", len(records), INPUT_FILE.name)

    # Step 1: Exact dedup
    records = exact_dedup(records)
    logger.info("After exact dedup: %d records", len(records))

    # Step 2: Semantic dedup
    records = semantic_dedup(records)
    logger.info("After semantic dedup: %d records", len(records))

    # Write output
    with open(OUTPUT_FILE, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Wrote %d records to %s",
        len(records),
        OUTPUT_FILE.name,
    )

    # Write stats
    stats_file = DATA_DIR / "cptsd_dedup_stats.json"
    with open(stats_file, "w") as fh:
        json.dump(
            {
                "input_file": str(INPUT_FILE),
                "output_file": str(OUTPUT_FILE),
                "input_count": len(records),
                "output_count": len(records),
                "threshold": SEMANTIC_THRESHOLD,
            },
            fh,
            indent=2,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
