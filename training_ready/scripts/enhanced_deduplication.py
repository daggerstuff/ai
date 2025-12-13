#!/usr/bin/env python3
"""
Enhanced Deduplication with Near-Duplicate Detection and Split Leakage Prevention
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class ConversationEntry:
    """Represents a conversation entry with metadata"""

    messages: list[dict[str, str]]
    source_family: str
    source_key: str
    content_hash: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.lower().strip()


def compute_content_hash(messages: list[dict[str, str]]) -> str:
    """Compute SHA256 hash of normalized conversation content (optimized)"""
    # Optimized: avoid intermediate list and string allocations
    # Use a generator for memory efficiency with large conversations
    content_parts = []
    for msg in messages:
        # Direct normalization without extra function call overhead
        if (
            isinstance(msg, dict)
            and "content" in msg
            and isinstance(content := msg["content"], str)
            and (normalized := content.lower().strip())
        ):  # Skip empty content
            content_parts.append(normalized)

    if not content_parts:
        # Empty conversation - return a consistent hash
        hash_digest = hashlib.sha256(b"").hexdigest()
    else:
        # For very large conversations, sort in-place to save memory
        # But sorting is necessary for consistent hashing
        content_parts.sort()  # In-place sort is more memory efficient
        # Use join with a generator-like approach for very large strings
        normalized = " ".join(content_parts)
        # Hash directly without storing the full normalized string if it's huge
        hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return f"sha256:{hash_digest}"


def extract_text_for_similarity(messages: list[dict[str, str]]) -> str:
    """Extract all text content for semantic similarity comparison"""
    content_parts = [
        msg["content"] for msg in messages if isinstance(msg, dict) and "content" in msg
    ]
    return " ".join(content_parts)


def compute_simple_similarity(text1: str, text2: str) -> float:
    """
    Compute simple similarity score using word overlap.
    For production, use sentence-transformers or similar.
    """
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    return len(intersection) / len(union) if union else 0.0


class EnhancedDeduplicator:
    """Enhanced deduplication with exact + near-duplicate detection"""

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.exact_duplicates: dict[str, list[ConversationEntry]] = defaultdict(list)
        self.near_duplicates: list[tuple[ConversationEntry, ConversationEntry, float]] = []
        self.processed_entries: list[ConversationEntry] = []

    def add_conversation(self, entry: ConversationEntry) -> None:
        """Add a conversation entry for deduplication"""
        self.processed_entries.append(entry)
        self.exact_duplicates[entry.content_hash].append(entry)

    def find_exact_duplicates(self) -> dict[str, list[ConversationEntry]]:
        """Find exact duplicates (same content hash)"""
        return {
            hash_val: entries
            for hash_val, entries in self.exact_duplicates.items()
            if len(entries) > 1
        }

    def find_near_duplicates(self) -> list[tuple[ConversationEntry, ConversationEntry, float]]:
        """
        Find near-duplicates using similarity threshold.
        Note: This is a simple implementation. For production, use sentence-transformers.
        """
        logger.info(
            f"Finding near-duplicates (similarity threshold: {self.similarity_threshold})..."
        )

        near_dups = []
        entries_text = [
            (entry, extract_text_for_similarity(entry.messages)) for entry in self.processed_entries
        ]

        # Compare all pairs (O(n²) - optimize for production)
        for i, (entry1, text1) in enumerate(entries_text):
            for _, (entry2, text2) in enumerate(entries_text[i + 1 :], start=i + 1):
                # Skip if same exact hash
                if entry1.content_hash == entry2.content_hash:
                    continue

                similarity = compute_simple_similarity(text1, text2)
                if similarity >= self.similarity_threshold:
                    near_dups.append((entry1, entry2, similarity))

        self.near_duplicates = near_dups
        return near_dups

    def check_split_leakage(self, holdout_families: list[str]) -> dict[str, Any]:
        """
        Check for split leakage violations.

        Args:
            holdout_families: List of families that should only be in test split

        Returns:
            Dictionary with leakage violations
        """
        violations = {
            "exact_duplicate_leakage": [],
            "near_duplicate_leakage": [],
            "holdout_family_leakage": [],
        }

        # Check exact duplicates across splits
        for hash_val, entries in self.exact_duplicates.items():
            if len(entries) > 1:
                splits = {entry.split for entry in entries if entry.split}
                if len(splits) > 1:
                    violations["exact_duplicate_leakage"].append(
                        {
                            "hash": hash_val,
                            "splits": list(splits),
                            "count": len(entries),
                            "source_families": list({e.source_family for e in entries}),
                        }
                    )

        # Check near-duplicates across splits
        for entry1, entry2, similarity in self.near_duplicates:
            if entry1.split and entry2.split and entry1.split != entry2.split:
                violations["near_duplicate_leakage"].append(
                    {
                        "entry1": {
                            "source_family": entry1.source_family,
                            "source_key": entry1.source_key,
                            "split": entry1.split,
                        },
                        "entry2": {
                            "source_family": entry2.source_family,
                            "source_key": entry2.source_key,
                            "split": entry2.split,
                        },
                        "similarity": similarity,
                    }
                )

        # Check holdout families in wrong splits
        for entry in self.processed_entries:
            if entry.source_family in holdout_families and entry.split and entry.split != "test":
                violations["holdout_family_leakage"].append(
                    {
                        "source_family": entry.source_family,
                        "source_key": entry.source_key,
                        "split": entry.split,
                        "expected_split": "test",
                    }
                )

        return violations

    def deduplicate(self, strategy: str = "keep_first") -> list[ConversationEntry]:
        """
        Deduplicate entries based on strategy.

        Args:
            strategy: 'keep_first', 'keep_best_quality', 'keep_longest'

        Returns:
            List of deduplicated entries
        """
        logger.info(f"Deduplicating with strategy: {strategy}")

        deduplicated = []
        seen_hashes = set()

        # Process exact duplicates
        strategy_map = {
            "keep_first": lambda entries: entries[0],
            "keep_longest": lambda entries: max(
                entries, key=lambda e: len(extract_text_for_similarity(e.messages))
            ),
            "keep_best_quality": lambda entries: max(
                entries, key=lambda e: e.metadata.get("quality_score", 0.0)
            ),
        }
        get_keep_entry = strategy_map.get(strategy, lambda entries: entries[0])

        for hash_val, entries in self.exact_duplicates.items():
            if len(entries) > 1:
                # Choose which entry to keep
                keep_entry = get_keep_entry(entries)
                deduplicated.append(keep_entry)
                logger.debug(
                    f"Removed {len(entries) - 1} exact duplicates for hash {hash_val[:16]}..."
                )
            else:
                deduplicated.append(entries[0])
            seen_hashes.add(hash_val)

        # Process near-duplicates (remove one of each pair)
        # Keep entry1, mark entry2 for removal
        near_dup_hashes = {
            entry2.content_hash
            for _, entry2, _ in self.near_duplicates
            if entry2.content_hash not in seen_hashes
        }

        # Filter out near-duplicates
        final_deduplicated = [
            entry for entry in deduplicated if entry.content_hash not in near_dup_hashes
        ]

        logger.info(
            f"Deduplication complete: {len(self.processed_entries)} -> {len(final_deduplicated)} entries"
        )
        return final_deduplicated


def load_conversations_from_jsonl(
    file_path: Path, source_family: str, source_key: str
) -> list[ConversationEntry]:
    """Load conversations from JSONL file"""
    entries = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages:
                    continue

                content_hash = compute_content_hash(messages)
                entry = ConversationEntry(
                    messages=messages,
                    source_family=source_family,
                    source_key=source_key,
                    content_hash=content_hash,
                    split=data.get("metadata", {}).get("split"),
                    metadata=data.get("metadata", {}),
                )
                entries.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")

    return entries


def main():
    """Example usage"""
    project_root = Path(__file__).parents[3]

    # Example: Load some conversations
    # In production, this would load from S3
    deduplicator = EnhancedDeduplicator(similarity_threshold=0.95)

    # Find exact duplicates
    exact_dups = deduplicator.find_exact_duplicates()
    logger.info(f"Found {len(exact_dups)} exact duplicate groups")

    # Find near-duplicates
    near_dups = deduplicator.find_near_duplicates()
    logger.info(f"Found {len(near_dups)} near-duplicate pairs")

    # Check split leakage
    holdout_families = ["long_running_therapy", "edge_case_crisis", "sarcasm", "voice_persona"]
    violations = deduplicator.check_split_leakage(holdout_families)

    logger.info("Split leakage violations:")
    logger.info(f"  Exact duplicate leakage: {len(violations['exact_duplicate_leakage'])}")
    logger.info(f"  Near-duplicate leakage: {len(violations['near_duplicate_leakage'])}")
    logger.info(f"  Holdout family leakage: {len(violations['holdout_family_leakage'])}")

    # Save violations report
    output_path = project_root / "ai" / "training_ready" / "data" / "deduplication_violations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(violations, f, indent=2, default=str)

    logger.info(f"Violations report saved to {output_path}")


if __name__ == "__main__":
    main()
