"""
PIX-32: Normalization Pipeline — orchestration layer combining normalization + dedup.

Coordinates the full PIX-32 pipeline:
  1. JSONL ingestion (single files or directories)
  2. Schema validation via DataNormalizer
  3. Text normalization and key standardization
  4. Deduplication (BloomFilter for speed, ConversationDeduplicator for quality,
     or stage-aware hash dedup for training data)
  5. Provenance metadata attachment
  6. Output to normalized JSONL with rejection report
"""

from __future__ import annotations

import glob as glob_mod
import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from .data_normalizer import (
    Conversation,
    DataNormalizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deduplication strategy enum
# ---------------------------------------------------------------------------


class DedupStrategy(StrEnum):
    """Available deduplication strategies."""

    NONE = "none"
    BLOOM = "bloom"  # Fast hash-based (BloomFilter)
    SIMILARITY = "similarity"  # Multi-metric similarity (ConversationDeduplicator)
    STAGE_AWARE = "stage_aware"  # SHA256/SHA1 hash with stage priority


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class DuplicateEvidence:
    """Audit record describing an explicit duplicate decision."""

    strategy: str
    content_hash: str
    retained_id: str
    duplicate_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy": self.strategy,
            "content_hash": self.content_hash,
            "retained_id": self.retained_id,
            "duplicate_id": self.duplicate_id,
            "reason": self.reason,
        }


@dataclass
class PipelineResult:
    """Aggregated result of the full normalization + dedup pipeline."""

    input_files: list[str] = field(default_factory=list)
    total_records: int = 0
    validated_records: int = 0
    rejected_records: int = 0
    duplicates_removed: int = 0
    final_records: int = 0
    processing_time_seconds: float = 0.0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    duplicate_evidence: list[DuplicateEvidence] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def dedup_rate(self) -> float:
        """Deduplication rate as a fraction of validated records."""
        validated = self.validated_records
        if validated == 0:
            return 0.0
        return self.duplicates_removed / validated

    def summary(self) -> str:
        """Human-readable summary of pipeline results."""
        lines = [
            "PIX-32 Pipeline Result",
            "=" * 40,
            f"  Input files:        {len(self.input_files)}",
            f"  Total records:      {self.total_records}",
            f"  Validated:          {self.validated_records}",
            f"  Rejected:           {self.rejected_records}",
            f"  Duplicates removed: {self.duplicates_removed}",
            f"  Final records:      {self.final_records}",
            f"  Dedup rate:         {self.dedup_rate:.2%}",
            f"  Processing time:    {self.processing_time_seconds:.2f}s",
        ]
        if self.rejection_reasons:
            lines.append("  Rejection reasons:")
            for reason, count in sorted(self.rejection_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
        if self.errors:
            lines.append("  Errors:")
            for error in self.errors:
                lines.append(f"    {error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Content hasher for BloomFilter dedup (no external deps)
# ---------------------------------------------------------------------------


class SimpleContentHasher:
    """SHA-256 content hasher — fallback when bitarray/mmh3 unavailable."""

    @staticmethod
    def hash_conversation(conv: Conversation) -> str:
        """Generate stable hash for a conversation's content."""
        content_parts = []
        for msg in conv.messages:
            content_parts.append(f"{msg.role}:{msg.content}")
        full_content = "|".join(content_parts).lower()
        return hashlib.sha256(full_content.encode()).hexdigest()

    @staticmethod
    def hash_record(record: dict[str, Any]) -> str:
        """Generate stable hash for a raw JSONL record."""
        canonical = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Simple set-based dedup (no BloomFilter dependency)
# ---------------------------------------------------------------------------


class SetDeduplicator:
    """Hash-set based deduplication — always available, no external deps."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self.duplicates_found = 0
        self.duplicate_evidence: list[DuplicateEvidence] = []

    def is_duplicate(self, content_hash: str, record_id: str) -> bool:
        """Return True if this hash has been seen before."""
        retained_id = self._seen.get(content_hash)
        if retained_id is not None:
            self.duplicates_found += 1
            self.duplicate_evidence.append(
                DuplicateEvidence(
                    strategy=DedupStrategy.BLOOM.value,
                    content_hash=content_hash,
                    retained_id=retained_id,
                    duplicate_id=record_id,
                    reason="exact_content_hash_match",
                )
            )
            return True
        self._seen[content_hash] = record_id
        return False

    def clear(self) -> None:
        self._seen.clear()
        self.duplicates_found = 0
        self.duplicate_evidence.clear()


# ---------------------------------------------------------------------------
# Similarity-based dedup (pure Python, no sentence-transformers required)
# ---------------------------------------------------------------------------


class SimilarityDeduplicator:
    """
    Multi-metric similarity deduplication — pure Python implementation
    based on ai/pipelines/orchestrator/processing/deduplication.py patterns.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold
        self.duplicates_found = 0
        self.duplicate_evidence: list[DuplicateEvidence] = []

    def deduplicate(self, conversations: list[Conversation]) -> list[Conversation]:
        """Remove similarity-based duplicates. Returns unique conversations."""
        if len(conversations) <= 1:
            return list(conversations)

        hasher = SimpleContentHasher()
        unique = self._deduplicate_exact(conversations, hasher)
        return self._deduplicate_similar(unique, hasher)

    def _deduplicate_exact(
        self,
        conversations: list[Conversation],
        hasher: SimpleContentHasher,
    ) -> list[Conversation]:
        """Remove exact hash duplicates before slower similarity checks."""
        unique: list[Conversation] = []
        hash_to_id: dict[str, str] = {}
        for conv in conversations:
            content_hash = hasher.hash_conversation(conv)
            retained_id = hash_to_id.get(content_hash)
            if retained_id is None:
                hash_to_id[content_hash] = conv.conversation_id
                unique.append(conv)
                continue

            self.duplicates_found += 1
            self._record_duplicate(
                content_hash=content_hash,
                retained_id=retained_id,
                duplicate_id=conv.conversation_id,
                reason="exact_content_hash_match",
            )
        return unique

    def _deduplicate_similar(
        self,
        conversations: list[Conversation],
        hasher: SimpleContentHasher,
    ) -> list[Conversation]:
        """Remove approximate duplicates from already exact-deduped records."""
        final: list[Conversation] = []
        final_precomputed = []

        for features in (self._features(conv) for conv in conversations):
            conv = features[0]
            duplicate_found = False
            for kept_features in final_precomputed:
                kept_conv = kept_features[0]
                sim = self._similarity(features, kept_features)
                if sim >= self.similarity_threshold:
                    duplicate_found = True
                    self._record_duplicate(
                        content_hash=hasher.hash_conversation(conv),
                        retained_id=kept_conv.conversation_id,
                        duplicate_id=conv.conversation_id,
                        reason=(f"similarity_{sim:.3f}_gte_{self.similarity_threshold:.3f}"),
                    )
                    break

            if duplicate_found:
                self.duplicates_found += 1
            else:
                final.append(conv)
                final_precomputed.append(features)

        return final

    @staticmethod
    def _features(conv: Conversation) -> tuple[Conversation, set[str], int, list[str]]:
        words = set()
        for msg in conv.messages:
            words.update(msg.content.lower().split())
        return conv, words, len(conv.messages), [m.role for m in conv.messages]

    @staticmethod
    def _similarity(
        left: tuple[Conversation, set[str], int, list[str]],
        right: tuple[Conversation, set[str], int, list[str]],
    ) -> float:
        _left_conv, words_a, len_a, roles_a = left
        _right_conv, words_b, len_b, roles_b = right
        max_count = max(len_a, len_b)
        count_sim = 1.0 if max_count == 0 else 1.0 - abs(len_a - len_b) / max_count
        role_sim = SimilarityDeduplicator._role_similarity(roles_a, roles_b)
        structural_sim = 0.5 * count_sim + 0.5 * role_sim
        content_sim = SimilarityDeduplicator._content_similarity(words_a, words_b)
        return 0.7 * content_sim + 0.3 * structural_sim

    @staticmethod
    def _role_similarity(roles_a: list[str], roles_b: list[str]) -> float:
        if not roles_a and not roles_b:
            return 1.0
        matches = sum(1 for x, y in zip(roles_a, roles_b, strict=False) if x == y)
        return matches / max(len(roles_a), len(roles_b))

    @staticmethod
    def _content_similarity(words_a: set[str], words_b: set[str]) -> float:
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _record_duplicate(
        self,
        *,
        content_hash: str,
        retained_id: str,
        duplicate_id: str,
        reason: str,
    ) -> None:
        self.duplicate_evidence.append(
            DuplicateEvidence(
                strategy=DedupStrategy.SIMILARITY.value,
                content_hash=content_hash,
                retained_id=retained_id,
                duplicate_id=duplicate_id,
                reason=reason,
            )
        )


# ---------------------------------------------------------------------------
# Stage-aware hash dedup (pure Python, mirrors orchestrator version)
# ---------------------------------------------------------------------------


class StageAwareDeduplicator:
    """
    SHA-256 primary hash dedup with stage priority conflict resolution.
    Pure Python implementation matching
    ai/pipelines/orchestrator/stage_aware_deduplication.py
    """

    STAGE_PRIORITY: ClassVar[dict[str, int]] = {
        "stage4_voice_persona": 5,
        "stage3_edge_stress_test": 4,
        "stage2_therapeutic_expertise": 3,
        "stage1_foundation": 2,
        "supplementary": 1,
    }

    def __init__(self) -> None:
        self.duplicates_found = 0
        self.duplicate_evidence: list[DuplicateEvidence] = []

    def deduplicate(self, conversations: list[Conversation]) -> list[Conversation]:
        """Deduplicate using primary hash + stage priority conflict resolution."""
        if not conversations:
            return []

        hash_groups: dict[str, list[Conversation]] = defaultdict(list)
        for conv in conversations:
            h = self._primary_hash(conv)
            hash_groups[h].append(conv)

        unique: list[Conversation] = []
        for content_hash, group in hash_groups.items():
            if len(group) == 1:
                unique.append(group[0])
            else:
                winner = max(group, key=self._stage_priority)
                unique.append(winner)
                self.duplicates_found += len(group) - 1
                for duplicate in group:
                    if duplicate.conversation_id == winner.conversation_id:
                        continue
                    self.duplicate_evidence.append(
                        DuplicateEvidence(
                            strategy=DedupStrategy.STAGE_AWARE.value,
                            content_hash=content_hash,
                            retained_id=winner.conversation_id,
                            duplicate_id=duplicate.conversation_id,
                            reason="same_primary_hash_stage_priority_retained",
                        )
                    )

        return unique

    @staticmethod
    def _primary_hash(conv: Conversation) -> str:
        content_parts = []
        for msg in conv.messages:
            content_parts.append(f"{msg.role}{msg.content}")
        full = "".join(content_parts).lower()
        return hashlib.sha256(full.encode()).hexdigest()

    @classmethod
    def _stage_priority(cls, conv: Conversation) -> int:
        meta = conv.metadata or {}
        stage = meta.get("stage", "supplementary")
        return cls.STAGE_PRIORITY.get(stage, 1)


# ---------------------------------------------------------------------------
# NormalizationPipeline
# ---------------------------------------------------------------------------


class NormalizationPipeline:
    """
    Full PIX-32 normalization + deduplication pipeline.

    Usage:
        pipeline = NormalizationPipeline(dedup_strategy=DedupStrategy.SIMILARITY)
        result = pipeline.run(
            input_paths=["data/raw/*.jsonl"],
            output_path="data/normalized/output.jsonl",
        )
        logger.info(result.summary())
    """

    import logging

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        dedup_strategy: DedupStrategy = DedupStrategy.SIMILARITY,
        similarity_threshold: float = 0.85,
        enforce_license: bool = False,
        enforce_phi_scan: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Args:
            dedup_strategy: Which deduplication strategy to use.
            similarity_threshold: Threshold for similarity-based dedup.
            enforce_license: Reject records without license field.
            enforce_phi_scan: Reject records without phi_scan_passed.
            on_progress: Optional callback(current, total) for progress reporting.
        """
        self.normalizer = DataNormalizer(
            enforce_license=enforce_license,
            enforce_phi_scan=enforce_phi_scan,
        )
        self.dedup_strategy = dedup_strategy
        self.similarity_threshold = similarity_threshold
        self.on_progress = on_progress
        self._last_duplicate_evidence: list[DuplicateEvidence] = []

    def run(
        self,
        input_paths: list[str | Path],
        output_path: str | Path | None = None,
        reject_path: str | Path | None = None,
    ) -> PipelineResult:
        """
        Execute the full normalization + dedup pipeline.

        Args:
            input_paths: List of JSONL file paths or directories to process.
            output_path: Output path for normalized JSONL.
            reject_path: Output path for rejected records JSONL.

        Returns:
            PipelineResult with full statistics.
        """
        start_time = time.monotonic()
        result = PipelineResult()

        # Resolve input files
        jsonl_files = self._resolve_inputs(input_paths)
        result.input_files = [str(f) for f in jsonl_files]

        if not jsonl_files:
            result.errors.append("No JSONL files found in input paths")
            return result

        # Set defaults for output paths
        if output_path is None:
            output_path = Path("output_normalized.jsonl")
        if reject_path is None:
            reject_path = Path("output_rejected.jsonl")

        output_path = Path(output_path)
        reject_path = Path(reject_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reject_path.parent.mkdir(parents=True, exist_ok=True)

        # Phase 1: Validate + Normalize
        all_conversations: list[Conversation] = []
        total = 0
        processed = 0

        for file_path in jsonl_files:
            logger.info("Processing %s", file_path.name)
            file_result = self.normalizer.process_jsonl_file(
                input_path=file_path,
                output_path=file_path.with_suffix(".normalized.tmp.jsonl"),
                reject_path=file_path.with_suffix(".rejected.tmp.jsonl"),
            )
            total += file_result.total_records
            processed += file_result.valid_records

            # Read back normalized conversations
            for norm_file in [file_path.with_suffix(".normalized.tmp.jsonl")]:
                if norm_file.exists():
                    with norm_file.open("r", encoding="utf-8") as f:
                        for raw_line in f:
                            line = raw_line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    conv = Conversation.from_dict(data)
                                    all_conversations.append(conv)
                                except (json.JSONDecodeError, TypeError) as exc:
                                    result.errors.append(f"Failed to parse normalized record: {exc}")
                    norm_file.unlink()

            # Merge rejection reasons
            for reason, count in file_result.rejected_reasons.items():
                result.rejection_reasons[reason] = result.rejection_reasons.get(reason, 0) + count

            if self.on_progress:
                self.on_progress(processed, total)

        result.total_records = total
        result.validated_records = len(all_conversations)
        result.rejected_records = total - len(all_conversations)

        # Phase 2: Deduplicate
        deduped = self._deduplicate(all_conversations)
        result.duplicates_removed = len(all_conversations) - len(deduped)
        result.final_records = len(deduped)
        result.duplicate_evidence = list(self._last_duplicate_evidence)

        # Phase 3: Write output
        with output_path.open("w", encoding="utf-8") as out:
            for conv in deduped:
                out.write(json.dumps(conv.to_dict(), ensure_ascii=False) + "\n")

        # Write rejection log
        with reject_path.open("w", encoding="utf-8") as rej:
            rej.write(
                json.dumps(
                    {
                        "pipeline": "PIX-32",
                        "total_records": result.total_records,
                        "validated": result.validated_records,
                        "rejected": result.rejected_records,
                        "duplicates_removed": result.duplicates_removed,
                        "final_records": result.final_records,
                        "rejection_reasons": result.rejection_reasons,
                        "duplicate_evidence": [evidence.to_dict() for evidence in result.duplicate_evidence],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        result.processing_time_seconds = time.monotonic() - start_time
        logger.info("Pipeline complete: %s", result.summary())
        return result

    def _resolve_inputs(self, input_paths: list[str | Path]) -> list[Path]:
        """Resolve input paths to a list of JSONL files."""
        files: list[Path] = []
        for path in input_paths:
            p = Path(path)
            if p.is_file() and p.suffix == ".jsonl":
                files.append(p)
            elif p.is_dir():
                files.extend(sorted(p.rglob("*.jsonl")))
            elif "*" in str(p) or "?" in str(p):
                for match in sorted(glob_mod.glob(str(p))):
                    mp = Path(match)
                    if mp.is_file():
                        files.append(mp)
            else:
                logger.warning("Input path not found or not JSONL: %s", p)
        return files

    def _deduplicate(self, conversations: list[Conversation]) -> list[Conversation]:
        """Apply the configured deduplication strategy."""
        self._last_duplicate_evidence = []

        if self.dedup_strategy == DedupStrategy.NONE:
            return conversations

        if self.dedup_strategy == DedupStrategy.BLOOM:
            return self._dedup_bloom(conversations)

        if self.dedup_strategy == DedupStrategy.SIMILARITY:
            return self._dedup_similarity(conversations)

        if self.dedup_strategy == DedupStrategy.STAGE_AWARE:
            return self._dedup_stage_aware(conversations)

        return conversations

    def _dedup_bloom(self, conversations: list[Conversation]) -> list[Conversation]:
        """BloomFilter-style dedup using hash set (no external deps)."""
        dedup = SetDeduplicator()
        hasher = SimpleContentHasher()
        unique: list[Conversation] = []

        for conv in conversations:
            h = hasher.hash_conversation(conv)
            if not dedup.is_duplicate(h, conv.conversation_id):
                unique.append(conv)

        self._last_duplicate_evidence = list(dedup.duplicate_evidence)
        return unique

    def _dedup_similarity(self, conversations: list[Conversation]) -> list[Conversation]:
        """Similarity-based dedup using multi-metric comparison."""
        dedup = SimilarityDeduplicator(similarity_threshold=self.similarity_threshold)
        result = dedup.deduplicate(conversations)
        self._last_duplicate_evidence = list(dedup.duplicate_evidence)
        return result

    def _dedup_stage_aware(self, conversations: list[Conversation]) -> list[Conversation]:
        """Stage-aware hash dedup with priority conflict resolution."""
        dedup = StageAwareDeduplicator()
        result = dedup.deduplicate(conversations)
        self._last_duplicate_evidence = list(dedup.duplicate_evidence)
        return result


__all__ = [
    "DedupStrategy",
    "DuplicateEvidence",
    "NormalizationPipeline",
    "PipelineResult",
    "SetDeduplicator",
    "SimilarityDeduplicator",
    "SimpleContentHasher",
    "StageAwareDeduplicator",
]
