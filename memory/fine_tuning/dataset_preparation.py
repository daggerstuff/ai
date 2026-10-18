"""Fine-Tuning Dataset Preparation — Sprint 5, Task 1.

Extracts memory-augmented training pairs from consolidated memories,
creates train/val/test splits, balances for emotional valence,
and validates PII safety.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

from ai.memory.schema import MemoryBlock

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingExample:
    query: str
    context: str
    response: str
    metadata: dict[str, object]


@dataclass
class DatasetSplit:
    train: list[TrainingExample]
    val: list[TrainingExample]
    test: list[TrainingExample]


@dataclass
class DatasetStats:
    total_examples: int
    train_count: int
    val_count: int
    test_count: int
    avg_valence: float
    crisis_ratio: float
    pii_leak_detected: bool


PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
]


class DatasetPreparator:
    """Prepare fine-tuning datasets from consolidated memories."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._seed = seed
        random.seed(seed)

    def prepare(
        self,
        memories: list[MemoryBlock],
        output_dir: Path | None = None,
    ) -> tuple[DatasetSplit, DatasetStats]:
        """Full dataset preparation pipeline."""
        t0 = time.perf_counter()

        examples = self._extract_examples(memories)
        log.info("Extracted %d training examples from %d memories", len(examples), len(memories))

        balanced = self._balance_valence(examples)
        log.info("Balanced to %d examples", len(balanced))

        pii_leak = self._check_pii(balanced)
        if pii_leak:
            log.warning("PII leak detected in training data!")

        split = self._split(balanced)
        log.info("Split: train=%d, val=%d, test=%d", len(split.train), len(split.val), len(split.test))

        stats = self._compute_stats(split, pii_leak)

        if output_dir:
            self._save(split, output_dir)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info("Dataset preparation complete in %.0f ms", elapsed)
        return split, stats

    def _extract_examples(self, memories: list[MemoryBlock]) -> list[TrainingExample]:
        """Convert memories into (query, context, response) training pairs."""
        examples: list[TrainingExample] = []
        session_groups: dict[str, list[MemoryBlock]] = {}
        for m in sorted(memories, key=lambda x: x.timestamp):
            session_groups.setdefault(m.sessionId, []).append(m)

        for session_id, session_memories in session_groups.items():
            for i, memory in enumerate(session_memories):
                prior_context = "\n".join(m.content for m in session_memories[:i])
                query = memory.content
                response = self._generate_response_template(memory)

                examples.append(
                    TrainingExample(
                        query=query,
                        context=prior_context,
                        response=response,
                        metadata={
                            "session_id": session_id,
                            "memory_id": memory.id,
                            "valence": memory.emotions.valence,
                            "arousal": memory.emotions.arousal,
                            "crisis_flag": memory.gating.crisisFlag,
                            "consolidation_phase": memory.consolidation.phase.value,
                        },
                    )
                )
        return examples

    def _balance_valence(self, examples: list[TrainingExample]) -> list[TrainingExample]:
        """Balance examples across emotional valence buckets."""
        buckets: dict[str, list[TrainingExample]] = {
            "negative": [],
            "neutral": [],
            "positive": [],
        }
        for ex in examples:
            v = ex.metadata.get("valence", 0)
            if v < -0.2:
                buckets["negative"].append(ex)
            elif v > 0.2:
                buckets["positive"].append(ex)
            else:
                buckets["neutral"].append(ex)

        max_size = max(len(b) for b in buckets.values()) if buckets else 0
        target = max(max_size, 1)

        balanced: list[TrainingExample] = []
        for _bucket_name, bucket in buckets.items():
            if len(bucket) >= target:
                balanced.extend(bucket[:target])
            else:
                balanced.extend(bucket)
                repeats = bucket * (math.ceil(target / len(bucket)) if bucket else 0)
                balanced.extend(repeats[: target - len(bucket)])

        random.shuffle(balanced)
        return balanced

    def _check_pii(self, examples: list[TrainingExample]) -> bool:
        """Check for PII leakage in training data."""
        import re

        for ex in examples:
            for pattern in PII_PATTERNS:
                if re.search(pattern, ex.query) or re.search(pattern, ex.response):
                    return True
        return False

    def _split(self, examples: list[TrainingExample]) -> DatasetSplit:
        """Create train/val/test splits."""
        shuffled = list(examples)
        random.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * self._train_ratio)
        val_end = train_end + int(n * self._val_ratio)
        return DatasetSplit(
            train=shuffled[:train_end],
            val=shuffled[train_end:val_end],
            test=shuffled[val_end:],
        )

    def _compute_stats(self, split: DatasetSplit, pii_leak: bool) -> DatasetStats:
        """Compute dataset statistics."""
        all_examples = split.train + split.val + split.test
        valences = [ex.metadata.get("valence", 0) for ex in all_examples]
        crisis_count = sum(1 for ex in all_examples if ex.metadata.get("crisis_flag"))
        return DatasetStats(
            total_examples=len(all_examples),
            train_count=len(split.train),
            val_count=len(split.val),
            test_count=len(split.test),
            avg_valence=round(sum(valences) / len(valences), 3) if valences else 0,
            crisis_ratio=round(crisis_count / len(all_examples), 3) if all_examples else 0,
            pii_leak_detected=pii_leak,
        )

    def _save(self, split: DatasetSplit, output_dir: Path) -> None:
        """Save dataset splits as JSONL."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, examples in [
            ("train", split.train),
            ("val", split.val),
            ("test", split.test),
        ]:
            path = output_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(
                        json.dumps(
                            {
                                "query": ex.query,
                                "context": ex.context,
                                "response": ex.response,
                                "metadata": ex.metadata,
                            }
                        )
                        + "\n"
                    )
        log.info("Dataset saved to %s", output_dir)

    @staticmethod
    def _generate_response_template(memory: MemoryBlock) -> str:
        """Generate a therapeutic response template for a memory."""
        emotion_labels = ", ".join(memory.emotions.categories or ["general"])
        valence_desc = (
            "negative emotional state"
            if memory.emotions.valence < -0.2
            else "positive emotional state"
            if memory.emotions.valence > 0.2
            else "neutral emotional state"
        )
        return (
            f"Client is in a {valence_desc} ({emotion_labels}). "
            f"Content: {memory.content}. "
            f"Importance: {memory.importance.raw:.2f}. "
            f"Continue therapeutic engagement with appropriate emotional attunement."
        )
