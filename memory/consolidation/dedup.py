"""Semantic Deduplication Engine — Sprint 3, Task 2.

Clusters memories by semantic similarity (threshold 0.92 cosine),
merges near-duplicates with provenance tracking, and preserves
emotional intensity from all merged sources.

Uses TF-IDF + cosine similarity (no external ML dependency).
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from ..schema import MemoryBlock

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.92


@dataclass(frozen=True)
class DedupCluster:
    """A cluster of near-duplicate memories."""

    cluster_id: str
    members: list[MemoryBlock]
    representative: MemoryBlock
    similarity_scores: list[float]
    provenance: list[str]


@dataclass
class DedupResult:
    """Result of deduplication pass."""

    clusters: list[DedupCluster]
    unique_memories: list[MemoryBlock]
    merged_memories: list[MemoryBlock]
    total_before: int
    total_after: int
    reduction_pct: float
    elapsed_ms: float


class SemanticDeduplicator:
    """Deduplicate memories using TF-IDF cosine similarity."""

    def __init__(self, threshold: float = DEDUP_THRESHOLD) -> None:
        self._threshold = threshold
        self._idf: dict[str, float] = {}
        self._vocabulary: list[str] = []

    def deduplicate(self, memories: list[MemoryBlock]) -> DedupResult:
        """Run deduplication on a list of memories.

        Returns clusters of duplicates and the deduplicated set.
        """
        t0 = time.perf_counter()
        if len(memories) < 2:
            return DedupResult(
                clusters=[],
                unique_memories=list(memories),
                merged_memories=[],
                total_before=len(memories),
                total_after=len(memories),
                reduction_pct=0.0,
                elapsed_ms=0.0,
            )

        self._build_index(memories)
        vectors = [self._tfidf_vector(m.content) for m in memories]

        used: set[int] = set()
        clusters: list[DedupCluster] = []
        unique: list[MemoryBlock] = []

        for i in range(len(memories)):
            if i in used:
                continue
            cluster_members = [memories[i]]
            cluster_scores = [1.0]
            used.add(i)

            for j in range(i + 1, len(memories)):
                if j in used:
                    continue
                sim = self._cosine(vectors[i], vectors[j])
                if sim >= self._threshold:
                    cluster_members.append(memories[j])
                    cluster_scores.append(sim)
                    used.add(j)

            if len(cluster_members) > 1:
                rep = max(cluster_members, key=lambda m: m.importance.raw)
                cluster = DedupCluster(
                    cluster_id=f"cluster_{len(clusters)}",
                    members=cluster_members,
                    representative=rep,
                    similarity_scores=cluster_scores,
                    provenance=[m.id for m in cluster_members],
                )
                clusters.append(cluster)
                unique.append(self._merge_cluster(cluster))
            else:
                unique.append(memories[i])

        elapsed = (time.perf_counter() - t0) * 1000
        merged = [m for c in clusters for m in c.members[1:]]
        reduction = (len(memories) - len(unique)) / len(memories) * 100 if memories else 0.0

        result = DedupResult(
            clusters=clusters,
            unique_memories=unique,
            merged_memories=merged,
            total_before=len(memories),
            total_after=len(unique),
            reduction_pct=round(reduction, 2),
            elapsed_ms=round(elapsed, 2),
        )
        log.info(
            "Dedup: %d -> %d memories (%.1f%% reduction) in %.0f ms",
            result.total_before,
            result.total_after,
            result.reduction_pct,
            result.elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # TF-IDF internals
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def _build_index(self, memories: list[MemoryBlock]) -> None:
        doc_freq: dict[str, int] = defaultdict(int)
        all_terms: set[str] = set()
        for m in memories:
            terms = set(self._tokenize(m.content))
            all_terms.update(terms)
            for t in terms:
                doc_freq[t] += 1

        n = len(memories)
        self._vocabulary = sorted(all_terms)
        self._idf = {term: math.log((n + 1) / (df + 1)) + 1 for term, df in doc_freq.items()}

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        terms = self._tokenize(text)
        if not terms:
            return {}
        tf: dict[str, float] = defaultdict(float)
        for t in terms:
            tf[t] += 1
        max_tf = max(tf.values())
        return {t: (count / max_tf) * self._idf.get(t, 1.0) for t, count in tf.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = a.keys() & b.keys()
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _merge_cluster(cluster: DedupCluster) -> MemoryBlock:
        """Merge a cluster into a single representative memory.

        Preserves the highest emotional intensity from all sources.
        """
        rep = cluster.representative
        max_arousal = max(m.emotions.arousal for m in cluster.members)
        max_emotional = max(m.importance.emotionalWeight for m in cluster.members)
        all_indicators: list[str] = []
        for m in cluster.members:
            all_indicators.extend(m.gating.traumaIndicators)

        merged = rep.model_copy(deep=True)
        merged.emotions.arousal = max_arousal
        merged.importance.emotionalWeight = max_emotional
        merged.gating.traumaIndicators = list(set(all_indicators))
        merged.consolidation.remCycles = max(m.consolidation.remCycles for m in cluster.members)
        return merged
