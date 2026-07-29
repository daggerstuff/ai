"""REM-Style Dream Scheduler — Sprint 3, Task 3.

Four-phase dream processing: replay high-importance memories,
cross-link semantically related ones, compressive summarization,
and schema extraction (episodic to semantic generalization).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..reverie_types import ReverieSeed
from ..schema import MemoryBlock

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossLink:
    memory_a_id: str
    memory_b_id: str
    similarity: float
    link_type: str


@dataclass(frozen=True)
class Schema:
    schema_id: str
    title: str
    generalization: str
    source_memory_ids: list[str]
    confidence: float


@dataclass
class DreamResult:
    replayed: list[MemoryBlock]
    cross_links: list[CrossLink]
    schemas: list[Schema]
    summaries: dict[str, str]
    reverie_seeds: list[ReverieSeed]
    elapsed_ms: float
    memories_processed: int


SummarizerFn = Callable[[list[MemoryBlock]], str]


class RemDreamScheduler:
    """Schedule and execute REM-style dream consolidation."""

    def __init__(
        self,
        summarizer: SummarizerFn | None = None,
        crosslink_threshold: float = 0.7,
    ) -> None:
        self._summarizer = summarizer or self._default_summarizer
        self._crosslink_threshold = crosslink_threshold

    def process_session(self, memories: list[MemoryBlock]) -> DreamResult:
        """Run full four-phase dream processing on a session's memories."""
        t0 = time.perf_counter()

        sorted_memories = sorted(memories, key=lambda m: m.importance.raw, reverse=True)
        replayed = self._replay(sorted_memories)
        cross_links = self._crosslink(sorted_memories)
        schemas = self._extract_schemas(sorted_memories)
        summaries = self._summarize(sorted_memories)
        reverie_seeds = self._reverie_seeding(sorted_memories)

        elapsed = (time.perf_counter() - t0) * 1000

        result = DreamResult(
            replayed=replayed,
            cross_links=cross_links,
            schemas=schemas,
            summaries=summaries,
            reverie_seeds=reverie_seeds,
            elapsed_ms=round(elapsed, 2),
            memories_processed=len(memories),
        )
        log.info(
            "Dream: %d memories, %d links, %d schemas in %.0f ms",
            result.memories_processed,
            len(result.cross_links),
            len(result.schemas),
            result.elapsed_ms,
        )
        return result

    def _replay(self, memories: list[MemoryBlock]) -> list[MemoryBlock]:
        """Phase 1: Replay high-importance memories first."""
        top_n = max(1, len(memories) // 3)
        replayed = memories[:top_n]
        for m in replayed:
            updated = m.model_copy(deep=True)
            updated.consolidation.remCycles = max(updated.consolidation.remCycles - 1, 0)
            updated.consolidation.lastProcessed = int(time.time() * 1000)
        return replayed

    def _crosslink(self, memories: list[MemoryBlock]) -> list[CrossLink]:
        """Phase 2: Create cross-links between semantically related memories."""
        links: list[CrossLink] = []
        from .dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator(threshold=self._crosslink_threshold)
        dedup._build_index(memories)
        vectors = [dedup._tfidf_vector(m.content) for m in memories]

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                sim = dedup._cosine(vectors[i], vectors[j])
                if sim >= self._crosslink_threshold:
                    same_emotion = bool(set(memories[i].emotions.categories) & set(memories[j].emotions.categories))
                    link_type = "emotional_co_occurrence" if same_emotion else "semantic_similarity"
                    links.append(
                        CrossLink(
                            memory_a_id=memories[i].id,
                            memory_b_id=memories[j].id,
                            similarity=round(sim, 4),
                            link_type=link_type,
                        )
                    )
        return links

    def _extract_schemas(self, memories: list[MemoryBlock]) -> list[Schema]:
        """Phase 3: Extract generalizations from episodic memories."""

        category_groups: dict[str, list[MemoryBlock]] = {}
        for m in memories:
            for cat in m.emotions.categories or ["general"]:
                category_groups.setdefault(cat, []).append(m)

        schemas: list[Schema] = []
        for idx, (category, group) in enumerate(category_groups.items()):
            if len(group) < 2:
                continue
            avg_valence = sum(m.emotions.valence for m in group) / len(group)
            valence_label = "positive" if avg_valence > 0.2 else "negative" if avg_valence < -0.2 else "neutral"
            schemas.append(
                Schema(
                    schema_id=f"schema_{idx}",
                    title=f"Pattern: {category}",
                    generalization=(
                        f"Multiple memories show {valence_label} "
                        f"{category}-related content across {len(group)} instances"
                    ),
                    source_memory_ids=[m.id for m in group],
                    confidence=round(min(len(group) / 10.0, 1.0), 2),
                )
            )
        return schemas

    def _summarize(self, memories: list[MemoryBlock]) -> dict[str, str]:
        """Phase 4: Compressive summarization per session."""
        session_groups: dict[str, list[MemoryBlock]] = {}
        for m in memories:
            session_groups.setdefault(m.sessionId, []).append(m)

        summaries: dict[str, str] = {}
        for session_id, group in session_groups.items():
            summaries[session_id] = self._summarizer(group)
        return summaries

    def _reverie_seeding(self, memories: list[MemoryBlock]) -> list[ReverieSeed]:
        """Phase 5: Identify archived/forgotten memories as reverie candidates.

        Memories with emotionalWeight >= 2.0 that are not crisis-flagged
        become reverie seeds — latent memories eligible for fishhook detection.
        """
        seeds: list[ReverieSeed] = []
        now_ms = int(time.time() * 1000)

        for m in memories:
            if m.gating.crisisFlag:
                continue
            if m.importance.emotionalWeight < 2.0:
                continue
            if m.consolidation.phase not in ("archived", "forgotten", "latent"):
                continue

            emotional = min(m.importance.emotionalWeight / 5.0, 1.0)
            category_diversity = min(len(m.emotions.categories) / 5.0, 1.0)
            schema_richness = min(len(m.consolidation.schemaReferences) / 3.0, 1.0)
            age_days = max((now_ms - m.timestamp) / 86400000.0, 0.0)
            recency = max(1.0 - age_days / 30.0, 0.0)
            potential = 0.4 * emotional + 0.2 * category_diversity + 0.2 * schema_richness + 0.2 * recency

            seeds.append(
                ReverieSeed(
                    memory_id=m.id,
                    reason=f"Emotional weight {m.importance.emotionalWeight}, phase {m.consolidation.phase}",
                    potential=round(potential, 4),
                )
            )

        return seeds

    @staticmethod
    def _default_summarizer(memories: list[MemoryBlock]) -> str:
        if not memories:
            return ""
        top = sorted(memories, key=lambda m: m.importance.raw, reverse=True)[:3]
        topics = set()
        for m in top:
            topics.update(m.emotions.categories or ["general"])
        return (
            f"Session with {len(memories)} memories. "
            f"Key themes: {', '.join(sorted(topics))}. "
            f"Highest importance: {top[0].content[:100]}"
        )
