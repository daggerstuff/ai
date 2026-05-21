# -*- coding: utf-8 -*-
"""REM-Style Dream Scheduler — Sprint 3, Task 3.

Four-phase dream processing: replay high-importance memories,
cross-link semantically related ones, compressive summarization,
and schema extraction (episodic to semantic generalization).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
    source_memory_ids: List[str]
    confidence: float


@dataclass
class DreamResult:
    replayed: List[MemoryBlock]
    cross_links: List[CrossLink]
    schemas: List[Schema]
    summaries: Dict[str, str]
    elapsed_ms: float
    memories_processed: int


SummarizerFn = Callable[[List[MemoryBlock]], str]


class RemDreamScheduler:
    """Schedule and execute REM-style dream consolidation."""

    def __init__(
        self,
        summarizer: Optional[SummarizerFn] = None,
        crosslink_threshold: float = 0.7,
    ) -> None:
        self._summarizer = summarizer or self._default_summarizer
        self._crosslink_threshold = crosslink_threshold

    def process_session(self, memories: List[MemoryBlock]) -> DreamResult:
        """Run full four-phase dream processing on a session's memories."""
        t0 = time.perf_counter()

        sorted_memories = sorted(
            memories, key=lambda m: m.importance.raw, reverse=True
        )
        replayed = self._replay(sorted_memories)
        cross_links = self._crosslink(sorted_memories)
        schemas = self._extract_schemas(sorted_memories)
        summaries = self._summarize(sorted_memories)

        elapsed = (time.perf_counter() - t0) * 1000

        result = DreamResult(
            replayed=replayed,
            cross_links=cross_links,
            schemas=schemas,
            summaries=summaries,
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

    def _replay(self, memories: List[MemoryBlock]) -> List[MemoryBlock]:
        """Phase 1: Replay high-importance memories first."""
        top_n = max(1, len(memories) // 3)
        replayed = memories[:top_n]
        for m in replayed:
            updated = m.model_copy(deep=True)
            updated.consolidation.remCycles = max(
                updated.consolidation.remCycles - 1, 0
            )
            updated.consolidation.lastProcessed = int(time.time() * 1000)
        return replayed

    def _crosslink(self, memories: List[MemoryBlock]) -> List[CrossLink]:
        """Phase 2: Create cross-links between semantically related memories."""
        links: List[CrossLink] = []
        from .dedup import SemanticDeduplicator

        dedup = SemanticDeduplicator(threshold=self._crosslink_threshold)
        dedup._build_index(memories)
        vectors = [dedup._tfidf_vector(m.content) for m in memories]

        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                sim = dedup._cosine(vectors[i], vectors[j])
                if sim >= self._crosslink_threshold:
                    same_emotion = bool(
                        set(memories[i].emotions.categories)
                        & set(memories[j].emotions.categories)
                    )
                    link_type = (
                        "emotional_co_occurrence"
                        if same_emotion
                        else "semantic_similarity"
                    )
                    links.append(
                        CrossLink(
                            memory_a_id=memories[i].id,
                            memory_b_id=memories[j].id,
                            similarity=round(sim, 4),
                            link_type=link_type,
                        )
                    )
        return links

    def _extract_schemas(self, memories: List[MemoryBlock]) -> List[Schema]:
        """Phase 3: Extract generalizations from episodic memories."""
        from collections import Counter

        category_groups: Dict[str, List[MemoryBlock]] = {}
        for m in memories:
            for cat in m.emotions.categories or ["general"]:
                category_groups.setdefault(cat, []).append(m)

        schemas: List[Schema] = []
        for idx, (category, group) in enumerate(category_groups.items()):
            if len(group) < 2:
                continue
            avg_valence = sum(m.emotions.valence for m in group) / len(group)
            valence_label = (
                "positive" if avg_valence > 0.2
                else "negative" if avg_valence < -0.2
                else "neutral"
            )
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

    def _summarize(self, memories: List[MemoryBlock]) -> Dict[str, str]:
        """Phase 4: Compressive summarization per session."""
        session_groups: Dict[str, List[MemoryBlock]] = {}
        for m in memories:
            session_groups.setdefault(m.sessionId, []).append(m)

        summaries: Dict[str, str] = {}
        for session_id, group in session_groups.items():
            summaries[session_id] = self._summarizer(group)
        return summaries

    @staticmethod
    def _default_summarizer(memories: List[MemoryBlock]) -> str:
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
