"""Tests for REM-Style Dream Scheduler (Sprint 3, Task 3)."""

from __future__ import annotations

import time
from typing import Any

from ai.research.consolidation.rem_dream import CrossLink, DreamResult, RemDreamScheduler, Schema
from ai.research.schema import (
    ConsentGate,
    ConsolidationPhase,
    MemoryBlock,
    MemoryConsolidation,
    MemoryEmotions,
    MemoryGating,
    MemoryImportance,
    PIIStatus,
)

CROSSLINK_TEST_THRESHOLD = 0.2
DEFAULT_SCHEDULER_THRESHOLD = 0.6
PROCESS_SESSION_THRESHOLD = 0.3
EXPECTED_TOTAL_MEMORIES = 3
EXPECTED_SCHEMA_SOURCE_COUNT = 3


def _make_memory(
    memory_id: str,
    content: str,
    **kwargs: Any,
) -> MemoryBlock:
    raw_importance: float = kwargs.get("raw_importance", 0.5)
    emotional_weight: float = kwargs.get("emotional_weight", 1.0)
    categories: list[str] = kwargs.get("categories") or ["trust"]
    valence: float = kwargs.get("valence", 0.0)
    phase: ConsolidationPhase = kwargs.get("phase", ConsolidationPhase.RAW)
    crisis_flag: bool = kwargs.get("crisis_flag", False)
    session_id: str = kwargs.get("session_id", "session_1")
    return MemoryBlock(
        id=memory_id,
        tenantId="tenant_default",
        sessionId=session_id,
        userId="user_default",
        content=content,
        timestamp=int(time.time() * 1000) - 10000,
        importance=MemoryImportance(
            raw=raw_importance,
            recency=0.8,
            relevance=0.8,
            emotionalWeight=emotional_weight,
            actionability=0.5,
        ),
        emotions=MemoryEmotions(
            valence=valence,
            arousal=0.5,
            categories=categories or ["trust"],
        ),
        gating=MemoryGating(
            piiStatus=PIIStatus.ABSENT,
            crisisFlag=crisis_flag,
            traumaIndicators=[],
            consentGate=ConsentGate.OPEN,
        ),
        consolidation=MemoryConsolidation(
            phase=phase,
            lastProcessed=0,
            remCycles=3,
            schemaReferences=[],
        ),
    )


def test_rem_dream_scheduler_initialization() -> None:
    scheduler = RemDreamScheduler(crosslink_threshold=DEFAULT_SCHEDULER_THRESHOLD)
    assert scheduler._crosslink_threshold == DEFAULT_SCHEDULER_THRESHOLD
    assert callable(scheduler._summarizer)


def test_rem_dream_process_session() -> None:
    scheduler = RemDreamScheduler(crosslink_threshold=PROCESS_SESSION_THRESHOLD)
    memories = [
        _make_memory(
            "m1",
            "I felt anxiety when speaking in public during the team meeting",
            raw_importance=0.9,
            categories=["fear", "sadness"],
            valence=-0.6,
        ),
        _make_memory(
            "m2",
            "I experienced intense anxiety speaking in front of my team today",
            raw_importance=0.8,
            categories=["fear"],
            valence=-0.5,
        ),
        _make_memory(
            "m3",
            "The breathing exercise helped reduce my public speaking anxiety",
            raw_importance=0.6,
            categories=["trust", "fear"],
            valence=0.4,
        ),
    ]

    result = scheduler.process_session(memories)

    assert isinstance(result, DreamResult)
    assert result.memories_processed == EXPECTED_TOTAL_MEMORIES
    assert len(result.replayed) >= 1
    assert result.elapsed_ms >= 0.0
    assert "session_1" in result.summaries
    assert "anxiety" in result.summaries["session_1"].lower() or "fear" in result.summaries["session_1"].lower()


def test_rem_dream_replay() -> None:
    scheduler = RemDreamScheduler()
    memories = [
        _make_memory(f"m{i}", f"Memory content {i}", raw_importance=i * 0.2)
        for i in range(1, 6)
    ]
    replayed = scheduler._replay(memories)
    assert len(replayed) == max(1, len(memories) // 3)


def test_rem_dream_crosslink() -> None:
    scheduler = RemDreamScheduler(crosslink_threshold=CROSSLINK_TEST_THRESHOLD)
    memories = [
        _make_memory(
            "m1",
            "clinical cognitive behavioral therapy session on anxiety management",
            categories=["fear"],
        ),
        _make_memory(
            "m2",
            "cognitive behavioral therapy for anxiety management and distress",
            categories=["fear"],
        ),
        _make_memory(
            "m3",
            "completely unrelated cooking recipe with tomatoes and basil pasta",
            categories=["joy"],
        ),
    ]
    links = scheduler._crosslink(memories)
    assert len(links) >= 1
    first_link = links[0]
    assert isinstance(first_link, CrossLink)
    assert first_link.link_type == "emotional_co_occurrence"
    assert first_link.similarity >= CROSSLINK_TEST_THRESHOLD


def test_rem_dream_schema_extraction() -> None:
    scheduler = RemDreamScheduler()
    memories = [
        _make_memory("m1", "Feeling hopeful about progress", categories=["joy"], valence=0.7),
        _make_memory("m2", "Celebrating a major breakthrough", categories=["joy"], valence=0.8),
        _make_memory("m3", "Quiet moment of content satisfaction", categories=["joy"], valence=0.6),
    ]
    schemas = scheduler._extract_schemas(memories)
    assert len(schemas) >= 1
    joy_schema = schemas[0]
    assert isinstance(joy_schema, Schema)
    assert "joy" in joy_schema.title
    assert "positive" in joy_schema.generalization
    assert len(joy_schema.source_memory_ids) == EXPECTED_SCHEMA_SOURCE_COUNT


def test_rem_dream_reverie_seeding() -> None:
    scheduler = RemDreamScheduler()
    eligible = _make_memory(
        "m_eligible",
        "Deep emotional reflection on past loss",
        emotional_weight=3.5,
        phase=ConsolidationPhase.ARCHIVED,
        crisis_flag=False,
    )
    crisis_flagged = _make_memory(
        "m_crisis",
        "Crisis event",
        emotional_weight=4.0,
        phase=ConsolidationPhase.ARCHIVED,
        crisis_flag=True,
    )
    low_emotion = _make_memory(
        "m_low",
        "Routine grocery shopping list",
        emotional_weight=1.2,
        phase=ConsolidationPhase.ARCHIVED,
        crisis_flag=False,
    )
    raw_phase = _make_memory(
        "m_raw",
        "Unconsolidated new memory",
        emotional_weight=3.0,
        phase=ConsolidationPhase.RAW,
        crisis_flag=False,
    )

    seeds = scheduler._reverie_seeding([eligible, crisis_flagged, low_emotion, raw_phase])
    seed_ids = [s.memory_id for s in seeds]
    assert "m_eligible" in seed_ids
    assert "m_crisis" not in seed_ids
    assert "m_low" not in seed_ids
    assert "m_raw" not in seed_ids


def test_rem_dream_custom_summarizer() -> None:
    def custom_summary(memories: list[MemoryBlock]) -> str:
        return f"Custom: {len(memories)} entries"

    scheduler = RemDreamScheduler(summarizer=custom_summary)
    memories = [_make_memory("m1", "Test note", session_id="s_custom")]
    result = scheduler.process_session(memories)
    assert result.summaries["s_custom"] == "Custom: 1 entries"
