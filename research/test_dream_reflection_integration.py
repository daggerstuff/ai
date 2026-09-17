"""Tests for Dream Reflection Integration (PIX-4382)."""

from __future__ import annotations

from typing import Any

import pytest

from ai.research.dream_reflection_integration import (
    DreamOutput,
    DreamPhase,
    DreamReflectionConfig,
    DreamReflectionIntegration,
    ReflectionInsight,
)
from ai.research.reflection_types import MemoryCategory

TEST_POST_DREAM_DELAY_MINUTES = 2
EXPECTED_THEME_INSIGHTS_COUNT = 2


class InMemoryDreamMemoryStore:
    """In-memory store conforming to DreamMemoryStore Protocol."""

    def __init__(self) -> None:
        self.memories: list[dict[str, Any]] = []
        self.dream_cycles: list[Any] = []

    async def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Any = None,
        category: str | None = None,
    ) -> str:
        memory_id = f"mem_{len(self.memories) + 1}"
        self.memories.append(
            {
                "id": memory_id,
                "content": content,
                "user_id": user_id,
                "metadata": metadata,
                "category": category,
            }
        )
        return memory_id

    async def get_all_memories(
        self, user_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        return [m for m in self.memories if m["user_id"] == user_id][:limit]

    async def save_dream_cycle(self, record: Any) -> None:
        self.dream_cycles.append(record)

    async def close(self) -> None:
        pass


@pytest.fixture
def memory_store():
    return InMemoryDreamMemoryStore()


@pytest.fixture
def sample_dream_output():
    return DreamOutput(
        dream_id="dream_test_123",
        user_id="user_test_456",
        phase=DreamPhase.REM,
        themes=["vulnerability", "social connection"],
        patterns=["fear of opening up", "longing for acceptance"],
        consolidated_memories=[
            {"id": "m1", "content": "Shared childhood story with a friend"},
            {"id": "m2", "content": "Avoided eye contact during confrontation"},
        ],
        emotional_tone="cautiously optimistic",
    )


def test_integration_init(memory_store):
    config = DreamReflectionConfig(post_dream_delay_minutes=TEST_POST_DREAM_DELAY_MINUTES)
    integration = DreamReflectionIntegration(memory_store=memory_store, config=config)
    assert integration.config.post_dream_delay_minutes == TEST_POST_DREAM_DELAY_MINUTES
    assert integration.memory_store == memory_store


@pytest.mark.asyncio
async def test_trigger_post_dream_reflection_disabled(memory_store, sample_dream_output):
    config = DreamReflectionConfig(enable_post_dream_reflection=False)
    integration = DreamReflectionIntegration(memory_store=memory_store, config=config)

    task_id = await integration.trigger_post_dream_reflection(
        user_id="user_test_456",
        dream_output=sample_dream_output,
    )
    assert task_id is None


@pytest.mark.asyncio
async def test_trigger_post_dream_reflection_no_patterns(memory_store):
    integration = DreamReflectionIntegration(memory_store=memory_store)
    empty_dream = DreamOutput(
        dream_id="dream_empty",
        user_id="user_test_456",
        phase=DreamPhase.REM,
        themes=[],
        patterns=[],
        consolidated_memories=[],
    )

    task_id = await integration.trigger_post_dream_reflection(
        user_id="user_test_456",
        dream_output=empty_dream,
    )
    assert task_id is None


@pytest.mark.asyncio
async def test_execute_reflection(memory_store, sample_dream_output):
    integration = DreamReflectionIntegration(memory_store=memory_store)
    insights = await integration._execute_reflection(
        user_id="user_test_456",
        dream_output=sample_dream_output,
    )

    assert len(insights) >= EXPECTED_THEME_INSIGHTS_COUNT
    theme_insights = [i for i in insights if i.dream_id == "dream_test_123"]
    assert len(theme_insights) == EXPECTED_THEME_INSIGHTS_COUNT
    assert any("vulnerability" in i.content for i in theme_insights)

    pattern_insights = [i for i in insights if i.dream_id == "consolidated"]
    assert len(pattern_insights) == 1
    assert "fear of opening up" in pattern_insights[0].content


@pytest.mark.asyncio
async def test_store_insights(memory_store, sample_dream_output):
    integration = DreamReflectionIntegration(memory_store=memory_store)
    insights = [
        ReflectionInsight(
            insight_id="insight_test_1",
            dream_id=sample_dream_output.dream_id,
            user_id="user_test_456",
            content="Key insight on interpersonal trust",
            category=MemoryCategory.THERAPEUTIC_INSIGHT,
            related_themes=["vulnerability"],
            related_patterns=["fear of opening up"],
            confidence=0.85,
        )
    ]

    await integration._store_insights("user_test_456", sample_dream_output, insights)

    assert len(memory_store.memories) == 1
    stored = memory_store.memories[0]
    assert stored["content"] == "Key insight on interpersonal trust"
    assert any("dream_lineage:" in tag for tag in stored["metadata"].tags)
