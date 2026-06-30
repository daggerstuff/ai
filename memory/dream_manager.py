#!/usr/bin/env python3
"""
Dream Manager - Coordinates dream cycles and memory consolidation.

The Dream Manager orchestrates the dream cycle process:
1. Collects memories from wakeful period
2. Enters dream cycle (NREM → REM phases)
3. Consolidates memories during REM phase
4. Triggers post-dream reflection
5. Stores consolidated memories with insights

This implements the sleep-inspired memory consolidation model where:
- NREM phase: Reactivation of recent memories
- REM phase: Integration and pattern extraction
- Post-dream: Reflection and insight generation
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .dream_memory_store import (
    DreamCycleRecord,
    DreamMemoryStore,
    LocalDreamMemoryStore,
)
from .dream_reflection_integration import (
    DreamOutput,
    DreamReflectionConfig,
    DreamReflectionIntegration,
)

logger = logging.getLogger(__name__)


@dataclass
class DreamCycleResult:
    """Result of a complete dream cycle."""

    dream_id: str
    user_id: str
    start_time: str
    end_time: str

    # Phases completed
    nrem_completed: bool = False
    rem_completed: bool = False
    consolidation_completed: bool = False
    reflection_triggered: bool = False

    # Outputs
    themes: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    consolidated_memories: list[dict[str, Any]] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    emotional_tone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "user_id": self.user_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "phases": {
                "nrem_completed": self.nrem_completed,
                "rem_completed": self.rem_completed,
                "consolidation_completed": self.consolidation_completed,
                "reflection_triggered": self.reflection_triggered,
            },
            "themes": self.themes,
            "patterns": self.patterns,
            "consolidated_memories": self.consolidated_memories,
            "insights": self.insights,
            "emotional_tone": self.emotional_tone,
        }


@dataclass
class DreamManagerConfig:
    """Configuration for Dream Manager."""

    # Timing (in seconds for testing, normally minutes)
    nrem_duration: int = 60  # NREM phase duration
    rem_duration: int = 90  # REM phase duration
    post_dream_delay: int = 5  # Delay before reflection

    # Content thresholds
    min_memories_for_dream: int = 5
    max_dream_themes: int = 5
    max_dream_patterns: int = 3

    # Integration
    enable_reflection_integration: bool = True

    # Storage
    store_dream_lineage: bool = True


class DreamManager:
    """
    Manages dream cycles for memory consolidation.

    Coordinates the full dream cycle process from memory reactivation
    through pattern extraction to reflection triggering.
    """

    def __init__(
        self,
        memory_store: DreamMemoryStore | None = None,
        config: DreamManagerConfig | None = None,
    ):
        """
        Initialize Dream Manager.

        Args:
            memory_store: Async store for dream cycles, consolidated memories,
                          and user memories. Defaults to LocalDreamMemoryStore.
            config: Configuration for dream cycles
        """
        self.memory_store = memory_store or LocalDreamMemoryStore()
        self.config = config or DreamManagerConfig()

        # Reflection integration
        reflection_config = DreamReflectionConfig(
            post_dream_delay_minutes=1,  # Short for testing
            enable_post_dream_reflection=self.config.enable_reflection_integration,
        )

        self.reflection_integration = DreamReflectionIntegration(
            memory_store=self.memory_store,
            config=reflection_config,
        )

        # Active dream cycles
        self._active_dreams: dict[str, DreamCycleResult] = {}

        logger.info("DreamManager initialized")

    async def start_dream_cycle(
        self,
        user_id: str,
        memories: list[dict[str, Any]] | None = None,
    ) -> DreamCycleResult:
        """
        Start a complete dream cycle.

        Args:
            user_id: User identifier
            memories: Memories to process (or fetch from memory manager)

        Returns:
            Dream cycle result
        """
        dream_id = self._generate_dream_id(user_id)
        start_time = datetime.now(UTC).isoformat()

        logger.info(f"Starting dream cycle {dream_id} for user {user_id}")

        result = DreamCycleResult(
            dream_id=dream_id,
            user_id=user_id,
            start_time=start_time,
            end_time="",
        )

        self._active_dreams[dream_id] = result

        try:
            # Fetch memories if not provided
            if not memories:
                memories = await self._fetch_recent_memories(user_id)

            # Check minimum memories
            if len(memories) < self.config.min_memories_for_dream:
                logger.info(f"Only {len(memories)} memories, skipping dream cycle")
                result.end_time = datetime.now(UTC).isoformat()
                return result

            # NREM Phase: Memory reactivation
            logger.debug("NREM phase: Reactivating memories")
            nrem_result = await self._nrem_phase(memories)
            result.nrem_completed = True

            # REM Phase: Pattern extraction and integration
            logger.debug("REM phase: Extracting patterns")
            rem_result = await self._rem_phase(memories, nrem_result)
            result.rem_completed = True
            result.themes = rem_result.get("themes", [])
            result.patterns = rem_result.get("patterns", [])
            result.emotional_tone = rem_result.get("emotional_tone")

            # Consolidation phase
            logger.debug("Consolidation phase")
            consolidated = await self._consolidate_memories(user_id, memories, rem_result)
            result.consolidation_completed = True
            result.consolidated_memories = consolidated

            # Trigger post-dream reflection
            if self.config.enable_reflection_integration:
                from .dream_reflection_integration import DreamPhase

                dream_output = DreamOutput(
                    dream_id=dream_id,
                    user_id=user_id,
                    phase=DreamPhase.REM,
                    themes=result.themes,
                    patterns=result.patterns,
                    consolidated_memories=consolidated,
                    emotional_tone=result.emotional_tone,
                )

                await self.reflection_integration.trigger_post_dream_reflection(
                    user_id=user_id,
                    dream_output=dream_output,
                )
                result.reflection_triggered = True

            result.end_time = datetime.now(UTC).isoformat()
            result.insights = []  # Will be populated by reflection

            dream_record = DreamCycleRecord(
                dream_id=dream_id,
                user_id=user_id,
                start_time=start_time,
                end_time=result.end_time,
                themes=result.themes,
                patterns=result.patterns,
                emotional_tone=result.emotional_tone,
                insight_count=0,
                consolidated_memory_ids=[
                    m.get("_id", m.get("id", "")) for m in consolidated if m.get("_id") or m.get("id")
                ],
                nrem_completed=result.nrem_completed,
                rem_completed=result.rem_completed,
                consolidation_completed=result.consolidation_completed,
                reflection_triggered=result.reflection_triggered,
            )
            await self.memory_store.save_dream_cycle(dream_record)

            logger.info(
                f"Dream cycle {dream_id} completed: {len(result.themes)} themes, {len(result.patterns)} patterns"
            )

        except Exception as e:
            logger.error(f"Dream cycle {dream_id} failed: {e}")
            result.end_time = datetime.now(UTC).isoformat()
            raise

        return result

    async def _nrem_phase(
        self,
        memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        NREM sleep phase: Memory reactivation.

        During NREM, memories are reactivated and strengthened.
        This is a simplified simulation - in production, this would
        involve actual memory reactivation patterns.
        """
        # Simulate memory reactivation
        reactivated = []
        for memory in memories:
            # Strengthen emotional memories
            if "emotional" in str(memory.get("category", "")).lower():
                reactivated.append(memory)

        return {
            "reactivated_memories": reactivated,
            "reactivation_count": len(reactivated),
        }

    async def _rem_phase(
        self,
        memories: list[dict[str, Any]],
        nrem_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        REM sleep phase: Pattern extraction and integration.

        During REM, the brain extracts themes, patterns, and creates
        novel associations between memories.
        """
        # Extract themes
        themes = self._extract_themes(memories)

        # Extract patterns
        patterns = self._extract_patterns(memories)

        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(memories)

        return {
            "themes": themes[: self.config.max_dream_themes],
            "patterns": patterns[: self.config.max_dream_patterns],
            "emotional_tone": emotional_tone,
        }

    def _extract_themes(
        self,
        memories: list[dict[str, Any]],
    ) -> list[str]:
        """Extract themes from memories."""
        # Simplified theme extraction
        # In production, this would use LLM-based theme extraction
        themes = []

        for memory in memories:
            memory.get("content", "")
            category = memory.get("category", "general")

            # Extract theme from category
            if category not in themes and len(themes) < self.config.max_dream_themes:
                themes.append(category)

        return themes or ["general_processing"]

    def _extract_patterns(
        self,
        memories: list[dict[str, Any]],
    ) -> list[str]:
        """Extract patterns from memories."""
        # Simplified pattern extraction
        # In production, this would use LLM-based pattern recognition
        patterns = []

        # Look for recurring elements
        categories = {}
        for memory in memories:
            cat = memory.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

        # Patterns are recurring categories
        for cat, count in categories.items():
            if count > 1:
                patterns.append(f"recurring_{cat}")

        return patterns or ["memory_integration"]

    def _determine_emotional_tone(
        self,
        memories: list[dict[str, Any]],
    ) -> str | None:
        """Determine overall emotional tone of memories."""
        # Simplified emotional tone detection
        # In production, this would use sentiment analysis
        emotional_categories = {
            "emotional_state": "emotional",
            "crisis_context": "distressing",
            "therapeutic_insight": "insightful",
        }

        for memory in memories:
            category = memory.get("category", "")
            if category in emotional_categories:
                return emotional_categories[category]

        return None

    async def _consolidate_memories(
        self,
        user_id: str,
        memories: list[dict[str, Any]],
        rem_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        consolidated = []

        themes = rem_result.get("themes", [])
        patterns = rem_result.get("patterns", [])

        for memory in memories:
            category = memory.get("category", "")

            should_consolidate = category in themes or any(p.split("_")[1] in category for p in patterns if "_" in p)

            if should_consolidate:
                content = memory.get("content", "")
                mem_id = await self.memory_store.add_memory(
                    content=content,
                    user_id=user_id,
                    metadata=memory.get("metadata"),
                    category=category,
                )
                consolidated.append(
                    {
                        **memory,
                        "_id": mem_id,
                        "consolidated": True,
                        "consolidation_time": datetime.now(UTC).isoformat(),
                        "dream_consolidated": self.config.store_dream_lineage,
                    }
                )

        return consolidated

    async def _fetch_recent_memories(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fetch recent memories for dream processing."""
        return await self.memory_store.get_all_memories(user_id=user_id, limit=limit)

    def _generate_dream_id(self, user_id: str) -> str:
        """Generate unique dream ID."""
        import time

        return hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()[:16]

    async def get_dream_status(self, dream_id: str) -> dict[str, Any] | None:
        """Get status of a dream cycle."""
        dream = self._active_dreams.get(dream_id)
        if not dream:
            return None
        return dream.to_dict()

    async def close(self) -> None:
        """Cancel pending reflections and release the memory store."""
        await self.reflection_integration.cancel_all()
        await self.memory_store.close()
