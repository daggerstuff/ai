#!/usr/bin/env python3
"""
Dream Cycle Integration with Reflection

Integrates reflection tasks with dream cycles so that insights from reflections
feed back into long-term memory. This module implements:

1. Timing Coordination: Schedule reflection tasks after REM-style dreaming phases
2. Data Flow: Dream Manager outputs → Reflection Tasks input
3. Feedback Loop: Memories → Dreaming → Consolidated Memories → Reflection → New Insights
4. Storage: Link reflection insights to dream cycles for lineage tracking
5. Configuration: Enable/disable reflection-after-dreaming with tunable delays

Usage:
    from ai.memory.dream_reflection_integration import DreamReflectionIntegration

    integration = DreamReflectionIntegration()

    # After dream cycle completes
    await integration.trigger_post_dream_reflection(
        user_id="user_123",
        dream_output=dream_result,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .dream_memory_store import DreamMemoryStore, LocalDreamMemoryStore
from .reflection_types import MemoryCategory, MemoryMetadata

logger = logging.getLogger(__name__)


class DreamPhase(StrEnum):
    """Dream cycle phases."""

    NREM = "nrem"  # Non-REM sleep
    REM = "rem"  # REM sleep - where dream consolidation happens
    POST_DREAM = "post_dream"  # Post-dream processing
    REFLECTION = "reflection"  # Reflection phase


@dataclass
class DreamOutput:
    """Output from a dream cycle."""

    dream_id: str
    user_id: str
    phase: DreamPhase
    themes: list[str]
    patterns: list[str]
    consolidated_memories: list[dict[str, Any]]
    emotional_tone: str | None = None
    insights: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dream_id": self.dream_id,
            "user_id": self.user_id,
            "phase": self.phase.value,
            "themes": self.themes,
            "patterns": self.patterns,
            "consolidated_memories": self.consolidated_memories,
            "emotional_tone": self.emotional_tone,
            "insights": self.insights,
            "timestamp": self.timestamp,
        }


@dataclass
class ReflectionInsight:
    """Insight generated from reflection on dream content."""

    insight_id: str
    dream_id: str
    user_id: str
    content: str
    category: MemoryCategory
    related_themes: list[str]
    related_patterns: list[str]
    confidence: float  # 0.0-1.0
    requires_consolidation: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_memory_metadata(self) -> MemoryMetadata:
        """Convert to memory metadata for storage."""
        return MemoryMetadata(
            category=self.category,
            user_id=self.user_id,
            tags=[
                f"dream:{self.dream_id}",
                *[f"theme:{t}" for t in self.related_themes[:3]],
            ],
            created_at=datetime.now(UTC).timestamp(),
        )


@dataclass
class DreamReflectionConfig:
    """Configuration for dream-reflection integration."""

    # Timing
    post_dream_delay_minutes: int = 5  # Delay before triggering reflection
    max_reflection_timeout_minutes: int = 30  # Max time to wait for reflection

    # Feature flags
    enable_post_dream_reflection: bool = True
    enable_dream_lineage_tracking: bool = True

    # Content filtering
    min_dream_confidence: float = 0.5  # Minimum confidence to trigger reflection
    max_reflection_topics: int = 5  # Max topics to reflect on

    # Storage
    store_dream_lineage: bool = True
    lineage_depth: int = 3  # How many generations of dream lineage to track


class DreamReflectionIntegration:
    """
    Integrates reflection tasks with dream cycles.

    Coordinates the flow from dream cycle completion through reflection
    and back to long-term memory storage.
    """

    def __init__(
        self,
        memory_store: DreamMemoryStore | None = None,
        config: DreamReflectionConfig | None = None,
    ):
        self.memory_store = memory_store or LocalDreamMemoryStore()
        self.config = config or DreamReflectionConfig()

        # Pending reflections (dream_id -> ReflectionTask)
        self._pending_reflections: dict[str, asyncio.Task] = {}

        logger.info("DreamReflectionIntegration initialized")

    async def trigger_post_dream_reflection(
        self,
        user_id: str,
        dream_output: DreamOutput,
    ) -> str | None:
        """
        Trigger reflection after a dream cycle completes.

        Args:
            user_id: User identifier
            dream_output: Output from the dream cycle

        Returns:
            Reflection task ID if scheduled, None if skipped
        """
        if not self.config.enable_post_dream_reflection:
            logger.debug("Post-dream reflection disabled")
            return None

        # Check confidence threshold
        if not dream_output.patterns or not dream_output.themes:
            logger.debug("No patterns/themes from dream, skipping reflection")
            return None

        # Schedule reflection with delay
        delay_seconds = self.config.post_dream_delay_minutes * 60

        logger.info(
            f"Scheduling post-dream reflection for user {user_id} dream {dream_output.dream_id} in {delay_seconds}s"
        )

        # Create reflection task
        task = asyncio.create_task(
            self._delayed_reflection(
                user_id=user_id,
                dream_output=dream_output,
                delay_seconds=delay_seconds,
            )
        )

        self._pending_reflections[dream_output.dream_id] = task

        return dream_output.dream_id

    async def _delayed_reflection(
        self,
        user_id: str,
        dream_output: DreamOutput,
        delay_seconds: int,
    ) -> None:
        """Wait then perform reflection."""
        try:
            # Wait for delay
            await asyncio.sleep(delay_seconds)

            # Check if still pending (not cancelled)
            if dream_output.dream_id not in self._pending_reflections:
                return

            # Execute reflection
            insights = await self._execute_reflection(user_id, dream_output)

            # Store insights
            await self._store_insights(user_id, dream_output, insights)

            logger.info(f"Stored {len(insights)} insights from dream {dream_output.dream_id}")

        except asyncio.CancelledError:
            logger.debug(f"Reflection for dream {dream_output.dream_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Reflection failed for dream {dream_output.dream_id}: {e}")
        finally:
            # Clean up pending task
            self._pending_reflections.pop(dream_output.dream_id, None)

    async def _execute_reflection(
        self,
        user_id: str,
        dream_output: DreamOutput,
    ) -> list[ReflectionInsight]:
        """
        Execute reflection on dream content.

        This is a simplified implementation. In production, this would
        call an LLM-based reflection agent.

        Args:
            user_id: User identifier
            dream_output: Dream output to reflect on

        Returns:
            List of reflection insights
        """
        insights = []

        # Generate insights from themes
        for i, theme in enumerate(dream_output.themes[: self.config.max_reflection_topics]):
            insight_content = self._generate_insight_from_theme(
                theme=theme,
                patterns=dream_output.patterns,
                emotional_tone=dream_output.emotional_tone,
            )

            insight = ReflectionInsight(
                insight_id=f"insight_{dream_output.dream_id}_{i}",
                dream_id=dream_output.dream_id,
                user_id=user_id,
                content=insight_content,
                category=MemoryCategory.THERAPEUTIC_INSIGHT,
                related_themes=[theme],
                related_patterns=dream_output.patterns[:3],
                confidence=0.8,  # Would be computed by reflection agent
            )
            insights.append(insight)

        # Generate insight from patterns
        if dream_output.patterns:
            pattern_insight = self._generate_insight_from_patterns(
                patterns=dream_output.patterns,
                themes=dream_output.themes,
            )

            if pattern_insight:
                insights.append(pattern_insight)

        return insights

    def _generate_insight_from_theme(
        self,
        theme: str,
        patterns: list[str],
        emotional_tone: str | None,
    ) -> str:
        """Generate insight from a single theme."""
        emotion_context = f" with {emotional_tone} tone" if emotional_tone else ""

        pattern_context = ""
        if patterns:
            pattern_str = ", ".join(patterns[:2])
            pattern_context = f" in relation to patterns: {pattern_str}"

        return (
            f"Dream theme '{theme}'{emotion_context} suggests processing of "
            f"underlying emotional content{pattern_context}. "
            f"This theme may benefit from further exploration in therapeutic context."
        )

    def _generate_insight_from_patterns(
        self,
        patterns: list[str],
        themes: list[str],
    ) -> ReflectionInsight | None:
        """Generate consolidated insight from multiple patterns."""
        if len(patterns) < 2:
            return None

        pattern_text = ", ".join(patterns[:3])
        theme_text = themes[0] if themes else "current experiences"

        content = (
            f"Pattern analysis across {pattern_text} reveals recurring "
            f"themes related to '{theme_text}'. "
            f"Consider exploring connections between these patterns "
            f"in future sessions."
        )

        return ReflectionInsight(
            insight_id=f"pattern_insight_{len(patterns)}",
            dream_id="consolidated",
            user_id="system",
            content=content,
            category=MemoryCategory.THERAPEUTIC_INSIGHT,
            related_themes=themes[:2],
            related_patterns=patterns,
            confidence=0.7,
        )

    async def _store_insights(
        self,
        user_id: str,
        dream_output: DreamOutput,
        insights: list[ReflectionInsight],
    ) -> None:
        """
        Store reflection insights as memories.

        Args:
            user_id: User identifier
            dream_output: Related dream output
            insights: List of insights to store
        """
        for insight in insights:
            try:
                # Create metadata with dream lineage
                metadata = insight.to_memory_metadata()

                # Add dream lineage tracking
                if self.config.store_dream_lineage:
                    metadata.tags.append(f"dream_lineage:{dream_output.dream_id}")

                await self.memory_store.add_memory(
                    content=insight.content,
                    user_id=user_id,
                    metadata=metadata,
                    category=insight.category.value,
                )

                logger.debug(f"Stored insight {insight.insight_id} for user {user_id}")

            except Exception as e:
                logger.error(f"Failed to store insight {insight.insight_id}: {e}")

    async def cancel_reflection(self, dream_id: str) -> bool:
        """
        Cancel a pending reflection task.

        Args:
            dream_id: Dream ID to cancel reflection for

        Returns:
            True if cancelled, False if not found
        """
        task = self._pending_reflections.get(dream_id)
        if not task:
            return False

        task.cancel()
        self._pending_reflections.pop(dream_id, None)
        return True

    async def get_reflection_status(self, dream_id: str) -> dict[str, Any]:
        """
        Get status of a reflection task.

        Args:
            dream_id: Dream ID to check

        Returns:
            Status dictionary with state and insights
        """
        task = self._pending_reflections.get(dream_id)

        if not task:
            return {"status": "not_found"}

        if task.done():
            if task.cancelled():
                return {"status": "cancelled"}
            if task.exception():
                return {"status": "failed", "error": str(task.exception())}
            return {"status": "completed"}
        return {"status": "pending"}

    async def cancel_all(self) -> None:
        """Cancel all pending reflection tasks without closing the store."""
        for task in list(self._pending_reflections.values()):
            task.cancel()

        if self._pending_reflections:
            await asyncio.gather(
                *self._pending_reflections.values(),
                return_exceptions=True,
            )

        self._pending_reflections.clear()

    async def close(self) -> None:
        """Cancel pending reflections and release the memory store."""
        await self.cancel_all()
        if hasattr(self.memory_store, "close"):
            await self.memory_store.close()


def create_dream_output(
    user_id: str,
    themes: list[str],
    patterns: list[str],
    emotional_tone: str | None = None,
) -> DreamOutput:
    """Helper to create dream output for testing."""
    import hashlib
    import time

    dream_id = hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()[:16]

    return DreamOutput(
        dream_id=dream_id,
        user_id=user_id,
        phase=DreamPhase.REM,
        themes=themes,
        patterns=patterns,
        consolidated_memories=[],
        emotional_tone=emotional_tone,
    )
