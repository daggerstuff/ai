from __future__ import annotations

"""
Reflection subagent for crisis-aware memory consolidation.

This subagent orchestrates reflection flow and delegates LLM-facing analysis
to `reflection_analysis`, keeping the shared service shell narrow.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace

from .reflection_analysis import ReflectionAnalysisService
from .reflection_memory import LocalReflectionMemoryClient, Memory
from .reflection_types import ReflectionConfig, ReflectionResult, ReflectionTrigger

logger = logging.getLogger(__name__)


class ReflectionSubagent:
    """Coordinate reflection analysis and consolidation against local memory."""

    def __init__(
        self,
        memory_provider: LocalReflectionMemoryClient,
        config: ReflectionConfig | None = None,
        llm_callback: Callable | None = None,
    ) -> None:
        self.memory = memory_provider
        self.config = config or ReflectionConfig()
        self.analysis = ReflectionAnalysisService(llm_callback)
        self.llm_callback = llm_callback
        self._message_count = 0

    async def analyze_conversation(
        self,
        conversation_text: str,
        user_id: str,
        existing_memories: list[Memory] | None = None,
    ) -> ReflectionResult:
        logger.info("Starting reflection analysis for user %s", user_id)

        memories_text = self._format_memories(existing_memories or [])
        crisis_result, analysis = await asyncio.gather(
            self.analysis.detect_crisis(conversation_text),
            self.analysis.run_reflection(
                conversation_text=conversation_text,
                existing_memories=memories_text,
                include_crisis_context=self.config.include_crisis_context,
                crisis_detected=None,
            ),
        )
        result = self.analysis.parse_analysis(
            analysis,
            crisis_detected=crisis_result.get("crisis_detected", False),
        )

        if crisis_result.get("crisis_detected"):
            result.requires_manual_review = True
            result.crisis_indicators = crisis_result.get("indicators", [])

        logger.info(
            "Reflection complete: %s preserved, %s consolidated",
            len(result.memories_preserved),
            len(result.memories_consolidated),
        )
        return result

    async def consolidate_memories(
        self,
        user_id: str,
        result: ReflectionResult,
    ) -> dict[str, int]:
        if result.crisis_detected and not self.config.auto_consolidate:
            logger.info("Crisis detected - skipping auto-consolidation, preserving all memories")
            result = replace(result, memories_deleted=[])
        if hasattr(self.memory, "execute_consolidation"):
            return await self.memory.execute_consolidation(
                result,
                user_id=user_id,
                allow_crisis_deletions=self.config.auto_consolidate,
            )

        memories_deleted = result.memories_deleted
        if result.crisis_detected and not self.config.auto_consolidate:
            memories_deleted = []
            memories_consolidated = []
        else:
            memories_consolidated = result.memories_consolidated

        return {
            "preserved": len(result.memories_preserved),
            "consolidated": len(memories_consolidated),
            "deleted": len(memories_deleted),
            "errors": 0,
        }

    def _format_memories(self, memories: list[Memory]) -> str:
        if not memories:
            return "No existing memories."

        lines = ["## Existing Memories:"]
        for memory in memories[: self.config.max_memories_to_review]:
            lines.append(memory.to_prompt_line())
        return "\n".join(lines)

    def increment_message_count(self) -> None:
        self._message_count += 1

    def should_reflect(self) -> bool:
        if self.config.trigger == ReflectionTrigger.MANUAL:
            return False
        if self.config.trigger == ReflectionTrigger.STEP_COUNT:
            return self._message_count >= self.config.step_threshold
        return False

    def reset_message_count(self) -> None:
        self._message_count = 0

    async def close(self) -> None:
        await self.memory.close()
