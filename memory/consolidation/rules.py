"""Consolidation Trigger Engine — Sprint 3, Task 5.

Defines and manages consolidation triggers: MANUAL, STEP_COUNT,
COMPACTION, CRISIS, SESSION_END. Crisis trigger takes precedence.
Rules are configurable without code changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum

from ..consolidation_rules import ConsolidationConfig, ConsolidationRules
from ..schema import MemoryBlock

log = logging.getLogger(__name__)


class TriggerType(StrEnum):
    MANUAL = "manual"
    STEP_COUNT = "step_count"
    COMPACTION = "compaction"
    CRISIS = "crisis"
    SESSION_END = "session_end"


@dataclass(frozen=True)
class TriggerEvent:
    trigger_type: TriggerType
    timestamp_ms: int
    context: dict[str, object]
    priority: int


@dataclass
class TriggerConfig:
    step_interval: int = 50
    compaction_threshold: int = 200
    crisis_takes_precedence: bool = True


class ConsolidationTriggerEngine:
    """Manage consolidation triggers with configurable rules."""

    def __init__(
        self,
        config: TriggerConfig | None = None,
        consolidation_config: ConsolidationConfig | None = None,
    ) -> None:
        self._config = config or TriggerConfig()
        self._consolidation_config = consolidation_config or ConsolidationConfig()
        self._rules = ConsolidationRules(self._consolidation_config)
        self._step_counter: int = 0
        self._pending_triggers: list[TriggerEvent] = []
        self._crisis_reflection_prompts: dict[str, str] = {
            "immediate": "Process recent crisis content for safety review.",
            "post_session": "Reflect on crisis patterns from this session.",
        }

    def record_step(self) -> TriggerEvent | None:
        """Record an interaction step. Returns trigger if threshold reached."""
        self._step_counter += 1
        if self._step_counter >= self._config.step_interval:
            self._step_counter = 0
            event = TriggerEvent(
                trigger_type=TriggerType.STEP_COUNT,
                timestamp_ms=int(time.time() * 1000),
                context={"steps_since_last": self._config.step_interval},
                priority=3,
            )
            self._pending_triggers.append(event)
            log.info("Trigger: step_count threshold reached")
            return event
        return None

    def check_compaction(self, memories: list[MemoryBlock]) -> TriggerEvent | None:
        """Check if memory count exceeds compaction threshold."""
        if len(memories) >= self._config.compaction_threshold:
            event = TriggerEvent(
                trigger_type=TriggerType.COMPACTION,
                timestamp_ms=int(time.time() * 1000),
                context={"memory_count": len(memories)},
                priority=2,
            )
            self._pending_triggers.append(event)
            log.info("Trigger: compaction threshold reached (%d memories)", len(memories))
            return event
        return None

    def detect_crisis_trigger(self, memories: list[MemoryBlock]) -> TriggerEvent | None:
        """Check for crisis content that needs post-crisis processing."""
        crisis_memories = [m for m in memories if m.gating.crisisFlag]
        if crisis_memories:
            event = TriggerEvent(
                trigger_type=TriggerType.CRISIS,
                timestamp_ms=int(time.time() * 1000),
                context={"crisis_count": len(crisis_memories)},
                priority=1,
            )
            self._pending_triggers.append(event)
            log.info("Trigger: crisis content detected (%d memories)", len(crisis_memories))
            return event
        return None

    def on_session_end(self) -> TriggerEvent:
        """Fire session end trigger."""
        event = TriggerEvent(
            trigger_type=TriggerType.SESSION_END,
            timestamp_ms=int(time.time() * 1000),
            context={},
            priority=4,
        )
        self._pending_triggers.append(event)
        return event

    def request_manual(self, context: dict[str, object] | None = None) -> TriggerEvent:
        """Request manual consolidation."""
        event = TriggerEvent(
            trigger_type=TriggerType.MANUAL,
            timestamp_ms=int(time.time() * 1000),
            context=context or {},
            priority=5,
        )
        self._pending_triggers.append(event)
        return event

    def get_next_trigger(self) -> TriggerEvent | None:
        """Get the highest-priority pending trigger. Crisis takes precedence."""
        if not self._pending_triggers:
            return None
        if self._config.crisis_takes_precedence:
            crisis_triggers = [t for t in self._pending_triggers if t.trigger_type == TriggerType.CRISIS]
            if crisis_triggers:
                return crisis_triggers[0]
        self._pending_triggers.sort(key=lambda t: t.priority)
        return self._pending_triggers.pop(0)

    def clear_triggers(self) -> None:
        self._pending_triggers.clear()

    def get_crisis_reflection_prompt(self, context: str = "immediate") -> str:
        return self._crisis_reflection_prompts.get(context, self._crisis_reflection_prompts["immediate"])

    def should_trigger(self, memories: list[MemoryBlock]) -> bool:
        """Check if consolidation should run based on current state."""
        # Count non-crisis memories as "general" (MemoryBlock has no MemoryCategory)
        general_count = sum(1 for m in memories if not m.gating.crisisFlag)
        if general_count > self._rules.config.max_general_memories:
            return True
        if any(m.gating.crisisFlag for m in memories):
            return True
        return len(memories) >= self._config.compaction_threshold

    @property
    def pending_count(self) -> int:
        return len(self._pending_triggers)

    def reset(self) -> None:
        self._step_counter = 0
        self._pending_triggers.clear()
