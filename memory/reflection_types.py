from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReflectionTrigger(StrEnum):
    """What triggers reflection."""

    MANUAL = "manual"
    STEP_COUNT = "step_count"
    COMPACTION = "compaction"
    CRISIS = "crisis"
    SESSION_END = "session_end"


@dataclass
class ReflectionConfig:
    """Configuration for reflection subagent."""

    trigger: ReflectionTrigger = ReflectionTrigger.STEP_COUNT
    step_threshold: int = 10
    include_crisis_context: bool = True
    auto_consolidate: bool = False
    max_memories_to_review: int = 50
    llm_model: str = "claude-sonnet-4-6"


@dataclass
class ReflectionResult:
    """Result of reflection analysis."""

    crisis_detected: bool = False
    crisis_indicators: list[str] = field(default_factory=list)
    memories_preserved: list[str] = field(default_factory=list)
    memories_consolidated: list[str] = field(default_factory=list)
    memories_deleted: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    requires_manual_review: bool = False


class MemoryCategory(StrEnum):
    GENERAL = "general"
    CRISIS_CONTEXT = "crisis_context"
    EMOTIONAL_STATE = "emotional_state"
    THERAPEUTIC_INSIGHT = "therapeutic_insight"
    TREATMENT_PROGRESS = "treatment_progress"
    SESSION_SUMMARY = "session_summary"
    PREFERENCE = "preference"


class CrisisSeverity(StrEnum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryMetadata:
    category: MemoryCategory = MemoryCategory.GENERAL
    crisis_severity: CrisisSeverity = CrisisSeverity.NONE
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: float | None = None
    updated_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "crisis_severity": self.crisis_severity.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryMetadata:
        return cls(
            category=MemoryCategory(data.get("category", MemoryCategory.GENERAL.value)),
            crisis_severity=CrisisSeverity(data.get("crisis_severity", CrisisSeverity.NONE.value)),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class Memory:
    id: str
    content: str
    metadata: MemoryMetadata
    embedding: list[float] | None = None

    def to_prompt_line(self, max_chars: int = 200) -> str:
        return f"- [{self.metadata.category.value}] {self.content[:max_chars]}"
