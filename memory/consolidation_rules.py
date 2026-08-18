"""
Memory Consolidation Rules - Rules for memory compaction and cleanup.

This module defines rules for when and how to consolidate memories,
with special handling for crisis content.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .reflection_types import CrisisSeverity, Memory, MemoryCategory

logger = logging.getLogger(__name__)


class ConsolidationRule(StrEnum):
    """Types of consolidation rules."""

    PRESERVE = "preserve"  # Never consolidate
    CONSOLIDATE = "consolidate"  # Can consolidate
    DELETE = "delete"  # Can delete if redundant
    REVIEW = "review"  # Requires manual review


@dataclass
class RuleResult:
    """Result of rule evaluation."""

    rule: ConsolidationRule
    reason: str
    priority: int = 5  # 1 = highest


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation."""

    max_general_memories: int = 100  # Max general memories before consolidation
    max_age_days: int = 90  # Max age before consolidation consideration
    crisis_retention_years: int = 7  # Crisis memories retained for 7 years
    auto_consolidate_crisis: bool = False  # Never auto-consolidate crisis


class ConsolidationRules:
    """
    Rules for memory consolidation.

    These rules ensure crisis content is preserved while allowing
    consolidation of general conversation memories.
    """

    def __init__(self, config: ConsolidationConfig | None = None):
        """
        Initialize consolidation rules.

        Args:
            config: Consolidation configuration
        """
        self.config = config or ConsolidationConfig()

        # Crisis categories - NEVER consolidate
        self.crisis_categories: set[MemoryCategory] = {
            MemoryCategory.CRISIS_CONTEXT,
            MemoryCategory.EMOTIONAL_STATE,
            MemoryCategory.THERAPEUTIC_INSIGHT,
        }

        # General categories - CAN consolidate
        self.general_categories: set[MemoryCategory] = {
            MemoryCategory.GENERAL,
            MemoryCategory.SESSION_SUMMARY,
            MemoryCategory.PREFERENCE,
        }

    def evaluate_memory(self, memory: Memory) -> RuleResult:
        """
        Evaluate a memory for consolidation.

        Args:
            memory: Memory to evaluate

        Returns:
            RuleResult with consolidation rule
        """
        # Rule 1: Crisis severity = PRESERVE
        if hasattr(memory.metadata, "crisis_severity"):
            if memory.metadata.crisis_severity != CrisisSeverity.NONE:
                return RuleResult(
                    rule=ConsolidationRule.PRESERVE,
                    reason=f"Crisis severity: {memory.metadata.crisis_severity.value}",
                    priority=1,
                )

        # Rule 2: Crisis category = PRESERVE
        if memory.metadata.category in self.crisis_categories:
            return RuleResult(
                rule=ConsolidationRule.PRESERVE,
                reason=f"Crisis category: {memory.metadata.category.value}",
                priority=1,
            )

        # Rule 3: Therapeutic insight = PRESERVE
        if memory.metadata.category == MemoryCategory.THERAPEUTIC_INSIGHT:
            return RuleResult(
                rule=ConsolidationRule.PRESERVE,
                reason="Therapeutic insight",
                priority=2,
            )

        # Rule 4: Treatment progress = PRESERVE
        if memory.metadata.category == MemoryCategory.TREATMENT_PROGRESS:
            return RuleResult(
                rule=ConsolidationRule.PRESERVE,
                reason="Treatment progress marker",
                priority=2,
            )

        # Rule 5: General conversation = CONSOLIDATE
        if memory.metadata.category in self.general_categories:
            return RuleResult(
                rule=ConsolidationRule.CONSOLIDATE,
                reason="General conversation",
                priority=5,
            )

        # Default: REVIEW
        return RuleResult(
            rule=ConsolidationRule.REVIEW,
            reason="Unknown category - requires review",
            priority=10,
        )

    def get_consolidation_candidates(
        self,
        memories: list[Memory],
    ) -> list[Memory]:
        """
        Get memories that can be consolidated.

        Args:
            memories: List of memories to evaluate

        Returns:
            List of memories that can be consolidated
        """
        candidates = []
        for memory in memories:
            result = self.evaluate_memory(memory)
            if result.rule == ConsolidationRule.CONSOLIDATE:
                candidates.append(memory)
        return candidates

    def get_preservation_list(
        self,
        memories: list[Memory],
    ) -> list[Memory]:
        """
        Get memories that must be preserved.

        Args:
            memories: List of memories to evaluate

        Returns:
            List of memories to preserve
        """
        preserved = []
        for memory in memories:
            result = self.evaluate_memory(memory)
            if result.rule == ConsolidationRule.PRESERVE:
                preserved.append(memory)
        return preserved

    def group_for_consolidation(
        self,
        memories: list[Memory],
    ) -> dict[str, list[Memory]]:
        """
        Group memories by consolidation strategy.

        Args:
            memories: List of memories to group

        Returns:
            Dict with keys: preserve, consolidate, delete, review
        """
        groups = {
            "preserve": [],
            "consolidate": [],
            "delete": [],
            "review": [],
        }

        for memory in memories:
            result = self.evaluate_memory(memory)
            if result.rule == ConsolidationRule.PRESERVE:
                groups["preserve"].append(memory)
            elif result.rule == ConsolidationRule.CONSOLIDATE:
                groups["consolidate"].append(memory)
            elif result.rule == ConsolidationRule.DELETE:
                groups["delete"].append(memory)
            else:  # REVIEW
                groups["review"].append(memory)

        return groups

    def should_trigger_consolidation(
        self,
        memories: Sequence[Memory],
    ) -> bool:
        """
        Check if consolidation should be triggered.

        Args:
            memories: List of current memories

        Returns:
            True if consolidation should run
        """
        # Count general memories
        general_count = sum(1 for m in memories if m.metadata.category in self.general_categories)

        # Trigger if over threshold
        return general_count > self.config.max_general_memories
