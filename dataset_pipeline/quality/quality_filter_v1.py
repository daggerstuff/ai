#!/usr/bin/env python3
"""
Quality Filter v1 - Standalone Quality Filter Component

Standalone quality filter using Quality Scoring v1 for filtering conversations
based on quality scores and decisions.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from .quality_scoring_v1 import QualityScoringV1
except ImportError:
    from ai.dataset_pipeline.quality.quality_scoring_v1 import QualityScoringV1

logger = logging.getLogger(__name__)


class QualityFilterV1:
    """
    Quality filter using Quality Scoring v1.

    Filters conversations based on quality scoring decisions and thresholds.
    """

    def __init__(
        self,
        min_decision: str = "curate",
        min_composite: float | None = None,
        config_path: str | None = None,
        enabled: bool = True,
    ):
        """
        Initialize quality filter.

        Args:
            min_decision: Minimum decision level ("accept", "curate", or "reject")
            min_composite: Optional minimum composite score
            config_path: Optional path to quality scoring config
            enabled: Whether filtering is enabled
        """
        self.quality_scoring = QualityScoringV1(
            config_path=config_path, enabled=enabled
        )
        self.min_decision = min_decision
        self.min_composite = min_composite
        self.enabled = enabled

        logger.info(
            f"Quality Filter v1 initialized: "
            f"min_decision={min_decision}, min_composite={min_composite}, enabled={enabled}"
        )

    def filter(self, conversation: Any) -> bool:
        """
        Filter a single conversation.

        Args:
            conversation: Conversation object to filter

        Returns:
            True if conversation passes filter, False otherwise
        """
        if not self.enabled:
            return True

        return self.quality_scoring.filter_conversation(
            conversation,
            min_decision=self.min_decision,
            min_composite=self.min_composite,
        )

    def filter_batch(
        self, conversations: list[Any]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """
        Filter a batch of conversations.

        Args:
            conversations: List of conversation objects

        Returns:
            Tuple of (filtered_conversations, scoring_results)
        """
        if not self.enabled:
            return conversations, []

        return self.quality_scoring.filter_conversations(
            conversations,
            min_decision=self.min_decision,
            min_composite=self.min_composite,
        )
