#!/usr/bin/env python3
"""
Quality Scoring v1 Integration for Dataset Pipeline

Integrates the quality scoring system (KAN-12) into the dataset pipeline.
Provides adapters for conversation objects and quality gate functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Import quality scoring system
try:
    import sys
    from pathlib import Path as PathLib

    # Add scripts directory to path if not already there
    scripts_path = PathLib(__file__).parent.parent.parent.parent / "scripts"
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    from scripts.quality_scoring.pipeline_integration import (
        score_conversation_text,
    )
    from scripts.quality_scoring.scoring_interface import (
        compose_score,
        compute_signals,
    )

    QUALITY_SCORING_AVAILABLE = True
except ImportError as e:
    QUALITY_SCORING_AVAILABLE = False
    logging.warning(
        f"Quality scoring v1 not available: {e}. "
        f"Some quality scoring features will be disabled."
    )

logger = logging.getLogger(__name__)


class QualityScoringV1:
    """
    Quality Scoring v1 adapter for dataset pipeline.

    Integrates the KAN-12 quality scoring system into the pipeline,
    providing conversation-level scoring and filtering.
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        weights: dict[str, float] | None = None,
        thresholds: dict[str, float] | None = None,
        enabled: bool = True,
    ):
        """
        Initialize quality scoring adapter.

        Args:
            config_path: Path to quality scoring config file
            weights: Custom weights for signals (overrides config)
            thresholds: Custom thresholds (overrides config)
            enabled: Whether quality scoring is enabled
        """
        self.enabled = enabled and QUALITY_SCORING_AVAILABLE

        if not QUALITY_SCORING_AVAILABLE:
            logger.warning("Quality scoring v1 components not available")
            return

        # Load default config
        self.weights = {
            "empathy": 0.25,
            "fidelity": 0.25,
            "domain": 0.25,
            "harm": 0.25,
        }
        self.thresholds = {
            "harm_max": 0.05,
            "accept_min": 0.60,
            "curate_min": 0.45,
        }

        # Load from config file if provided
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        if weights is None:
                            self.weights = config.get("weights", self.weights)
                        if thresholds is None:
                            self.thresholds = config.get("thresholds", self.thresholds)
                except Exception as e:
                    logger.warning(f"Failed to load quality scoring config: {e}")

        # Override with provided values
        if weights:
            self.weights = weights
        if thresholds:
            self.thresholds = thresholds

        logger.info(
            f"Quality Scoring v1 initialized: enabled={self.enabled}, "
            f"weights={self.weights}, thresholds={self.thresholds}"
        )

    def score_conversation_text(self, text: str) -> dict[str, Any]:
        """
        Score a conversation text.

        Args:
            text: Conversation text to score

        Returns:
            Dictionary with signals, composite score, and decision
        """
        if not self.enabled:
            return {
                "signals": {
                    "empathy": 0.5,
                    "fidelity": 0.5,
                    "domain": 0.5,
                    "harm": 0.0,
                },
                "composite": 0.5,
                "decision": "curate",
                "enabled": False,
            }

        try:
            return score_conversation_text(text, self.weights, self.thresholds)
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            return {
                "signals": {
                    "empathy": 0.5,
                    "fidelity": 0.5,
                    "domain": 0.5,
                    "harm": 0.0,
                },
                "composite": 0.5,
                "decision": "curate",
                "error": str(e),
            }

    def extract_text_from_conversation(self, conversation: Any) -> str:
        """
        Extract text from a conversation object.

        Handles various conversation formats:
        - Dict with 'messages' or 'turns'
        - Conversation dataclass
        - Plain text string

        Args:
            conversation: Conversation object

        Returns:
            Extracted text as string
        """
        if isinstance(conversation, str):
            return conversation

        if isinstance(conversation, dict):
            # Try messages first
            if "messages" in conversation:
                messages = conversation["messages"]
                if isinstance(messages, list):
                    return " ".join(
                        msg.get("content", "") if isinstance(msg, dict) else str(msg)
                        for msg in messages
                    )

            # Try turns
            if "turns" in conversation:
                turns = conversation["turns"]
                if isinstance(turns, list):
                    return " ".join(
                        turn.get("content", "") if isinstance(turn, dict) else str(turn)
                        for turn in turns
                    )

            # Try direct text field
            if "text" in conversation:
                return str(conversation["text"])

            # Try conversation field
            if "conversation" in conversation:
                return str(conversation["conversation"])

            # Last resort: stringify the whole dict
            return json.dumps(conversation)

        # Try to get text attribute
        if hasattr(conversation, "text"):
            return str(conversation.text)

        if hasattr(conversation, "messages"):
            messages = conversation.messages
            if isinstance(messages, list):
                return " ".join(
                    msg.content if hasattr(msg, "content") else str(msg)
                    for msg in messages
                )

        if hasattr(conversation, "turns"):
            turns = conversation.turns
            if isinstance(turns, list):
                return " ".join(
                    turn.content if hasattr(turn, "content") else str(turn)
                    for turn in turns
                )

        # Last resort
        return str(conversation)

    def score_conversation(self, conversation: Any) -> dict[str, Any]:
        """
        Score a conversation object.

        Args:
            conversation: Conversation object (various formats supported)

        Returns:
            Dictionary with quality scores and decision
        """
        if not self.enabled:
            return {
                "signals": {
                    "empathy": 0.5,
                    "fidelity": 0.5,
                    "domain": 0.5,
                    "harm": 0.0,
                },
                "composite": 0.5,
                "decision": "curate",
                "enabled": False,
            }

        text = self.extract_text_from_conversation(conversation)
        return self.score_conversation_text(text)

    def filter_conversation(
        self,
        conversation: Any,
        min_decision: str = "curate",
        min_composite: float | None = None,
    ) -> bool:
        """
        Filter a conversation based on quality scoring.

        Args:
            conversation: Conversation object
            min_decision: Minimum decision level ("accept", "curate", or "reject")
            min_composite: Optional minimum composite score

        Returns:
            True if conversation passes filter, False otherwise
        """
        if not self.enabled:
            return True  # Pass through if disabled

        result = self.score_conversation(conversation)

        decision = result.get("decision", "reject")
        composite = result.get("composite", 0.0)

        # Decision hierarchy: reject < curate < accept
        decision_levels = {"reject": 0, "curate": 1, "accept": 2}
        min_level = decision_levels.get(min_decision, 1)
        actual_level = decision_levels.get(decision, 0)

        # Check decision level
        if actual_level < min_level:
            return False

        # Check composite score if specified
        if min_composite is not None and composite < min_composite:
            return False

        return True

    def filter_conversations(
        self,
        conversations: list[Any],
        min_decision: str = "curate",
        min_composite: float | None = None,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """
        Filter a list of conversations based on quality scoring.

        Args:
            conversations: List of conversation objects
            min_decision: Minimum decision level
            min_composite: Optional minimum composite score

        Returns:
            Tuple of (filtered_conversations, scoring_results)
        """
        filtered = []
        results = []

        for conversation in conversations:
            result = self.score_conversation(conversation)
            results.append(result)

            if self.filter_conversation(
                conversation, min_decision=min_decision, min_composite=min_composite
            ):
                filtered.append(conversation)

        return filtered, results
