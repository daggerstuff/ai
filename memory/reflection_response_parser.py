from __future__ import annotations

import json
import logging

from .reflection_types import ReflectionResult

logger = logging.getLogger(__name__)


class ReflectionResponseParser:
    """Parse LLM JSON responses into reflection domain objects."""

    @staticmethod
    def parse_analysis(analysis_text: str, *, crisis_detected: bool) -> ReflectionResult:
        try:
            data = json.loads(analysis_text)
            requires_review = crisis_detected or data.get("requires_review", False)
            return ReflectionResult(
                crisis_detected=crisis_detected,
                crisis_indicators=data.get("crisis_indicators", []),
                memories_preserved=data.get("preserve_individual", []),
                memories_consolidated=data.get("can_consolidate", []),
                memories_deleted=data.get("delete", []),
                recommendations=data.get("recommendations", []),
                requires_manual_review=requires_review,
            )
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse reflection analysis: %s", exc)
            return ReflectionResult(
                crisis_detected=crisis_detected,
                requires_manual_review=crisis_detected,
            )
