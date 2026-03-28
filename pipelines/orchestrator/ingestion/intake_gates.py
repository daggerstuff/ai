"""
Orchestrator-owned intake gates for feeder admission.

This module provides a narrow, production-grade wrapper around the existing
context detector and taxonomy classifier so promoted feeders can be routed into
the canonical ladder before deeper ingestion work occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.pipelines.design.context_detector import ContextDetector
from ai.pipelines.design.taxonomy_classifier import (
    TaxonomyClassifier,
    TherapeuticCategory,
)
from ai.pipelines.orchestrator.configs.intake_routing import (
    CONTINUITY_HOLDOUT_LANE,
    resolve_intake_route,
)
from ai.pipelines.orchestrator.configs.stages import STAGE1_ID, STAGE3_ID


@dataclass(frozen=True)
class IntakeGateDecision:
    """Decision returned by the orchestrator intake gates."""

    source_family: str
    target_lane: str
    split: str | None
    context_is_therapeutic: bool
    context_confidence: float
    taxonomy_category: str
    taxonomy_confidence: float
    requires_human_review: bool
    reasons: list[str] = field(default_factory=list)


class OrchestratorIntakeGates:
    """Thin gate wrapper for early intake routing decisions."""

    def __init__(self, human_review_threshold: float = 0.70):
        self.context_detector = ContextDetector()
        self.taxonomy_classifier = TaxonomyClassifier()
        self.human_review_threshold = human_review_threshold

    @staticmethod
    def _conversation_text(messages: list[dict[str, str]]) -> str:
        """Flatten ChatML-style messages into classifier-ready text."""

        parts: list[str] = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _messages_from_record(record: dict[str, object]) -> list[dict[str, str]]:
        """Extract classifier-ready messages from a record payload."""
        raw_messages = record.get("messages")
        if isinstance(raw_messages, list):
            return [message for message in raw_messages if isinstance(message, dict)]

        prompt = str(record.get("prompt", "")).strip()
        response = str(record.get("response", "")).strip()
        messages: list[dict[str, str]] = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response:
            messages.append({"role": "assistant", "content": response})

        if messages:
            return messages

        text = str(record.get("text", "")).strip()
        if text:
            return [{"role": "user", "content": text}]
        return []

    def evaluate(
        self,
        *,
        source_family: str,
        record: dict[str, object] | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> IntakeGateDecision:
        """Evaluate intake routing and gating for a feeder payload."""

        route = resolve_intake_route(source_family)
        reasons = [route.reason]
        record_payload = record if isinstance(record, dict) else {}
        resolved_messages = (
            messages if isinstance(messages, list) else self._messages_from_record(record_payload)
        )

        if route.target_lane == CONTINUITY_HOLDOUT_LANE:
            reasons.append("Long-running therapy is held out for continuity evaluation.")
            return IntakeGateDecision(
                source_family=route.source_family,
                target_lane=route.target_lane,
                split=route.split_preference,
                context_is_therapeutic=True,
                context_confidence=1.0,
                taxonomy_category="continuity_holdout",
                taxonomy_confidence=1.0,
                requires_human_review=False,
                reasons=reasons,
            )

        classifier_record = {"messages": resolved_messages}
        conversation_text = self._conversation_text(resolved_messages)
        context = self.context_detector.detect_context(conversation_text)
        classification = self.taxonomy_classifier.classify_record(classifier_record)

        target_lane = route.target_lane
        split = route.split_preference

        if target_lane != STAGE1_ID and not context.is_therapeutic and context.confidence >= 0.70:
            target_lane = STAGE1_ID
            reasons.append("Educational or meta-discussion context rerouted to Stage 1.")

        if (
            route.target_lane != CONTINUITY_HOLDOUT_LANE
            and classification.category == TherapeuticCategory.CRISIS_SUPPORT
        ):
            target_lane = STAGE3_ID
            reasons.append("Crisis-support content rerouted to Stage 3.")

        requires_human_review = classification.confidence < self.human_review_threshold
        if requires_human_review:
            reasons.append("Low classification confidence requires human review.")

        return IntakeGateDecision(
            source_family=route.source_family,
            target_lane=target_lane,
            split=split,
            context_is_therapeutic=context.is_therapeutic,
            context_confidence=context.confidence,
            taxonomy_category=classification.category.value,
            taxonomy_confidence=classification.confidence,
            requires_human_review=requires_human_review,
            reasons=reasons,
        )


__all__ = ["IntakeGateDecision", "OrchestratorIntakeGates"]
