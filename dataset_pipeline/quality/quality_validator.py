"""
Quality validator for real-time conversation assessment.

This module provides comprehensive quality validation for conversations during
dataset acquisition, including coherence, authenticity, and overall quality
scoring to ensure high-quality training data.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .conversation_schema import Conversation
from .logger import get_logger

logger = get_logger("dataset_pipeline.quality_validator")


@dataclass
class QualityResult:
    """Result of quality validation for a conversation."""
    conversation_id: str
    overall_score: float
    coherence_score: float
    authenticity_score: float
    completeness_score: float
    issues: list[str]
    strengths: list[str]
    metadata: dict[str, Any]
    validated_at: datetime


class QualityValidator:
    """
    Comprehensive quality validator for conversations.

    Provides real-time assessment of conversation quality including
    coherence, authenticity, completeness, and overall scoring.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the quality validator."""
        self.config = config or {}

        # Quality thresholds
        self.min_message_length = self.config.get("min_message_length", 10)
        self.max_message_length = self.config.get("max_message_length", 2000)
        self.min_conversation_length = self.config.get("min_conversation_length", 2)
        self.max_conversation_length = self.config.get("max_conversation_length", 50)

        # Pattern matching for quality assessment
        self._initialize_quality_patterns()

        logger.info("Quality Validator initialized")

    def _initialize_quality_patterns(self) -> None:
        """Initialize patterns for quality assessment."""

        # Patterns that indicate low quality
        self.low_quality_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"\b(lorem ipsum|placeholder|test|dummy)\b",
                r"\b(asdf|qwerty|123456)\b",
                r"^(.)\1{10,}",  # Repeated characters
                r"[A-Z]{10,}",   # All caps sequences
                r"[!?]{3,}",     # Multiple punctuation
                r"\b(spam|scam|click here|buy now)\b"
            ]
        ]

        # Patterns that indicate good therapeutic content
        self.therapeutic_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"\b(feel|feeling|emotion|understand|support)\b",
                r"\b(therapy|counseling|mental health|wellbeing)\b",
                r"\b(anxiety|depression|stress|trauma|grief)\b",
                r"\b(coping|healing|recovery|growth|resilience)\b",
                r"\b(how does that make you feel|tell me more|I hear you)\b"
            ]
        ]

        # Patterns for authentic conversation flow
        self.conversation_flow_patterns = [
            re.compile(p) for p in [
                r"\b(yes|no|maybe|I think|I feel|I believe)\b",
                r"\b(thank you|thanks|appreciate|helpful)\b",
                r"\b(question|wondering|curious|confused)\b",
                r"\b(sorry|apologize|understand|clarify)\b"
            ]
        ]

        # Patterns for formal language
        self.formal_patterns = [
            re.compile(p) for p in [
                r"\b(furthermore|moreover|nevertheless|consequently)\b",
                r"\b(I am pleased to inform|I would like to state)\b",
                r"\b(as per|in accordance with|pursuant to)\b"
            ]
        ]

        # Patterns for natural language
        self.natural_patterns = [
            re.compile(p) for p in [
                r"\b(yeah|yep|nope|gonna|wanna|kinda)\b",
                r"\b(I\'m|you\'re|we\'re|they\'re|can\'t|won\'t)\b",
                r"\b(hmm|oh|ah|well|so|like)\b"
            ]
        ]

        # Patterns for greetings
        self.greeting_patterns = [
            re.compile(p) for p in [
                r"\b(hello|hi|hey|good morning|good afternoon)\b"
            ]
        ]

        # Patterns for closure
        self.closure_patterns = [
            re.compile(p) for p in [
                r"\b(goodbye|bye|see you|take care|thank you)\b",
                r"\b(that helps|feeling better|makes sense)\b"
            ]
        ]

        self.word_pattern = re.compile(r"\b\w+\b")

    def validate_conversation(self, conversation: Conversation) -> QualityResult:
        """Validate the quality of a conversation."""

        issues = []
        strengths = []

        # Basic structure validation
        structure_score = self._validate_structure(conversation, issues, strengths)

        # Content quality validation
        content_score = self._validate_content(conversation, issues, strengths)

        # Coherence validation
        coherence_score = self._validate_coherence(conversation, issues, strengths)

        # Authenticity validation
        authenticity_score = self._validate_authenticity(conversation, issues, strengths)

        # Completeness validation
        completeness_score = self._validate_completeness(conversation, issues, strengths)

        # Calculate overall score (weighted average)
        overall_score = (
            structure_score * 0.2 +
            content_score * 0.3 +
            coherence_score * 0.25 +
            authenticity_score * 0.15 +
            completeness_score * 0.1
        )

        result = QualityResult(
            conversation_id=conversation.id,
            overall_score=overall_score,
            coherence_score=coherence_score,
            authenticity_score=authenticity_score,
            completeness_score=completeness_score,
            issues=issues,
            strengths=strengths,
            metadata={
                "structure_score": structure_score,
                "content_score": content_score,
                "message_count": len(conversation.messages),
                "total_length": sum(len(msg.content) for msg in conversation.messages)
            },
            validated_at=datetime.now()
        )

        logger.debug(f"Validated conversation {conversation.id}: score {overall_score:.3f}")
        return result

    def _validate_structure(self, conversation: Conversation, issues: list[str], strengths: list[str]) -> float:
        """Validate conversation structure."""
        score = 1.0

        # Check message count
        message_count = len(conversation.messages)
        if message_count < self.min_conversation_length:
            issues.append(f"Too few messages ({message_count})")
            score -= 0.3
        elif message_count > self.max_conversation_length:
            issues.append(f"Too many messages ({message_count})")
            score -= 0.2
        else:
            strengths.append("Appropriate conversation length")

        # Check for alternating speakers (if roles are defined)
        if conversation.messages:
            roles = [msg.role for msg in conversation.messages if msg.role]
            if len(set(roles)) >= 2:
                strengths.append("Multiple speakers present")
            else:
                issues.append("Single speaker conversation")
                score -= 0.2

        # Check for empty messages
        empty_messages = sum(1 for msg in conversation.messages if not msg.content.strip())
        if empty_messages > 0:
            issues.append(f"{empty_messages} empty messages")
            score -= 0.1 * empty_messages

        return max(0.0, score)

    def _validate_content(self, conversation: Conversation, issues: list[str], strengths: list[str]) -> float:
        """Validate content quality."""
        score = 1.0

        for i, message in enumerate(conversation.messages):
            content = message.content.strip()

            # Check message length
            if len(content) < self.min_message_length:
                issues.append(f"Message {i+1} too short ({len(content)} chars)")
                score -= 0.1
            elif len(content) > self.max_message_length:
                issues.append(f"Message {i+1} too long ({len(content)} chars)")
                score -= 0.05

            # Check for low-quality patterns
            for pattern in self.low_quality_patterns:
                if pattern.search(content):
                    issues.append(f"Low-quality content detected in message {i+1}")
                    score -= 0.15
                    break

            # Check for therapeutic content
            therapeutic_matches = sum(
                1 for pattern in self.therapeutic_patterns
                if pattern.search(content)
            )

            if therapeutic_matches > 0:
                strengths.append(f"Therapeutic content in message {i+1}")
                score += 0.05

        return min(1.0, max(0.0, score))

    def _validate_coherence(self, conversation: Conversation, issues: list[str], strengths: list[str]) -> float:
        """Validate conversation coherence and flow."""
        score = 1.0

        if len(conversation.messages) < 2:
            return score

        # Check for conversation flow indicators
        flow_indicators = 0
        total_messages = len(conversation.messages)

        for message in conversation.messages:
            content = message.content.lower()

            # Count flow patterns
            for pattern in self.conversation_flow_patterns:
                if pattern.search(content):
                    flow_indicators += 1
                    break

        # Calculate flow score
        flow_ratio = flow_indicators / total_messages
        if flow_ratio > 0.5:
            strengths.append("Good conversation flow")
            score += 0.1
        elif flow_ratio < 0.2:
            issues.append("Poor conversation flow")
            score -= 0.2

        # Check for topic consistency (simple keyword overlap)
        all_words = set()
        message_words = []

        for message in conversation.messages:
            words = set(self.word_pattern.findall(message.content.lower()))
            words = {w for w in words if len(w) > 3}  # Filter short words
            message_words.append(words)
            all_words.update(words)

        if len(all_words) > 0:
            # Calculate topic consistency
            overlaps = []
            for i in range(len(message_words) - 1):
                overlap = len(message_words[i] & message_words[i + 1])
                total_unique = len(message_words[i] | message_words[i + 1])
                if total_unique > 0:
                    overlaps.append(overlap / total_unique)

            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                if avg_overlap > 0.3:
                    strengths.append("Good topic consistency")
                elif avg_overlap < 0.1:
                    issues.append("Poor topic consistency")
                    score -= 0.15

        return max(0.0, score)

    def _validate_authenticity(self, conversation: Conversation, issues: list[str], strengths: list[str]) -> float:
        """Validate conversation authenticity."""
        score = 1.0

        formal_count = 0
        natural_count = 0

        for message in conversation.messages:
            content = message.content.lower()

            # Count formal patterns
            for pattern in self.formal_patterns:
                formal_count += len(pattern.findall(content))

            # Count natural patterns
            for pattern in self.natural_patterns:
                natural_count += len(pattern.findall(content))

        # Evaluate authenticity
        total_patterns = formal_count + natural_count
        if total_patterns > 0:
            natural_ratio = natural_count / total_patterns
            if natural_ratio > 0.6:
                strengths.append("Natural, authentic language")
            elif natural_ratio < 0.3:
                issues.append("Overly formal or artificial language")
                score -= 0.2

        # Check for repetitive responses
        response_patterns = defaultdict(int)
        for message in conversation.messages:
            # Simple pattern: first 20 characters
            pattern = message.content[:20].lower().strip()
            if len(pattern) > 5:
                response_patterns[pattern] += 1

        max_repetition = max(response_patterns.values()) if response_patterns else 0
        if max_repetition > 2:
            issues.append("Repetitive responses detected")
            score -= 0.15

        return max(0.0, score)

    def _validate_completeness(self, conversation: Conversation, issues: list[str], strengths: list[str]) -> float:
        """Validate conversation completeness."""
        score = 1.0

        # Check if conversation has a clear beginning
        if conversation.messages:
            first_message = conversation.messages[0].content.lower()

            has_greeting = any(
                pattern.search(first_message)
                for pattern in self.greeting_patterns
            )

            if has_greeting:
                strengths.append("Clear conversation opening")
            else:
                issues.append("Abrupt conversation start")
                score -= 0.1

        # Check if conversation has reasonable closure
        if len(conversation.messages) > 2:
            last_message = conversation.messages[-1].content.lower()

            has_closure = any(
                pattern.search(last_message)
                for pattern in self.closure_patterns
            )

            if has_closure:
                strengths.append("Natural conversation closure")
            else:
                issues.append("Abrupt conversation ending")
                score -= 0.1

        # Check for context information
        if conversation.context:
            strengths.append("Rich context information")
        else:
            issues.append("Missing context information")
            score -= 0.05

        return max(0.0, score)
