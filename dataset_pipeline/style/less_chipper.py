#!/usr/bin/env python3
"""
Less-Chipper Tone Labeling and Enforcement Module
Implements tone labeling for crisis and complex cases to prevent overly optimistic responses.
Based on the "Less Chipper" response style training from the expanded project brief.
"""

from enum import Enum
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import re

from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.style.less_chipper")


class Tone(Enum):
    """
    Therapeutic response tone levels.
    Matches the brief's distinction between foundation, clinical, and crisis-direct styles.
    """
    FOUNDATION = "foundation"  # Gentle, supportive, rapport-building
    CLINICAL = "clinical"  # Matter-of-fact, professional, evidence-based
    CRISIS_DIRECT = "crisis_direct"  # Direct, serious, minimal fluff, appropriate for crisis


@dataclass
class ToneLabel:
    """Tone label with confidence and metadata"""
    tone: Tone
    confidence: float = 1.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tone label to dictionary"""
        return {
            "tone": self.tone.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


class LessChipperToneLabeler:
    """
    Labels therapeutic responses with appropriate tone levels.
    Prevents generic post-processors from inserting saccharine positivity.
    """

    # Patterns that indicate overly chipper/toxic positivity
    TOXIC_POSITIVITY_PATTERNS = [
        r"everything.*will.*be.*okay",
        r"just.*think.*positive",
        r"look.*on.*the.*bright.*side",
        r"it.*could.*be.*worse",
        r"everything.*happens.*for.*a.*reason",
        r"just.*stay.*positive",
        r"you.*got.*this",
        r"everything.*will.*work.*out",
        r"don.*t.*worry.*be.*happy",
        r"just.*smile.*more",
    ]

    # Patterns that indicate crisis-direct tone
    CRISIS_DIRECT_PATTERNS = [
        r"immediate.*danger",
        r"suicide.*plan",
        r"self.*harm",
        r"emergency.*services",
        r"crisis.*line",
        r"immediate.*help",
        r"serious.*concern",
        r"urgent.*situation",
        r"life.*threatening",
        r"imminent.*risk",
    ]

    # Patterns that indicate clinical/professional tone
    CLINICAL_PATTERNS = [
        r"evidence.*based",
        r"research.*shows",
        r"clinical.*practice",
        r"therapeutic.*technique",
        r"diagnostic.*criteria",
        r"treatment.*protocol",
        r"professional.*assessment",
    ]

    # Patterns that indicate foundation/rapport tone
    FOUNDATION_PATTERNS = [
        r"i.*understand",
        r"that.*sounds.*difficult",
        r"thank.*you.*for.*sharing",
        r"i.*hear.*you",
        r"let.*s.*explore",
        r"tell.*me.*more",
    ]

    def __init__(self):
        """Initialize the tone labeler"""
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self._toxic_positivity_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.TOXIC_POSITIVITY_PATTERNS
        ]
        self._crisis_direct_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CRISIS_DIRECT_PATTERNS
        ]
        self._clinical_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.CLINICAL_PATTERNS
        ]
        self._foundation_re = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.FOUNDATION_PATTERNS
        ]

    def label_tone(self, text: str, context: Optional[Dict[str, Any]] = None) -> ToneLabel:
        """
        Label the tone of a therapeutic response.

        Args:
            text: The response text to label
            context: Optional context (e.g., crisis indicators, conversation history)

        Returns:
            ToneLabel with tone classification
        """
        text_lower = text.lower()

        # Check for toxic positivity (should be flagged regardless of other signals)
        has_toxic_positivity = any(
            pattern.search(text_lower) for pattern in self._toxic_positivity_re
        )

        if has_toxic_positivity:
            logger.warning("Detected toxic positivity patterns in response")

        # Check context for crisis indicators
        is_crisis_context = False
        if context:
            crisis_flags = context.get("crisis_indicators", [])
            intensity = context.get("intensity", "").lower()
            is_crisis_context = (
                len(crisis_flags) > 0 or
                intensity in ["very_high", "extreme", "high"] or
                context.get("is_crisis", False)
            )

        # Score patterns
        crisis_score = sum(bool(pattern.search(text_lower))
                       for pattern in self._crisis_direct_re)
        clinical_score = sum(bool(pattern.search(text_lower))
                         for pattern in self._clinical_re)
        foundation_score = sum(bool(pattern.search(text_lower))
                           for pattern in self._foundation_re)

        # Determine tone based on scores and context
        if is_crisis_context or crisis_score > 0:
            # Crisis context or crisis language detected -> CRISIS_DIRECT
            tone = Tone.CRISIS_DIRECT
            reasoning = "Crisis context or crisis language detected"
            confidence = 0.9 if is_crisis_context else 0.7
        elif clinical_score > foundation_score and clinical_score > 0:
            # Clinical language dominates -> CLINICAL
            tone = Tone.CLINICAL
            reasoning = f"Clinical patterns detected (score: {clinical_score})"
            confidence = 0.8
        elif foundation_score > 0:
            # Foundation/rapport patterns -> FOUNDATION
            tone = Tone.FOUNDATION
            reasoning = f"Foundation/rapport patterns detected (score: {foundation_score})"
            confidence = 0.75
        else:
            # Default to CLINICAL for professional responses
            tone = Tone.CLINICAL
            reasoning = "Default clinical tone (no strong patterns detected)"
            confidence = 0.5

        return ToneLabel(
            tone=tone,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "has_toxic_positivity": has_toxic_positivity,
                "crisis_score": crisis_score,
                "clinical_score": clinical_score,
                "foundation_score": foundation_score,
                "is_crisis_context": is_crisis_context,
            }
        )

    def enforce_less_chipper_policy(
        self,
        text: str,
        tone: Union[Tone, str],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, bool]:
        """
        Enforce less-chipper policy: prevent generic post-processors from inserting
        saccharine positivity into CRISIS_DIRECT examples.

        Args:
            text: The response text to check/enforce
            tone: The tone label (Tone enum or string)
            context: Optional context

        Returns:
            Tuple of (processed_text, was_modified)
        """
        # Normalize tone - raise error for unknown tones instead of silently defaulting
        if isinstance(tone, str):
            try:
                tone_enum = Tone(tone.lower())
            except ValueError as e:
                allowed_tones = [t.value for t in Tone]
                raise ValueError(
                    f"Unknown tone string: {tone}. "
                    f"Allowed tones: {', '.join(allowed_tones)}"
                ) from e
        else:
            tone_enum = tone

        # For CRISIS_DIRECT tone, check for and remove toxic positivity
        if tone_enum == Tone.CRISIS_DIRECT:
            modified = False
            processed_text = text

            # Check for toxic positivity patterns
            for pattern in self._toxic_positivity_re:
                if pattern.search(processed_text.lower()):
                    logger.warning(
                        f"Removing toxic positivity from CRISIS_DIRECT response: "
                        f"{pattern.pattern}"
                    )
                    # Remove the matching phrase with better text cleanup
                    processed_text = pattern.sub("", processed_text)
                    modified = True

            # Clean up extra whitespace and orphaned punctuation
            if modified:
                # Clean up orphaned punctuation sequences (safe from ReDoS)
                # Use bounded quantifiers to prevent catastrophic backtracking
                # Pattern: punctuation, optional whitespace (max 10 chars), 1-10 punctuation marks
                processed_text = re.sub(r'[,.;]\s{0,10}[,.;]{1,10}', '.', processed_text)
                # Clean up whitespace
                processed_text = re.sub(r'\s+', ' ', processed_text).strip()

            return processed_text, modified

        # For other tones, just return as-is (but could add checks here)
        return text, False

    def validate_tone_appropriateness(
        self,
        text: str,
        expected_tone: Union[Tone, str],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that a response's tone matches the expected tone for the context.

        Args:
            text: Response text
            expected_tone: Expected tone (Tone enum or string)
            context: Optional context

        Returns:
            Tuple of (is_appropriate, error_message)
            - is_appropriate=True only when tones match exactly
            - is_appropriate=False whenever there's any mismatch (error_message is non-None)
        """
        # Normalize expected tone
        if isinstance(expected_tone, str):
            try:
                expected_tone_enum = Tone(expected_tone.lower())
            except ValueError:
                return False, f"Unknown expected tone: {expected_tone}"
        else:
            expected_tone_enum = expected_tone

        # Label the actual tone
        actual_label = self.label_tone(text, context)
        actual_tone = actual_label.tone

        # Exact match: appropriate
        if actual_tone == expected_tone_enum:
            return True, None

        # All mismatches are treated as inappropriate, with varying messages
        if expected_tone_enum == Tone.CRISIS_DIRECT and actual_tone == Tone.FOUNDATION:
            return False, (
                "Response tone is FOUNDATION but expected CRISIS_DIRECT. "
                "Crisis responses should be direct and serious, not overly gentle."
            )

        if expected_tone_enum == Tone.CRISIS_DIRECT and actual_label.metadata.get("has_toxic_positivity", False):
            return False, (
                "Response contains toxic positivity patterns. "
                "Crisis responses should avoid dismissive optimism."
            )

        # Generic mismatch message for all other cases
        return False, (
            f"Tone mismatch: expected {expected_tone_enum.value}, "
            f"got {actual_tone.value} (confidence: {actual_label.confidence:.2f})"
        )


# Module-level labeler instance for performance (reuse compiled regexes)
_labeler = LessChipperToneLabeler()


def label_tone(text: str, context: Optional[Dict[str, Any]] = None) -> ToneLabel:
    """
    Convenience function to label tone of a response.

    Args:
        text: Response text
        context: Optional context

    Returns:
        ToneLabel
    """
    return _labeler.label_tone(text, context)


def enforce_less_chipper_policy(
    text: str,
    tone: Union[Tone, str],
    context: Optional[Dict[str, Any]] = None
) -> tuple[str, bool]:
    """
    Convenience function to enforce less-chipper policy.

    Args:
        text: Response text
        tone: Tone label
        context: Optional context

    Returns:
        Tuple of (processed_text, was_modified)
    """
    return _labeler.enforce_less_chipper_policy(text, tone, context)

