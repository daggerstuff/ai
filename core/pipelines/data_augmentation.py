"""Data augmentation primitives for therapeutic conversation datasets."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass

from .schemas.conversation_schema import Conversation


@dataclass
class AugmentationConfig:
    paraphrase_enabled: bool = True
    paraphrase_probability: float = 0.35
    contextual_augmentation_enabled: bool = True
    contextual_probability: float = 0.25
    noise_injection_enabled: bool = True
    noise_probability: float = 0.20
    demographic_variation_enabled: bool = True
    demographic_variation_probability: float = 0.15
    safety_guardrails_enabled: bool = True
    random_seed: int | None = None


@dataclass
class AugmentationStats:
    total: int = 0
    augmented: int = 0
    rejected: int = 0
    reasons: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = {}


class SafetyGuardrails:
    """Guardrails that prevent critical safety information loss during augmentation."""

    CRITICAL_KEYWORDS = {
        "suicidal",
        "suicide",
        "kill myself",
        "hurt myself",
        "self-harm",
        "self harm",
        "i want to die",
        "end my life",
    }

    def validate_augmentation(self, original: str, candidate: str) -> tuple[bool, list[str]]:
        issues: list[str] = []
        original_lower = original.lower()
        candidate_lower = candidate.lower()

        # Preserve critical crisis-related expressions.
        for keyword in self.CRITICAL_KEYWORDS:
            if keyword in original_lower and keyword not in candidate_lower:
                issues.append(f"CRITICAL: lost safety phrase '{keyword}'")

        if re.search(r"\b(harm\b|hurt\b|die\b)", candidate_lower) and not re.search(
            r"\b(safety\b|support\b|help\b|resource\b)", candidate_lower
        ):
            issues.append("CRITICAL: candidate may contain high-risk language without escalation cues")

        return len(issues) == 0, issues


class DataAugmenter:
    """Rule-based data augmenter with deterministic optional randomness."""

    _PARAPHRASE_MAP = (
        ("I understand how you feel", "I can see why that would be difficult"),
        ("That must be hard", "I imagine that's challenging"),
        ("I know it can be overwhelming", "It makes sense that this can feel overwhelming"),
        ("let's work on", "we can work together on"),
    )

    _DEMOGRAPHIC_MAP = {
        "he": "she",
        "she": "he",
        "him": "her",
        "her": "him",
        "man": "woman",
        "woman": "man",
        "his": "her",
        "hers": "his",
        "father": "mother",
        "mother": "father",
        "boy": "girl",
        "girl": "boy",
    }

    _NOISE_PATTERNS = (
        "um",
        "uh",
        "you know",
        "like",
        "really",
        "I see",
        "so",
    )

    def __init__(
        self, config: AugmentationConfig | None = None, *, random_fn: Callable[[], float] | None = None
    ) -> None:
        self.config = config or AugmentationConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        self.random = random_fn if random_fn is not None else random.random

    def paraphrase_text(self, text: str) -> str:
        """Apply lightweight phrase-level paraphrasing."""

        output = text
        for source, target in self._PARAPHRASE_MAP:
            if source.lower() in output.lower() and self.random() < self.config.paraphrase_probability:
                # Preserve case loosely and perform case-insensitive replacement once.
                pattern = re.compile(re.escape(source), re.IGNORECASE)
                output = pattern.sub(target, output, count=1)
        return output

    def inject_noise(self, text: str) -> str:
        """Inject controlled speech-level noise and punctuation."""

        tokens = text.split()
        if not tokens:
            return text

        if self.random() > self.config.noise_probability:
            return text

        noise = self._NOISE_PATTERNS[int(self.random() * len(self._NOISE_PATTERNS))]
        index = int(self.random() * min(len(tokens), 5))
        tokens.insert(index, noise)

        if self.random() < 0.5:
            tokens.append("...")
        if self.random() < 0.5:
            tokens[-1] = f"{tokens[-1]}!"

        return " ".join(tokens)

    def demographic_variation(self, text: str) -> str:
        """Swap gender-coded terms to provide demographic robustness."""

        output = text
        if self.random() >= self.config.demographic_variation_probability:
            return output

        for source, replacement in self._DEMOGRAPHIC_MAP.items():
            pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
            if pattern.search(output):
                output = pattern.sub(lambda m: self._match_case(replacement, m.group(0)), output)
        return output

    def _match_case(self, replacement: str, original: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement

    def augment_text(self, text: str) -> str:
        if not text:
            return text

        current = text
        if self.config.paraphrase_enabled:
            current = self.paraphrase_text(current)
        if self.config.noise_injection_enabled:
            current = self.inject_noise(current)
        if self.config.demographic_variation_enabled:
            current = self.demographic_variation(current)
        return current

    def _augment_message(self, message_text: str) -> str:
        if self.random() <= 0.35:
            return self.augment_text(message_text)
        return message_text

    def augment_conversation(self, conversation: Conversation) -> Conversation:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be Conversation")

        safe = SafetyGuardrails()
        target = Conversation(
            conversation_id=f"{conversation.conversation_id}_aug",
            metadata=dict(conversation.metadata),
        )

        for message in conversation.messages:
            original = message.content
            candidate = self._augment_message(original)
            if self.config.safety_guardrails_enabled:
                ok, issues = safe.validate_augmentation(original, candidate)
                if not ok:
                    # Keep original and preserve safety-critical phrasing.
                    candidate = original
                    target.metadata.setdefault("guardrail_rejections", []).append({"text": original, "issues": issues})
            target.messages.append(type(message)(role=message.role, content=candidate))

        return target

    def batch_augment(self, conversations: list[Conversation]) -> tuple[list[Conversation], AugmentationStats]:
        """Augment conversation list and return stats."""

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        output: list[Conversation] = []
        stats = AugmentationStats()
        for conversation in conversations:
            stats.total += 1
            augmented = self.augment_conversation(conversation)
            if augmented.to_dict() != conversation.to_dict():
                stats.augmented += 1
            else:
                stats.rejected += 1
                stats.reasons = stats.reasons or {}
                stats.reasons["unchanged"] = stats.reasons.get("unchanged", 0) + 1
            output.append(augmented)
        return output, stats


__all__ = [
    "AugmentationConfig",
    "AugmentationStats",
    "DataAugmenter",
    "SafetyGuardrails",
]
