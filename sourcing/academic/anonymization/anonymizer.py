"""
Content anonymization module for academic sourcing.

Detects and redacts PII from documents before they enter the training pipeline.
This module wraps the privacy verifier's PII patterns into a standalone service
that can be used by the academic sourcing pipeline.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# PII patterns
_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_us": re.compile(r"\b(?:\+?1[\-\s]?)?\(?\d{3}\)?[\-\s.]?\d{3}[\-\s.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "date_of_birth": re.compile(r"\b(?:DOB|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    "street_address": re.compile(r"\b\d+\s+[A-Za-z]+\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Blvd|Boulevard|Ln|Lane)\b"),
    "zip_code": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}


@dataclass
class AnonymizationResult:
    """Result of content anonymization."""

    anonymized_text: str
    redactions: list[dict[str, str | int]] = field(default_factory=list)
    total_pii_found: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "anonymized_text": self.anonymized_text,
            "redactions": self.redactions,
            "total_pii_found": self.total_pii_found,
        }


class ContentAnonymizer:
    """Detect and redact PII from text content.

    Supports configurable PII patterns and two redaction modes:
    - ``mask`` (default): replace PII with ``[REDACTED_TYPE]``
    - ``hash``: replace PII with a short hash (useful for consistent linkage)
    """

    def __init__(self, custom_patterns: dict[str, str] | None = None) -> None:
        self._patterns: dict[str, re.Pattern[str]] = dict(_PATTERNS)
        if custom_patterns:
            for name, pattern_str in custom_patterns.items():
                self._patterns[name] = re.compile(pattern_str)

    def anonymize(self, text: str, mode: str = "mask") -> AnonymizationResult:
        """Anonymize text by detecting and redacting PII.

        Args:
            text: Input text to anonymize.
            mode: ``mask`` replaces PII with ``[REDACTED_TYPE]``,
                  ``hash`` replaces with a short hash token.

        Returns:
            AnonymizationResult with redacted text and redaction details.
        """
        result_text = text
        redactions: list[dict[str, str | int]] = []
        total = 0

        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(result_text):
                redactions.append({
                    "type": pii_type,
                    "position": match.start(),
                    "original_length": len(match.group(0)),
                })
                total += 1
            result_text = pattern.sub(
                lambda m, t=pii_type: (f"[HASH:{hashlib.sha256(m.group(0).encode()).hexdigest()[:8]}]"
                                       if mode == "hash"
                                       else f"[REDACTED_{t.upper()}]"),
                result_text,
            )

        return AnonymizationResult(
            anonymized_text=result_text,
            redactions=redactions,
            total_pii_found=total,
        )

    def detect_pii(self, text: str) -> list[dict[str, str | int]]:
        """Detect PII in text without redacting.

        Returns:
            List of dicts with ``type``, ``position``, and ``match`` keys.
        """
        findings: list[dict[str, str | int]] = []
        for pii_type, pattern in self._patterns.items():
            for match in pattern.finditer(text):
                findings.append({
                    "type": pii_type,
                    "position": match.start(),
                    "match": match.group(0),
                })
        return findings

    def has_pii(self, text: str) -> bool:
        """Quick check whether any PII pattern matches."""
        return any(pattern.search(text) for pattern in self._patterns.values())
