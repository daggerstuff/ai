"""
Production-grade PII (Personally Identifiable Information) scrubber for mental health data processing.
Detects and redacts PII types including:
- Names (person names)
- Email addresses
- Phone numbers (various formats)
- SSN/Social Insurance Numbers
- Dates of birth
- Medical record numbers
- Addresses
- Credit card numbers
- IP addresses
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiiScrubberConfig:
    """Configuration for PiiScrubber."""

    # Redaction style options: "REDACTED", "[TYPE]", "hash", "mask"
    redaction_style: str = "[TYPE]"

    # Whether to preserve original text length with masking
    preserve_length: bool = False

    # Custom patterns to add to default detection
    custom_patterns: dict[str, str] = field(default_factory=dict)

    # PII types to detect (None means all)
    pii_types: list[str] | None = None

    # Minimum confidence score for spaCy entities (0.0 to 1.0)
    spacy_confidence_threshold: float = 0.5

    # Whether to use spaCy for name detection (requires spaCy model)
    use_spacy_for_names: bool = True

    # Whether to log PII findings (counts only, not actual PII)
    log_findings: bool = True


@dataclass
class ScrubResult:
    """Result of PII scrubbing operation."""

    scrubbed_text: str
    pii_counts: dict[str, int] = field(default_factory=dict)
    total_pii_count: int = 0

    def __post_init__(self):
        """Calculate total PII count after initialization."""
        object.__setattr__(self, "total_pii_count", sum(self.pii_counts.values()))


class PiiScrubber:
    """
    Production-grade PII scrubber for mental health data processing.

    Features:
    - Detects and redacts multiple PII types using regex and spaCy NER
    - Configurable redaction styles (REDACTED, [TYPE], hash, mask)
    - Support for text, JSON, and dict inputs
    - Batch processing capability
    - Logging of PII found (without exposing PII)
    - Returns both scrubbed content and PII count metadata
    - Works offline with deterministic output
    """

    def __init__(self, config: PiiScrubberConfig | None = None):
        """Initialize PiiScrubber with optional configuration."""
        self.config = config or PiiScrubberConfig()
        self._pii_types_found: list[str] = []

        # Initialize regex patterns
        self._regex_patterns = self._compile_regex_patterns()

        # Initialize spaCy if available and requested
        self._nlp = None
        if self.config.use_spacy_for_names and SPACY_AVAILABLE:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded for NER-based name detection")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm"
                )
                self._nlp = None
        elif self.config.use_spacy_for_names and not SPACY_AVAILABLE:
            logger.warning("spaCy not available. Install with: pip install spacy")

    def _compile_regex_patterns(self) -> dict[str, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {
            # Email addresses
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE),
            # Phone numbers (various formats)
            "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
            # SSN (Social Security Number)
            "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),
            # Dates of birth (common formats)
            "dob": re.compile(r"\b(?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b"),
            # Medical record numbers (common patterns)
            "medical_record_number": re.compile(
                r"\b(?:MRN|mrn|Medical\s+Record\s+#?|MR#?)\s*:?\s*[A-Z0-9]{6,12}\b",
                re.IGNORECASE,
            ),
            # Addresses (simplified pattern)
            "address": re.compile(
                r"\b\d+\s+[A-Za-z0-9\s,.'-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b",
                re.IGNORECASE,
            ),
            # Credit card numbers (basic pattern with Luhn check in validation)
            "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            # IP addresses (IPv4)
            "ip_address": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
        }

        # Add custom patterns if provided
        patterns.update(self.config.custom_patterns)

        return patterns

    def _redact_match(self, match: str, pii_type: str) -> str:
        """Apply redaction style to a matched PII item."""
        if self.config.redaction_style == "REDACTED":
            return "[REDACTED]"
        if self.config.redaction_style == "[TYPE]":
            return f"[{pii_type.upper()}]"
        if self.config.redaction_style == "hash":
            # Return consistent hash for same input
            hash_obj = hashlib.md5(match.encode())
            return f"[HASH:{hash_obj.hexdigest()[:8]}]"
        if self.config.redaction_style == "mask":
            if self.config.preserve_length:
                # Preserve original length with asterisks
                return "*" * len(match)
            # Fixed length mask
            return "[MASKED]"
        # Default to [TYPE] style
        return f"[{pii_type.upper()}]"

    def _detect_pii_with_regex(self, text: str) -> list[tuple[str, str, int, int]]:
        """
        Detect PII using regex patterns.

        Returns:
            List of tuples: (pii_type, matched_text, start_pos, end_pos)
        """
        matches = []

        for pii_type, pattern in self._regex_patterns.items():
            # Skip if specific PII types are requested and this isn't one of them
            if self.config.pii_types and pii_type not in self.config.pii_types:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()

                # Additional validation for certain types
                if pii_type == "credit_card":
                    # Basic Luhn check for credit card numbers
                    if not self._is_valid_credit_card(matched_text):
                        continue
                elif pii_type == "ip_address":
                    # Validate IP address octets
                    if not self._is_valid_ip_address(matched_text):
                        continue

                matches.append((pii_type, matched_text, match.start(), match.end()))

        return matches

    def _detect_pii_with_spacy(self, text: str) -> list[tuple[str, str, int, int]]:
        """
        Detect PII (specifically names) using spaCy NER.

        Returns:
            List of tuples: (pii_type, matched_text, start_pos, end_pos)
        """
        if not self._nlp:
            return []

        matches = []
        doc = self._nlp(text)

        for ent in doc.ents:
            # Focus on person names for PII detection
            if ent.label_ == "PERSON" and ent.confidence >= self.config.spacy_confidence_threshold:
                # Skip if specific PII types are requested and this isn't one of them
                if self.config.pii_types and "name" not in self.config.pii_types:
                    continue

                matches.append(("name", ent.text, ent.start_char, ent.end_char))

        return matches

    def _is_valid_credit_card(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        # Remove non-digits
        digits = [int(d) for d in number if d.isdigit()]

        # Must be 13-16 digits
        if not (13 <= len(digits) <= 16):
            return False

        # Luhn algorithm
        checksum = 0
        parity = len(digits) % 2

        for i, digit in enumerate(digits):
            if i % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0

    def _is_valid_ip_address(self, ip: str) -> bool:
        """Validate IP address format."""
        parts = ip.split(".")
        if len(parts) != 4:
            return False

        for part in parts:
            try:
                num = int(part)
                if not (0 <= num <= 255):
                    return False
            except ValueError:
                return False

        return True

    def _scrub_text_internal(self, text: str) -> ScrubResult:
        """
        Internal method to scrub PII from text.

        Returns:
            ScrubResult with scrubbed text and PII counts
        """
        if not text or not isinstance(text, str):
            return ScrubResult(scrubbed_text=text)

        # Detect PII using both regex and spaCy
        regex_matches = self._detect_pii_with_regex(text)
        spacy_matches = self._detect_pii_with_spacy(text)

        # Combine and sort matches by position
        all_matches = regex_matches + spacy_matches
        all_matches.sort(key=lambda x: x[2])  # Sort by start position

        # Filter out overlapping matches (prefer earlier/larger matches)
        filtered_matches = []
        prev_end = 0

        for pii_type, matched_text, start, end in all_matches:
            if start >= prev_end:  # No overlap with previous match
                filtered_matches.append((pii_type, matched_text, start, end))
                prev_end = end
            elif end > prev_end:  # Overlap - extend previous match if current is longer
                # Replace previous match with current one if it's longer
                if filtered_matches and (end - filtered_matches[-1][3]) > (
                    filtered_matches[-1][3] - filtered_matches[-1][2]
                ):
                    filtered_matches[-1] = (pii_type, matched_text, start, end)
                    prev_end = end

        # Apply redactions from end to start to preserve positions
        scrubbed_text = text
        pii_counts: dict[str, int] = {}

        # Process matches in reverse order to maintain positions
        for pii_type, matched_text, start, end in reversed(filtered_matches):
            redaction = self._redact_match(matched_text, pii_type)
            scrubbed_text = scrubbed_text[:start] + redaction + scrubbed_text[end:]

            # Count PII instances
            pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

        # Track PII types found for reporting
        if self.config.log_findings and pii_counts:
            self._pii_types_found = list(pii_counts.keys())
            pii_summary = ", ".join(f"{k}: {v}" for k, v in pii_counts.items())
            logger.info(f"PII scrubbing completed. Found: {pii_summary}")

        return ScrubResult(
            scrubbed_text=scrubbed_text,
            pii_counts=pii_counts,
            total_pii_count=sum(pii_counts.values()),
        )

    def scrub(self, text: str) -> ScrubResult:
        """
        Scrub PII from a single text string.

        Args:
            text: Input text to scrub

        Returns:
            ScrubResult containing scrubbed text and PII metadata
        """
        return self._scrub_text_internal(text)

    def scrub_batch(self, texts: list[str]) -> list[ScrubResult]:
        """
        Scrub PII from a batch of text strings.

        Args:
            texts: List of input texts to scrub

        Returns:
            List of ScrubResult objects
        """
        return [self.scrub(text) for text in texts]

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Scrub PII from dictionary values (recursively).

        Args:
            data: Input dictionary to scrub

        Returns:
            Dictionary with PII scrubbed from string values
        """
        if not isinstance(data, dict):
            return data

        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                scrub_result = self.scrub(value)
                result[key] = scrub_result.scrubbed_text

            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = self.scrub_list(value)
            else:
                result[key] = value

        return result

    def scrub_list(self, data: list[Any]) -> list[Any]:
        """
        Scrub PII from list values (recursively).

        Args:
            data: Input list to scrub

        Returns:
            List with PII scrubbed from string values
        """
        if not isinstance(data, list):
            return data

        result = []
        for item in data:
            if isinstance(item, str):
                scrub_result = self.scrub(item)
                result.append(scrub_result.scrubbed_text)
            elif isinstance(item, dict):
                result.append(self.scrub_dict(item))
            elif isinstance(item, list):
                result.append(self.scrub_list(item))
            else:
                result.append(item)

        return result

    def get_pii_types_found(self) -> list[str]:
        """
        Get list of PII types found in the last scrubbing operation.

        Returns:
            List of PII type strings that were detected
        """
        return self._pii_types_found.copy()


# Convenience functions for quick usage
def scrub_text(text: str, config: PiiScrubberConfig | None = None) -> str:
    """
    Quick function to scrub text and return only the scrubbed string.

    Args:
        text: Input text to scrub
        config: Optional PiiScrubberConfig

    Returns:
        Scrubbed text string
    """
    scrubber = PiiScrubber(config)
    result = scrubber.scrub(text)
    return result.scrubbed_text


def scrub_text_with_metadata(text: str, config: PiiScrubberConfig | None = None) -> ScrubResult:
    """
    Quick function to scrub text and return full metadata.

    Args:
        text: Input text to scrub
        config: Optional PiiScrubberConfig

    Returns:
        ScrubResult with scrubbed text and PII metadata
    """
    scrubber = PiiScrubber(config)
    return scrubber.scrub(text)


# Export public interface
__all__ = [
    "PiiScrubber",
    "PiiScrubberConfig",
    "ScrubResult",
    "scrub_text",
    "scrub_text_with_metadata",
]
