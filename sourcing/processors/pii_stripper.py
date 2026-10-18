"""PII detection and redaction for ChatML records.

Uses regex-based heuristics to detect and redact personally identifiable
information from imported hackathon data. Designed for batch processing
without ML model dependencies.

Supported PII categories:
  - Email addresses
  - Phone numbers (US/international)
  - Social Security Numbers (SSN)
  - Credit card numbers
  - IP addresses (IPv4/IPv6)
  - Dates of birth
  - Medical record numbers
  - Street addresses (partial)
  - URLs containing user identifiers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Email
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # SSN (XXX-XX-XXXX or XX-XXXX-XXXX variations)
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Phone: (XXX) XXX-XXXX, XXX-XXX-XXXX, +X XXX-XXX-XXXX, etc.
    ("phone", re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")),
    # Credit card: 13-19 digit numbers with optional separators
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # IPv4
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # IPv6 (simplified)
    ("ipv6", re.compile(r"\b[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){7}\b")),
    # Date of birth patterns: MM/DD/YYYY, DD/MM/YYYY
    ("dob", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")),
    # Medical record number: MRN:XXXX or similar
    ("mrn", re.compile(r"\b(?:MRN|mrn|medical record)[:\s]*\d{5,12}\b", re.IGNORECASE)),
    # Street address (US pattern): number + street name + st/ave/rd/blvd/ln/dr
    ("address", re.compile(
        r"\b\d{1,6}\s+[A-Za-z0-9]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b",
        re.IGNORECASE,
    )),
    # URLs that might contain user identifiers (e.g., /user/12345, /profile/john)
    ("url_user_id", re.compile(
        r"https?://\S*/(?:user|profile|u|member)s?/\d{3,}\b\S*",
        re.IGNORECASE,
    )),
]

_REDACTION_PLACEHOLDER = "[REDACTED]"


@dataclass
class PIIStripResult:
    """Result of PII stripping on a single record."""

    record: dict[str, Any]
    redactions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def had_redactions(self) -> bool:
        return bool(self.redactions)


@dataclass
class PIIStripReport:
    """Aggregate report for a batch of records."""

    total_records: int = 0
    records_with_pii: int = 0
    total_redactions: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def add(self, result: PIIStripResult) -> None:
        self.total_records += 1
        if result.had_redactions:
            self.records_with_pii += 1
        for r in result.redactions:
            self.total_redactions += 1
            pii_type = r["type"]
            self.by_type[pii_type] = self.by_type.get(pii_type, 0) + 1


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def strip_pii_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Strip PII from a single text string.

    Returns the redacted text and a list of redaction metadata dicts.
    Each metadata dict has keys: type, match, position.
    """
    redactions: list[dict[str, Any]] = []
    redacted = text

    for pii_type, pattern in _PII_PATTERNS:
        for match in pattern.finditer(text):
            redactions.append({
                "type": pii_type,
                "match": match.group(),
                "position": match.start(),
            })
        redacted = pattern.sub(_REDACTION_PLACEHOLDER, redacted)

    return redacted, redactions


def strip_pii_from_record(record: dict[str, Any]) -> PIIStripResult:
    """Strip PII from all message contents in a ChatML record.

    Returns a new record with redacted content and a list of redaction metadata.
    """
    redactions: list[dict[str, Any]] = []
    new_record: dict[str, Any] = {}

    for key, value in record.items():
        if key == "messages" and isinstance(value, list):
            new_messages = []
            for i, msg in enumerate(value):
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        redacted, msg_redactions = strip_pii_from_text(content)
                        for r in msg_redactions:
                            r["message_index"] = i
                            r["role"] = msg.get("role", "unknown")
                        redactions.extend(msg_redactions)
                        new_msg = {**msg, "content": redacted}
                        new_messages.append(new_msg)
                    else:
                        new_messages.append(msg)
                else:
                    new_messages.append(msg)
            new_record["messages"] = new_messages
        else:
            new_record[key] = value

    return PIIStripResult(record=new_record, redactions=redactions)


def strip_pii_batch(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], PIIStripReport]:
    """Strip PII from a batch of ChatML records.

    Returns the list of redacted records and an aggregate report.
    """
    report = PIIStripReport()
    cleaned: list[dict[str, Any]] = []

    for record in records:
        result = strip_pii_from_record(record)
        report.add(result)
        cleaned.append(result.record)

    return cleaned, report
