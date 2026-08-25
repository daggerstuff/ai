"""PHI sanitization utilities for safe logging.

These functions redact or mask Protected Health Information (PHI) before
writing to logs. No PHI is ever persisted to disk — these utilities only
ensure that log output does not contain identifiable information.
"""

from __future__ import annotations

import hashlib
import re
from typing import Final

# --- Regex patterns for PHI detection ---

# Phone numbers: (XXX) XXX-XXXX or XXX-XXX-XXXX
_PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Email addresses
_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

# SSN: XXX-XX-XXXX
_SSN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Dates in common formats: MM/DD/YYYY, YYYY-MM-DD
_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)

# MRN-like patterns: letters + digits, 6-12 chars, prefixed by MRN/MRN:
_MRN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:MRN|mrn)[:\s]*[A-Za-z0-9]{6,12}\b",
    re.IGNORECASE,
)

# Street addresses: digits + street name
_ADDRESS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Dr|Drive|Ln|Lane|Ct|Court)\b",
    re.IGNORECASE,
)


def sanitize_for_logging(text: str) -> str:
    """Redact PHI patterns from text before logging.

    Replaces phone numbers, emails, SSNs, dates, MRNs, and addresses
    with ``[REDACTED-{type}]`` tokens. This is a best-effort sanitizer —
    it does not guarantee complete PHI removal for all edge cases, but
    covers the most common identifiable patterns.

    Args:
        text: Raw text that may contain PHI.

    Returns:
        Sanitized text safe for log output.
    """
    return (
        _ADDRESS_PATTERN.sub(
            "[REDACTED-ADDRESS]",
            _MRN_PATTERN.sub(
                "[REDACTED-MRN]",
                _DATE_PATTERN.sub(
                    "[REDACTED-DATE]",
                    _SSN_PATTERN.sub(
                        "[REDACTED-SSN]",
                        _EMAIL_PATTERN.sub(
                            "[REDACTED-EMAIL]",
                            _PHONE_PATTERN.sub("[REDACTED-PHONE]", text),
                        ),
                    ),
                ),
            ),
        )
    )


def redact_patient_id(patient_id: str) -> str:
    """Produce a non-reversible hash prefix for logging patient IDs.

    Args:
        patient_id: The raw patient identifier.

    Returns:
        A masked identifier like ``pid:ab12cd`` (first 6 chars of SHA-256).
    """
    digest = hashlib.sha256(patient_id.encode()).hexdigest()
    return f"pid:{digest[:6]}"


def redact_session_id(session_id: str) -> str:
    """Produce a non-reversible hash prefix for logging session IDs.

    Args:
        session_id: The raw session identifier.

    Returns:
        A masked identifier like ``sid:ef34gh`` (first 6 chars of SHA-256).
    """
    digest = hashlib.sha256(session_id.encode()).hexdigest()
    return f"sid:{digest[:6]}"
