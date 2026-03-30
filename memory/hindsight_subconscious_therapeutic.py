"""
Therapeutic Subconscious - Specialized memory blocks and tools for Pixelated Empathy.

Extends the base Subconscious agent with therapeutic training capabilities:
- Therapist pattern tracking (interruption rates, empathy ratios, modality fidelity)
- Crisis detection (suicide risk, domestic violence markers)
- Rupture-repair tracking
- Breakthrough moment detection
- Countertransference indicators
- Ethics monitoring
- Peer benchmarking (anonymized)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ai.memory.hindsight_subconscious import SubconsciousAgent, MemoryBlock

logger = logging.getLogger("therapeutic_subconscious")

# P0 Fix: PII Detection patterns
PII_PATTERNS = {
    "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    "credit_card": (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD_REDACTED]"),
}

def redact_pii(content: str) -> str:
    """
    P0 Fix: Redact PII from content before storage.

    Args:
        content: Content to redact

    Returns:
        Content with PII replaced with redaction markers
    """
    result = content
    for pattern_name, (pattern, replacement) in PII_PATTERNS.items():
        result = re.sub(pattern, replacement, result)
    return result
