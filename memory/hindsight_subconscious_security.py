from __future__ import annotations

import logging
import re

logger = logging.getLogger("hindsight_subconscious.security")

MAX_CONTENT_LENGTH = 100000
DANGEROUS_PATTERNS = [
    r"<script[^>]*>",
    r"javascript:",
    r"data:",
]


def validate_and_sanitize_content(content: str, *, field_name: str = "content") -> str:
    if not content:
        return ""
    sanitized = content
    if len(sanitized) > MAX_CONTENT_LENGTH:
        logger.warning(
            "%s exceeds max length (%s/%s), truncating",
            field_name,
            len(sanitized),
            MAX_CONTENT_LENGTH,
        )
        sanitized = sanitized[:MAX_CONTENT_LENGTH]
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning("Dangerous pattern detected in %s, removing", field_name)
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
    return sanitized
