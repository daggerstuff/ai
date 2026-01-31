"""
Security utilities for MCP Server.

This module provides input/output sanitization, security headers,
and other security-related utilities.
"""

import html
import json
import re
from typing import Any, Dict, List, Optional, Union

# Dangerous patterns that should be sanitized
DANGEROUS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'on\w+\s*=', re.IGNORECASE),  # Event handlers like onclick=
    re.compile(r'data:text/html', re.IGNORECASE),
    re.compile(r'vbscript:', re.IGNORECASE),
    re.compile(r'<iframe[^>]*>', re.IGNORECASE),
    re.compile(r'<object[^>]*>', re.IGNORECASE),
    re.compile(r'<embed[^>]*>', re.IGNORECASE),
]

# SQL injection patterns (for detection, not execution)
SQL_INJECTION_PATTERNS = [
    re.compile(r"('|(\\')|(;)|(\\;)|(--)|(\\--)|(/\*)|(\\/\*)|(\*/)|(\\\*/))", re.IGNORECASE),
    re.compile(r'\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b', re.IGNORECASE),
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    re.compile(r'[;&|`$()]', re.IGNORECASE),
    re.compile(r'\b(cat|ls|rm|mv|cp|chmod|chown|sudo|su|wget|curl|nc|netcat)\b', re.IGNORECASE),
]


class SecurityError(Exception):
    """Security-related error."""

    pass


def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize a string value by removing dangerous patterns and escaping HTML.

    Args:
        value: String value to sanitize
        max_length: Optional maximum length (truncate if exceeded)

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        # Convert to string if not already
        value = str(value)

    # Truncate if max length specified
    if max_length and len(value) > max_length:
        value = value[:max_length]

    # Remove dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        value = pattern.sub('', value)

    # Escape HTML entities
    value = html.escape(value, quote=True)

    return value


def sanitize_dict(data: Dict[str, Any], max_string_length: Optional[int] = None) -> Dict[str, Any]:
    """
    Recursively sanitize dictionary values.

    Args:
        data: Dictionary to sanitize
        max_string_length: Optional maximum length for string values

    Returns:
        Sanitized dictionary
    """
    sanitized: Dict[str, Any] = {}

    for key, value in data.items():
        # Sanitize key
        sanitized_key = sanitize_string(str(key), max_string_length)

        # Sanitize value based on type
        if isinstance(value, str):
            sanitized[sanitized_key] = sanitize_string(value, max_string_length)
        elif isinstance(value, dict):
            sanitized[sanitized_key] = sanitize_dict(value, max_string_length)
        elif isinstance(value, list):
            sanitized[sanitized_key] = sanitize_list(value, max_string_length)
        else:
            # For other types (int, float, bool, None), keep as-is
            sanitized[sanitized_key] = value

    return sanitized


def sanitize_list(data: List[Any], max_string_length: Optional[int] = None) -> List[Any]:
    """
    Recursively sanitize list values.

    Args:
        data: List to sanitize
        max_string_length: Optional maximum length for string values

    Returns:
        Sanitized list
    """
    sanitized: List[Any] = []

    for item in data:
        if isinstance(item, str):
            sanitized.append(sanitize_string(item, max_string_length))
        elif isinstance(item, dict):
            sanitized.append(sanitize_dict(item, max_string_length))
        elif isinstance(item, list):
            sanitized.append(sanitize_list(item, max_string_length))
        else:
            # For other types, keep as-is
            sanitized.append(item)

    return sanitized


def sanitize_input(data: Any, max_string_length: Optional[int] = None) -> Any:
    """
    Sanitize input data of any type.

    Args:
        data: Data to sanitize
        max_string_length: Optional maximum length for string values

    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        return sanitize_string(data, max_string_length)
    elif isinstance(data, dict):
        return sanitize_dict(data, max_string_length)
    elif isinstance(data, list):
        return sanitize_list(data, max_string_length)
    else:
        # For other types (int, float, bool, None), return as-is
        return data


def sanitize_output(data: Any, max_string_length: Optional[int] = 10000) -> Any:
    """
    Sanitize output data before sending to client.

    Args:
        data: Data to sanitize
        max_string_length: Maximum length for string values (default: 10000)

    Returns:
        Sanitized data
    """
    return sanitize_input(data, max_string_length)


def detect_sql_injection(value: str) -> bool:
    """
    Detect potential SQL injection patterns in string.

    Args:
        value: String to check

    Returns:
        True if potential SQL injection detected
    """
    if not isinstance(value, str):
        return False

    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(value):
            return True

    return False


def detect_command_injection(value: str) -> bool:
    """
    Detect potential command injection patterns in string.

    Args:
        value: String to check

    Returns:
        True if potential command injection detected
    """
    if not isinstance(value, str):
        return False

    for pattern in COMMAND_INJECTION_PATTERNS:
        if pattern.search(value):
            return True

    return False


def validate_and_sanitize_input(
    data: Any,
    max_string_length: Optional[int] = None,
    detect_injections: bool = True,
) -> Any:
    """
    Validate and sanitize input data, raising errors for dangerous patterns.

    Args:
        data: Data to validate and sanitize
        max_string_length: Optional maximum length for string values
        detect_injections: Whether to detect and raise errors for injection patterns

    Returns:
        Sanitized data

    Raises:
        SecurityError: If dangerous patterns detected and detect_injections is True
    """
    # Check for injection patterns if enabled
    if detect_injections:
        if isinstance(data, str):
            if detect_sql_injection(data):
                raise SecurityError("Potential SQL injection detected in input")
            if detect_command_injection(data):
                raise SecurityError("Potential command injection detected in input")
        elif isinstance(data, dict):
            for value in data.values():
                validate_and_sanitize_input(value, max_string_length, detect_injections)
        elif isinstance(data, list):
            for item in data:
                validate_and_sanitize_input(item, max_string_length, detect_injections)

    # Sanitize the data
    return sanitize_input(data, max_string_length)


def get_security_headers() -> Dict[str, str]:
    """
    Get security headers for HTTP responses.

    Returns:
        Dictionary of security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        ),
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        ),
    }


def validate_json_safe(data: Any) -> bool:
    """
    Validate that data can be safely serialized to JSON.

    Args:
        data: Data to validate

    Returns:
        True if data is JSON-safe
    """
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False


def sanitize_json_output(data: Any) -> Any:
    """
    Sanitize data for JSON output, ensuring it's safe to serialize.

    Args:
        data: Data to sanitize

    Returns:
        JSON-safe sanitized data
    """
    # First sanitize strings
    sanitized = sanitize_output(data)

    # Ensure JSON-serializable
    if not validate_json_safe(sanitized):
        # Convert non-serializable types to strings
        if isinstance(sanitized, dict):
            return {str(k): sanitize_json_output(v) for k, v in sanitized.items()}
        elif isinstance(sanitized, list):
            return [sanitize_json_output(item) for item in sanitized]
        else:
            return str(sanitized)

    return sanitized

