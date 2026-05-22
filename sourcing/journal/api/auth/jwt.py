"""
JWT token utilities for authentication.

This module provides JWT token creation, validation, and decoding.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt as pyjwt
from jwt.exceptions import DecodeError, ExpiredSignatureError, InvalidTokenError

from ai.sourcing.journal.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.jwt_expiration_minutes)
    to_encode.update({"exp": expire, "iat": datetime.now(UTC)})
    return pyjwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode a JWT access token."""
    try:
        return pyjwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except ExpiredSignatureError:
        raise ValueError("Token has expired") from None
    except DecodeError:
        raise ValueError("Invalid token format") from None
    except InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e!s}") from e


def verify_token(token: str) -> dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        return decode_access_token(token)
    except ValueError as e:
        logger.warning(f"Token verification failed: {e}")
        raise


def get_user_from_token(token: str) -> dict[str, Any]:
    """Get user information from a JWT token."""
    payload = verify_token(token)
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "viewer"),
        "permissions": payload.get("permissions", []),
    }
