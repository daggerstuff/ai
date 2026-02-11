"""
JWT token utilities for authentication.

This module provides JWT token creation, validation, and decoding.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt as pyjwt
from jwt.exceptions import DecodeError, ExpiredSignatureError, InvalidTokenError

from ai.sourcing.journal.api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.jwt_expiration_minutes
        )
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    encoded_jwt = pyjwt.encode(
        to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode a JWT access token."""
    try:
        payload = pyjwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except ExpiredSignatureError:
        raise ValueError("Token has expired")
    except DecodeError:
        raise ValueError("Invalid token format")
    except InvalidTokenError as e:
        raise ValueError(f"Invalid token: {str(e)}")


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode a JWT token."""
    try:
        payload = decode_access_token(token)
        return payload
    except ValueError as e:
        logger.warning(f"Token verification failed: {e}")
        raise


def get_user_from_token(token: str) -> Dict[str, Any]:
    """Get user information from a JWT token."""
    payload = verify_token(token)
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "viewer"),
        "permissions": payload.get("permissions", []),
    }

