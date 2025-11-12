"""
Authentication and authorization module for the API server.

This module provides JWT token validation, role-based access control,
and authentication utilities.
"""

from ai.journal_dataset_research.api.auth.jwt import (
    create_access_token,
    decode_access_token,
    verify_token,
)
from ai.journal_dataset_research.api.auth.rbac import (
    check_permission,
    get_user_role,
    require_role,
)

__all__ = [
    "create_access_token",
    "decode_access_token",
    "verify_token",
    "check_permission",
    "get_user_role",
    "require_role",
]

