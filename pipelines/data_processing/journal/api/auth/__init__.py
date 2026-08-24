"""
Authentication and authorization module for the API server.

This module provides JWT token validation, role-based access control,
and authentication utilities.
"""

from ai.pipelines.data_processing.journal.api.auth.jwt import (
    create_access_token,
    decode_access_token,
    verify_token,
)
from ai.pipelines.data_processing.journal.api.auth.rbac import (
    check_permission,
    get_user_role,
    require_role,
)

__all__ = [
    "check_permission",
    "create_access_token",
    "decode_access_token",
    "get_user_role",
    "require_role",
    "verify_token",
]
