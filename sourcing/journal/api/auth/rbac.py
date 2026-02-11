"""
Role-based access control (RBAC) utilities.

This module provides role-based permission checking and access control.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Role definitions
ROLES = {
    "admin": {
        "permissions": ["*"],  # All permissions
        "description": "Full access to all operations",
    },
    "research_coordinator": {
        "permissions": [
            "sessions:create",
            "sessions:read",
            "sessions:update",
            "sessions:delete",
            "discovery:initiate",
            "discovery:read",
            "evaluation:read",
            "evaluation:update",
            "acquisition:read",
            "acquisition:approve",
            "integration:read",
            "reports:generate",
            "reports:read",
        ],
        "description": "Can manage research sessions and operations",
    },
    "data_scientist": {
        "permissions": [
            "sessions:read",
            "discovery:read",
            "evaluation:read",
            "evaluation:update",
            "acquisition:read",
            "integration:read",
            "reports:read",
        ],
        "description": "Can view and evaluate datasets",
    },
    "viewer": {
        "permissions": [
            "sessions:read",
            "discovery:read",
            "evaluation:read",
            "acquisition:read",
            "integration:read",
            "reports:read",
        ],
        "description": "Read-only access",
    },
}


def get_user_role(user: Optional[Dict[str, Any]]) -> str:
    """Get user role from user dictionary."""
    if not user:
        return "viewer"
    return user.get("role", "viewer")


def get_role_permissions(role: str) -> List[str]:
    """Get permissions for a role."""
    role_config = ROLES.get(role, {})
    return role_config.get("permissions", [])


def check_permission(user: Optional[Dict[str, Any]], permission: str) -> bool:
    """Check if user has a specific permission."""
    if not user:
        return False

    role = get_user_role(user)
    permissions = get_role_permissions(role)

    # Check for wildcard permission
    if "*" in permissions:
        return True

    # Check for exact permission match
    if permission in permissions:
        return True

    # Check for wildcard match (e.g., "sessions:*" matches "sessions:read")
    permission_parts = permission.split(":")
    if len(permission_parts) == 2:
        resource, action = permission_parts
        wildcard_permission = f"{resource}:*"
        if wildcard_permission in permissions:
            return True

    return False


def require_role(user: Optional[Dict[str, Any]], required_role: str) -> None:
    """Require a specific role, raise HTTPException if not met."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    role = get_user_role(user)
    role_hierarchy = ["viewer", "data_scientist", "research_coordinator", "admin"]

    try:
        user_role_index = role_hierarchy.index(role)
        required_role_index = role_hierarchy.index(required_role)
        if user_role_index < required_role_index:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required, but user has role '{role}'",
            )
    except ValueError:
        # Role not in hierarchy, deny access
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Invalid role: {role}",
        )


def require_permission(user: Optional[Dict[str, Any]], permission: str) -> None:
    """Require a specific permission, raise HTTPException if not met."""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not check_permission(user, permission):
        role = get_user_role(user)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required, but user role '{role}' does not have it",
        )

