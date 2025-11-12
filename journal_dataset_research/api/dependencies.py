"""
FastAPI dependencies for authentication and authorization.

This module provides dependency injection for authentication and authorization.
"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai.journal_dataset_research.api.auth.jwt import get_user_from_token
from ai.journal_dataset_research.api.auth.rbac import (
    check_permission,
    get_user_role,
    require_permission,
    require_role,
)
from ai.journal_dataset_research.api.services.command_handler_service import (
    CommandHandlerService,
)

# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Get current user from JWT token."""
    # Check if authentication is enabled
    from ai.journal_dataset_research.api.config import get_settings

    settings = get_settings()
    if not settings.auth_enabled:
        # Return default user for development
        return {
            "user_id": "dev-user",
            "email": "dev@example.com",
            "role": "admin",
            "permissions": ["*"],
        }

    # Check if user is already in request state (from middleware)
    if hasattr(request.state, "user"):
        return request.state.user

    # Try to get token from authorization header
    if not credentials:
        # Check for token in request state (set by middleware)
        if hasattr(request.state, "token"):
            token = request.state.token
        else:
            return None
    else:
        token = credentials.credentials

    # Verify token and get user
    try:
        user = get_user_from_token(token)
        request.state.user = user
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_optional_user(
    current_user: Optional[dict] = Depends(get_current_user),
) -> Optional[dict]:
    """Get optional current user (doesn't raise error if not authenticated)."""
    return current_user


def require_authentication(
    current_user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Require authentication, raise error if not authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_role_dependency(required_role: str):
    """Create a dependency that requires a specific role."""

    async def role_checker(
        current_user: dict = Depends(require_authentication),
    ) -> dict:
        require_role(current_user, required_role)
        return current_user

    return role_checker


def require_permission_dependency(permission: str):
    """Create a dependency that requires a specific permission."""

    async def permission_checker(
        current_user: dict = Depends(require_authentication),
    ) -> dict:
        require_permission(current_user, permission)
        return current_user

    return permission_checker


def get_command_handler_service() -> CommandHandlerService:
    """Get CommandHandlerService instance."""
    return CommandHandlerService()
