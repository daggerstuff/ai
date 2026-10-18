"""
Authentication and Authorization for MCP Server.

This module provides authentication and authorization handlers.
"""

from ai.sourcing.journal.mcp.auth.authentication import (
    APIKeyAuth,
    AuthenticationHandler,
    CompositeAuth,
    JWTAuth,
    create_auth_handler,
)
from ai.sourcing.journal.mcp.auth.authorization import (
    RBAC,
    AuthorizationHandler,
    create_authorization_handler,
)

__all__ = [
    "RBAC",
    "APIKeyAuth",
    "AuthenticationHandler",
    "AuthorizationHandler",
    "CompositeAuth",
    "JWTAuth",
    "create_auth_handler",
    "create_authorization_handler",
]
