"""
Authentication handlers for MCP Server.

This module provides authentication handlers for API key and JWT token authentication.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jwt as pyjwt
from jwt.exceptions import DecodeError, ExpiredSignatureError, InvalidTokenError

from ai.journal_dataset_research.mcp.config import AuthConfig
from ai.journal_dataset_research.mcp.protocol import MCPError, MCPErrorCode, MCPRequest

logger = logging.getLogger(__name__)


class AuthenticationHandler(ABC):
    """Base class for authentication handlers."""

    def __init__(self, config: AuthConfig) -> None:
        """
        Initialize authentication handler.

        Args:
            config: Authentication configuration
        """
        self.config = config

    @abstractmethod
    async def authenticate(self, request: MCPRequest) -> Dict[str, Any]:
        """
        Authenticate a request and return user information.

        Args:
            request: MCP request to authenticate

        Returns:
            User information dictionary with keys: user_id, email, role, permissions

        Raises:
            MCPError: If authentication fails
        """
        pass

    def extract_auth_header(self, request: MCPRequest) -> Optional[str]:
        """
        Extract authentication header from request.

        Args:
            request: MCP request

        Returns:
            Authentication header value or None
        """
        # MCP requests may include auth in params or metadata
        # Check params first (common pattern)
        if request.params and isinstance(request.params, dict):
            # Check for Authorization header in params
            headers = request.params.get("headers", {})
            if isinstance(headers, dict):
                auth_header = headers.get("Authorization") or headers.get("authorization")
                if auth_header:
                    return auth_header

            # Check for direct auth token
            auth_token = request.params.get("auth_token") or request.params.get("token")
            if auth_token:
                return auth_token

        # Check for auth in request metadata (if available)
        if hasattr(request, "metadata") and isinstance(request.metadata, dict):
            auth_header = request.metadata.get("Authorization") or request.metadata.get(
                "authorization"
            )
            if auth_header:
                return auth_header

        return None


class APIKeyAuth(AuthenticationHandler):
    """API key authentication handler."""

    async def authenticate(self, request: MCPRequest) -> Dict[str, Any]:
        """
        Authenticate request using API key.

        Args:
            request: MCP request to authenticate

        Returns:
            User information dictionary

        Raises:
            MCPError: If authentication fails
        """
        if not self.config.api_key_required:
            # API key not required, return default user
            return {
                "user_id": "api-key-user",
                "email": "api@example.com",
                "role": "admin",
                "permissions": ["*"],
            }

        # Extract API key from request
        auth_header = self.extract_auth_header(request)
        if not auth_header:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "API key required but not provided",
            )

        # Extract API key from header (format: "Bearer <key>" or just "<key>")
        api_key = auth_header
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()
        elif auth_header.startswith("ApiKey "):
            api_key = auth_header[7:].strip()

        # Validate API key
        if not api_key:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "API key is empty",
            )

        if api_key not in self.config.allowed_api_keys:
            logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "Invalid API key",
            )

        # Return user info for API key
        # API keys typically have admin access, but this can be configured
        return {
            "user_id": f"api-key-{api_key[:8]}",
            "email": "api@example.com",
            "role": "admin",  # Default role for API keys
            "permissions": ["*"],  # Full permissions for API keys
        }


class JWTAuth(AuthenticationHandler):
    """JWT token authentication handler."""

    async def authenticate(self, request: MCPRequest) -> Dict[str, Any]:
        """
        Authenticate request using JWT token.

        Args:
            request: MCP request to authenticate

        Returns:
            User information dictionary

        Raises:
            MCPError: If authentication fails
        """
        # Extract JWT token from request
        auth_header = self.extract_auth_header(request)
        if not auth_header:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "JWT token required but not provided",
            )

        # Extract token from header (format: "Bearer <token>")
        token = auth_header
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif not auth_header.startswith("Bearer "):
            # Try to use the header value directly as token
            token = auth_header.strip()

        if not token:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "JWT token is empty",
            )

        # Decode and verify JWT token
        try:
            payload = pyjwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )
        except ExpiredSignatureError:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "JWT token has expired",
            )
        except DecodeError:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "Invalid JWT token format",
            )
        except InvalidTokenError as e:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                f"Invalid JWT token: {str(e)}",
            )
        except Exception as e:
            logger.exception("Unexpected error during JWT authentication")
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                f"JWT authentication failed: {str(e)}",
            )

        # Extract user information from payload
        user_id = payload.get("sub") or payload.get("user_id")
        email = payload.get("email")
        role = payload.get("role", "viewer")
        permissions = payload.get("permissions", [])

        if not user_id:
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "JWT token missing user identifier (sub or user_id)",
            )

        return {
            "user_id": str(user_id),
            "email": email,
            "role": role,
            "permissions": permissions,
        }


class CompositeAuth(AuthenticationHandler):
    """
    Composite authentication handler that tries multiple authentication methods.

    Tries API key first, then JWT token.
    """

    def __init__(self, config: AuthConfig) -> None:
        """
        Initialize composite authentication handler.

        Args:
            config: Authentication configuration
        """
        super().__init__(config)
        self.api_key_auth = APIKeyAuth(config)
        self.jwt_auth = JWTAuth(config)

    async def authenticate(self, request: MCPRequest) -> Dict[str, Any]:
        """
        Authenticate request using multiple methods.

        Tries API key authentication first, then JWT token authentication.

        Args:
            request: MCP request to authenticate

        Returns:
            User information dictionary

        Raises:
            MCPError: If all authentication methods fail
        """
        # Try API key authentication first
        if self.config.api_key_required:
            try:
                return await self.api_key_auth.authenticate(request)
            except MCPError:
                # API key auth failed, try JWT
                pass

        # Try JWT authentication
        try:
            return await self.jwt_auth.authenticate(request)
        except MCPError:
            # Both methods failed
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                "Authentication failed: API key or JWT token required",
            )


def create_auth_handler(config: AuthConfig) -> Optional[AuthenticationHandler]:
    """
    Create appropriate authentication handler based on configuration.

    Args:
        config: Authentication configuration

    Returns:
        Authentication handler instance or None if authentication is disabled
    """
    if not config.enabled:
        return None

    # If both API key and JWT are configured, use composite handler
    if config.api_key_required and config.jwt_secret:
        return CompositeAuth(config)

    # If only API key is required
    if config.api_key_required:
        return APIKeyAuth(config)

    # If only JWT is configured
    if config.jwt_secret and config.jwt_secret != "change-me-in-production":
        return JWTAuth(config)

    # Default: no authentication
    logger.warning("Authentication enabled but no method configured")
    return None

