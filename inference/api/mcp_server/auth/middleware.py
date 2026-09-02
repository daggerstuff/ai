import logging
from functools import wraps
from typing import Any

import jwt
from flask import Request, current_app, g, jsonify, request


class AuthenticationError(Exception):
    """Raised when authentication fails."""


class PermissionDeniedError(Exception):
    """Raised when an agent lacks required permissions."""


class RateLimitExceededError(Exception):
    """Raised when an agent exceeds its rate limit."""


class AgentContext:
    """Context information for an authenticated agent."""

    def __init__(self, agent_id: str, roles: list[str], clearance_level: int = 0):
        self.agent_id = agent_id
        self.roles = roles
        self.clearance_level = clearance_level


class AgentTokenValidator:
    """Validates agent tokens (JWT)."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def validate_token(self, token: str) -> AgentContext:
        """
        Validate JWT token and return AgentContext.

        In a real scenario, this would check against a database or cache.
        For now, we decode the JWT and extract claims.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return AgentContext(
                agent_id=str(payload.get("sub", "")),
                roles=payload.get("roles", []),
                clearance_level=payload.get("clearance", 0),
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired") from None
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}") from e


class RoleBasedAccessControl:
    """Manages role-based access control for agents."""

    def has_permission(self, context: AgentContext, required_permission: str) -> bool:
        """
        Check if agent has the required permission based on their roles.

        This is a simplified implementation.
        """
        # Map roles to permissions
        role_permissions = {
            "admin": ["*"],
            "pipeline_operator": [
                "pipeline.execute",
                "task.create",
                "task.assign",
                "agent.discover",
            ],
            "agent": ["agent.basic", "task.update"],
        }

        for role in context.roles:
            permissions = role_permissions.get(role, [])
            if "*" in permissions or required_permission in permissions:
                return True
        return False


class MCPAuthMiddleware:
    """
    Enhanced authentication middleware for MCP server.

    Acts as a WSGI middleware or can be used within Flask hooks.
    Based on ARCHITECTURE.md.
    """

    def __init__(self, app, config: Any):
        self.app = app
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.agent_token_validator = AgentTokenValidator(
            secret_key=config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM
        )
        self.role_based_access = RoleBasedAccessControl()

    def __call__(self, environ, start_response):
        """WSGI middleware entry point."""
        # For simplicity in this implementation, we delegate to Flask-level hooks
        # but provide the class structure as requested.
        return self.app(environ, start_response)

    async def authenticate_agent(self, request: Request) -> AgentContext:
        """Authenticate agent requests with enhanced validation."""
        auth_header = request.headers.get("Authorization", "")

        # Support both "Agent <token>" and "Bearer <token>" for compatibility
        token = ""
        if auth_header.startswith(("Agent ", "Bearer ")):
            token = auth_header.split(" ")[1]
        else:
            raise AuthenticationError("Missing or invalid agent authorization header")

        # Validate agent token
        agent_context = await self.agent_token_validator.validate_token(token)

        # Check agent permissions
        required_permission = self.get_required_permission(request.path, request.method)
        has_perm = self.role_based_access.has_permission(agent_context, required_permission)
        if not has_perm:
            raise PermissionDeniedError("Agent lacks required permissions")

        # Rate limiting check would go here

        return agent_context

    def get_required_permission(self, path: str, method: str) -> str:
        """Map API endpoints to required permissions."""
        permission_map = {
            ("/api/v1/agents/register", "POST"): "agent.register",
            ("/api/v1/agents/discover", "GET"): "agent.discover",
            ("/api/v1/tasks", "POST"): "task.create",
            ("/api/v1/tasks", "GET"): "task.view",
            ("/api/v1/pipeline/agent-execute", "POST"): "pipeline.execute",
        }

        # Handle dynamic paths (e.g. /api/v1/tasks/<id>/assign)
        if "/assign" in path:
            return "task.assign"

        for (endpoint, http_method), permission in permission_map.items():
            if path.startswith(endpoint) and method == http_method:
                return permission

        return "agent.basic"  # Default permission


def require_mcp_auth(f):
    """Decorator to require MCP authentication for a route."""

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_middleware = getattr(current_app, "auth_middleware", None)
        if not auth_middleware:
            return jsonify({"success": False, "error": "Auth middleware not initialized"}), 500

        try:
            import asyncio

            # Note: authenticate_agent is async in MCPAuthMiddleware
            # We use asyncio.run to execute it synchronously here

            context = asyncio.run(auth_middleware.authenticate_agent(request))

            g.agent_context = context
            return f(*args, **kwargs)
        except AuthenticationError as e:
            return jsonify({"success": False, "error": str(e)}), 401
        except PermissionDeniedError as e:
            return jsonify({"success": False, "error": str(e)}), 403
        except Exception as e:
            return jsonify({"success": False, "error": f"Internal auth error: {e}"}), 500

    return decorated


def require_mcp_role(roles: list[str]):
    """Decorator to require specific MCP roles for a route."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, "agent_context"):
                return jsonify({"success": False, "error": "Authentication required"}), 401

            context = g.agent_context
            # Check if agent has any of the required roles
            has_role = any(role in context.roles for role in roles)
            if not has_role and "admin" not in context.roles:
                return jsonify({"success": False, "error": f"Requires roles: {roles}"}), 403

            return f(*args, **kwargs)

        return decorated

    return decorator
