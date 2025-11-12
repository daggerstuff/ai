"""
Authorization handlers for MCP Server.

This module provides authorization handlers for role-based access control (RBAC).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.mcp.protocol import MCPError, MCPErrorCode

logger = logging.getLogger(__name__)

# Role definitions (matching API RBAC)
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

# Tool permission mapping
TOOL_PERMISSIONS = {
    # Session management tools
    "create_session": "sessions:create",
    "list_sessions": "sessions:read",
    "get_session": "sessions:read",
    "update_session": "sessions:update",
    "delete_session": "sessions:delete",
    # Source discovery tools
    "discover_sources": "discovery:initiate",
    "get_sources": "discovery:read",
    "get_source": "discovery:read",
    "filter_sources": "discovery:read",
    # Evaluation tools
    "evaluate_sources": "evaluation:create",
    "get_evaluations": "evaluation:read",
    "get_evaluation": "evaluation:read",
    "update_evaluation": "evaluation:update",
    # Acquisition tools
    "acquire_datasets": "acquisition:create",
    "get_acquisitions": "acquisition:read",
    "get_acquisition": "acquisition:read",
    "update_acquisition": "acquisition:update",
    # Integration tools
    "create_integration_plans": "integration:create",
    "get_integration_plans": "integration:read",
    "get_integration_plan": "integration:read",
    "generate_preprocessing_script": "integration:read",
    # Report tools
    "generate_report": "reports:generate",
    "get_report": "reports:read",
    "list_reports": "reports:read",
}

# Resource permission mapping
RESOURCE_PERMISSIONS = {
    "progress/metrics": "sessions:read",
    "progress/history": "sessions:read",
    "session/state": "sessions:read",
    "session/metrics": "sessions:read",
}


class AuthorizationHandler(ABC):
    """Base class for authorization handlers."""

    @abstractmethod
    async def authorize(
        self, user: Dict[str, Any], resource: str, action: str
    ) -> bool:
        """
        Check if user is authorized to perform action on resource.

        Args:
            user: User information dictionary
            resource: Resource identifier (e.g., tool name, resource URI)
            action: Action to perform (e.g., "execute", "read")

        Returns:
            True if authorized, False otherwise
        """
        pass

    @abstractmethod
    async def require_authorization(
        self, user: Dict[str, Any], resource: str, action: str
    ) -> None:
        """
        Require authorization, raise exception if not authorized.

        Args:
            user: User information dictionary
            resource: Resource identifier
            action: Action to perform

        Raises:
            MCPError: If authorization fails
        """
        pass


class RBAC(AuthorizationHandler):
    """Role-based access control handler."""

    def __init__(self) -> None:
        """Initialize RBAC handler."""
        self.roles = ROLES
        self.tool_permissions = TOOL_PERMISSIONS
        self.resource_permissions = RESOURCE_PERMISSIONS

    def get_user_role(self, user: Optional[Dict[str, Any]]) -> str:
        """
        Get user role from user dictionary.

        Args:
            user: User information dictionary

        Returns:
            User role (defaults to "viewer" if not provided)
        """
        if not user:
            return "viewer"
        return user.get("role", "viewer")

    def get_role_permissions(self, role: str) -> List[str]:
        """
        Get permissions for a role.

        Args:
            role: Role name

        Returns:
            List of permissions
        """
        role_config = self.roles.get(role, {})
        return role_config.get("permissions", [])

    def check_permission(self, user: Optional[Dict[str, Any]], permission: str) -> bool:
        """
        Check if user has a specific permission.

        Args:
            user: User information dictionary
            permission: Permission to check (e.g., "sessions:read")

        Returns:
            True if user has permission, False otherwise
        """
        if not user:
            return False

        # Check user's direct permissions first
        user_permissions = user.get("permissions", [])
        if "*" in user_permissions:
            return True
        if permission in user_permissions:
            return True

        # Check role permissions
        role = self.get_user_role(user)
        role_permissions = self.get_role_permissions(role)

        # Check for wildcard permission
        if "*" in role_permissions:
            return True

        # Check for exact permission match
        if permission in role_permissions:
            return True

        # Check for wildcard match (e.g., "sessions:*" matches "sessions:read")
        permission_parts = permission.split(":")
        if len(permission_parts) == 2:
            resource, action = permission_parts
            wildcard_permission = f"{resource}:*"
            if wildcard_permission in role_permissions:
                return True

        return False

    def get_tool_permission(self, tool_name: str) -> Optional[str]:
        """
        Get required permission for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Required permission or None if not found
        """
        return self.tool_permissions.get(tool_name)

    def get_resource_permission(self, resource_uri: str) -> Optional[str]:
        """
        Get required permission for a resource.

        Args:
            resource_uri: Resource URI

        Returns:
            Required permission or None if not found
        """
        # Extract resource type from URI (e.g., "research://progress/metrics/123" -> "progress/metrics")
        if "://" in resource_uri:
            parts = resource_uri.split("://", 1)
            if len(parts) == 2:
                path = parts[1].split("/")
                if len(path) >= 2:
                    resource_type = f"{path[0]}/{path[1]}"
                    return self.resource_permissions.get(resource_type)

        return None

    async def authorize(
        self, user: Dict[str, Any], resource: str, action: str
    ) -> bool:
        """
        Check if user is authorized to perform action on resource.

        Args:
            user: User information dictionary
            resource: Resource identifier (tool name or resource URI)
            action: Action to perform (e.g., "execute", "read")

        Returns:
            True if authorized, False otherwise
        """
        # Determine required permission based on resource type
        permission: Optional[str] = None

        # Check if resource is a tool
        tool_permission = self.get_tool_permission(resource)
        if tool_permission:
            permission = tool_permission
        else:
            # Check if resource is a resource URI
            resource_permission = self.get_resource_permission(resource)
            if resource_permission:
                permission = resource_permission
            else:
                # Default permission based on action
                if action == "execute":
                    # For unknown tools, require admin or wildcard permission
                    return self.check_permission(user, "*")
                elif action == "read":
                    # For unknown resources, require read permission
                    return self.check_permission(user, "sessions:read")

        if not permission:
            # No permission mapping found, deny by default
            logger.warning(
                f"No permission mapping for resource: {resource}, action: {action}"
            )
            return False

        return self.check_permission(user, permission)

    async def require_authorization(
        self, user: Dict[str, Any], resource: str, action: str
    ) -> None:
        """
        Require authorization, raise exception if not authorized.

        Args:
            user: User information dictionary
            resource: Resource identifier
            action: Action to perform

        Raises:
            MCPError: If authorization fails
        """
        if not user:
            raise MCPError(
                MCPErrorCode.AUTHORIZATION_ERROR,
                "Authentication required for authorization",
            )

        is_authorized = await self.authorize(user, resource, action)
        if not is_authorized:
            role = self.get_user_role(user)
            raise MCPError(
                MCPErrorCode.AUTHORIZATION_ERROR,
                f"User with role '{role}' is not authorized to perform '{action}' on '{resource}'",
            )


def create_authorization_handler() -> AuthorizationHandler:
    """
    Create authorization handler.

    Returns:
        Authorization handler instance (currently only RBAC is supported)
    """
    return RBAC()

