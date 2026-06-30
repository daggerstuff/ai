"""
Agent Management API Routes for MCP Server.

This module provides REST API endpoints for agent registration, discovery,
and health monitoring.
"""

import asyncio
import logging

from flask import Blueprint, current_app, jsonify, request

# Internal imports
from ai.api.mcp_server.auth.middleware import require_mcp_auth, require_mcp_role
from ai.api.mcp_server.core.agent_manager import AgentDiscoveryCriteria, AgentRegistrationData

logger = logging.getLogger(__name__)
agents_bp = Blueprint("agents", __name__)


@agents_bp.route("/register", methods=["POST"])
@require_mcp_auth
@require_mcp_role(["admin", "system"])
def register_agent():
    """Register a new agent with the MCP server."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Missing registration data"}), 400

        registration_data = AgentRegistrationData(
            name=data.get("name"),
            type=data.get("type"),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )

        # Access agent manager from current_app (initialized in app.py)
        # Note: app.py needs to be updated to attach agent_manager to app
        agent_manager = getattr(current_app, "agent_manager", None)
        if not agent_manager:
            return jsonify({"success": False, "error": "Agent manager not initialized"}), 500

        agent = asyncio_run(agent_manager.register_agent(registration_data))

        return jsonify({"success": True, "data": agent.to_dict()}), 201

    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@agents_bp.route("/discover", methods=["GET"])
@require_mcp_auth
def discover_agents():
    """Discover agents based on capability criteria."""
    try:
        criteria = AgentDiscoveryCriteria.from_request(request.args)

        agent_manager = getattr(current_app, "agent_manager", None)
        if not agent_manager:
            return jsonify({"success": False, "error": "Agent manager not initialized"}), 500

        agents = asyncio_run(agent_manager.discover_agents(criteria))

        return jsonify({"success": True, "data": [a.to_dict() for a in agents]}), 200

    except Exception as e:
        logger.error(f"Error discovering agents: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@agents_bp.route("/<agent_id>/health", methods=["GET"])
@require_mcp_auth
def get_agent_health(agent_id):
    """Get health status for a specific agent."""
    try:
        agent_manager = getattr(current_app, "agent_manager", None)
        if not agent_manager:
            return jsonify({"success": False, "error": "Agent manager not initialized"}), 500

        health_status = asyncio_run(agent_manager.check_agent_health(agent_id))

        if not health_status:
            return jsonify({"success": False, "error": "Agent not found"}), 404

        return jsonify({"success": True, "data": health_status}), 200

    except Exception as e:
        logger.error(f"Error checking agent health: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


def asyncio_run(coro):
    """Helper to run async code in sync Flask route."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
