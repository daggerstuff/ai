"""
Pipeline Management API Routes for MCP Server.

This module provides REST API endpoints for pipeline execution,
orchestration, and monitoring.
"""

import logging

from flask import Blueprint, g, jsonify, request

# Internal imports
from ai.api.mcp_server.auth.middleware import require_mcp_auth, require_mcp_role
from ai.api.mcp_server.core.pipeline_integration import PipelineConfig

from .agents import asyncio_run

logger = logging.getLogger(__name__)
pipeline_bp = Blueprint("pipeline", __name__)


@pipeline_bp.route("/agent-execute", methods=["POST"])
@require_mcp_auth
@require_mcp_role(["admin", "pipeline_operator"])
def execute_pipeline():
    """Initiate a pipeline execution with agent orchestration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Missing pipeline data"}), 400

        pipeline_config = PipelineConfig(
            pipeline_id=data.get("pipeline_id"), stages=data.get("stages", []), metadata=data.get("metadata", {})
        )

        pipeline_manager = g.pipeline_manager
        if not pipeline_manager:
            return jsonify({"success": False, "error": "Pipeline manager not initialized"}), 500

        result = asyncio_run(pipeline_manager.execute_pipeline_with_agents(pipeline_config))

        return jsonify(
            {
                "success": True,
                "data": {"execution_id": result.execution_id, "status": result.status, "data": result.data},
            }
        ), 201

    except Exception as e:
        logger.error(f"Error initiating pipeline execution: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@pipeline_bp.route("/monitor/<execution_id>", methods=["GET"])
@require_mcp_auth
def monitor_pipeline(execution_id):
    """Monitor progress for an active pipeline execution."""
    try:
        pipeline_manager = g.pipeline_manager
        if not pipeline_manager:
            return jsonify({"success": False, "error": "Pipeline manager not initialized"}), 500

        progress_data = asyncio_run(pipeline_manager.monitor_pipeline_progress(execution_id))

        return jsonify({"success": True, "data": progress_data}), 200

    except Exception as e:
        logger.error(f"Error monitoring pipeline: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
