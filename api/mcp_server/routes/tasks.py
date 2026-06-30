"""
Task Management API Routes for MCP Server.

This module provides REST API endpoints for task creation, assignment,
and status monitoring within the TechDeck-Python pipeline.
"""

import logging

from flask import Blueprint, g, jsonify, request

# Internal imports
from ai.api.mcp_server.auth.middleware import require_mcp_auth, require_mcp_role
from ai.api.mcp_server.core.task_orchestrator import TaskCreationData, TaskStatus

from .agents import asyncio_run

logger = logging.getLogger(__name__)
tasks_bp = Blueprint("tasks", __name__)


@tasks_bp.route("", methods=["POST"])
@require_mcp_auth
@require_mcp_role(["admin", "pipeline_operator"])
def create_task():
    """Create a new task and enqueue it for processing."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Missing task data"}), 400

        task_data = TaskCreationData(
            pipeline_id=data.get("pipeline_id"),
            stage=data.get("stage"),
            parameters=data.get("parameters", {}),
            required_capabilities=data.get("required_capabilities", []),
            priority=data.get("priority", 1),
        )

        task_orchestrator = g.task_orchestrator
        if not task_orchestrator:
            return jsonify({"success": False, "error": "Task orchestrator not initialized"}), 500

        task = asyncio_run(task_orchestrator.create_task(task_data))

        return jsonify({"success": True, "data": task.to_dict()}), 201

    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@tasks_bp.route("/<task_id>", methods=["GET"])
@require_mcp_auth
def get_task_status(task_id):
    """Get the current progress and status of a task."""
    try:
        task_orchestrator = g.task_orchestrator
        if not task_orchestrator:
            return jsonify({"success": False, "error": "Task orchestrator not initialized"}), 500

        # Direct lookup in MongoDB via task_orchestrator's internal queue/client
        task_data = asyncio_run(task_orchestrator.mongodb.find_one("tasks", {"id": task_id}))

        if not task_data:
            return jsonify({"success": False, "error": "Task not found"}), 404

        return jsonify({"success": True, "data": task_data}), 200

    except Exception as e:
        logger.error(f"Error fetching task status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@tasks_bp.route("/<task_id>/progress", methods=["PATCH"])
@require_mcp_auth
@require_mcp_role(["admin", "agent"])
def update_progress(task_id):
    """Update task progress, status, or results (typically called by agents)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Missing update data"}), 400

        progress = data.get("progress", 0.0)
        status_val = data.get("status")
        result = data.get("result")
        error = data.get("error")

        try:
            status = TaskStatus(status_val) if status_val else TaskStatus.RUNNING
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid status: {status_val}"}), 400

        task_orchestrator = g.task_orchestrator
        if not task_orchestrator:
            return jsonify({"success": False, "error": "Task orchestrator not initialized"}), 500

        success = asyncio_run(
            task_orchestrator.update_task_progress(
                task_id=task_id,
                progress=progress,
                status=status,
                result=result,
                error=error,
            )
        )

        if not success:
            return jsonify({"success": False, "error": "Failed to update task"}), 404

        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(f"Error updating task progress: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
