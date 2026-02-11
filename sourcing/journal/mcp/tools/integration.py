"""
Integration planning tools for MCP Server.

This module provides tools for creating and managing integration plans through the MCP protocol.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.tools.base import MCPTool
from ai.sourcing.journal.models.dataset_models import IntegrationPlan

logger = logging.getLogger(__name__)


class CreateIntegrationPlansTool(MCPTool):
    """Tool for creating integration plans for acquired datasets."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize CreateIntegrationPlansTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="create_integration_plans",
            description="Create integration plans for acquired datasets in a research session. This tool initiates integration planning by analyzing dataset structures and creating transformation specifications for integrating datasets into the training pipeline.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of source IDs to create integration plans for. If not provided, all acquired datasets in the session will be used.",
                    },
                    "target_format": {
                        "type": "string",
                        "enum": ["chatml", "conversation_record"],
                        "description": "Target format for integration (default: chatml)",
                        "default": "chatml",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute create_integration_plans tool.

        Args:
            params: Tool parameters

        Returns:
            Integration plans result with progress updates
        """
        try:
            session_id = params.get("session_id")
            source_ids = params.get("source_ids")
            target_format = params.get("target_format", "chatml")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if target_format not in ["chatml", "conversation_record"]:
                raise ValueError("target_format must be 'chatml' or 'conversation_record'")

            # Initiate integration planning
            result = self.service.initiate_integration(
                session_id=session_id,
                source_ids=source_ids if source_ids else None,
                target_format=target_format,
            )

            # Get integration plans
            plans_list = self.service.get_integration_plans(session_id)

            # Convert plans to dicts
            plans_data = []
            for plan in plans_list:
                plans_data.append(self._plan_to_dict(plan))

            return {
                "session_id": session_id,
                "plans": result.get("plans", []),
                "integration_plans": plans_data,
                "total_plans": len(plans_data),
                "status": "completed",
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error creating integration plans: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to create integration plans: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _plan_to_dict(self, plan: IntegrationPlan) -> Dict[str, Any]:
        """Convert IntegrationPlan to dictionary."""
        return {
            "plan_id": f"plan_{plan.source_id}",
            "source_id": plan.source_id,
            "dataset_format": plan.dataset_format,
            "schema_mapping": plan.schema_mapping,
            "required_transformations": plan.required_transformations,
            "preprocessing_steps": plan.preprocessing_steps,
            "complexity": plan.complexity,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "dependencies": plan.dependencies,
            "integration_priority": plan.integration_priority,
            "created_date": plan.created_date.isoformat(),
        }


class GetIntegrationPlansTool(MCPTool):
    """Tool for getting all integration plans for a session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetIntegrationPlansTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_integration_plans",
            description="Get all integration plans for a research session with optional filtering and sorting",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters for integration plans",
                        "properties": {
                            "complexity": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "dataset_format": {
                                "type": "string",
                                "description": "Filter by dataset format (e.g., 'csv', 'json', 'parquet')",
                            },
                            "min_estimated_effort_hours": {
                                "type": "number",
                                "description": "Minimum estimated effort in hours",
                            },
                            "max_estimated_effort_hours": {
                                "type": "number",
                                "description": "Maximum estimated effort in hours",
                            },
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["created_date", "estimated_effort_hours", "complexity", "integration_priority"],
                        "description": "Field to sort by",
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order",
                        "default": "desc",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_integration_plans tool.

        Args:
            params: Tool parameters

        Returns:
            List of integration plans with optional filtering and sorting
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Get integration plans
            plans_list = self.service.get_integration_plans(session_id)

            # Apply filters
            filters = params.get("filters", {})
            if filters:
                plans_list = self._apply_filters(plans_list, filters)

            # Apply sorting
            sort_by = params.get("sort_by")
            sort_order = params.get("sort_order", "desc")
            if sort_by:
                plans_list = self._apply_sorting(plans_list, sort_by, sort_order)

            # Convert plans to dicts
            plans_data = []
            for plan in plans_list:
                plans_data.append(self._plan_to_dict(plan))

            return {
                "session_id": session_id,
                "integration_plans": plans_data,
                "count": len(plans_data),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting integration plans: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get integration plans: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _apply_filters(
        self, plans: List[IntegrationPlan], filters: Dict[str, Any]
    ) -> List[IntegrationPlan]:
        """Apply filters to integration plans list."""
        filtered = plans

        if "complexity" in filters:
            complexity = filters["complexity"]
            filtered = [p for p in filtered if p.complexity == complexity]

        if "dataset_format" in filters:
            dataset_format = filters["dataset_format"]
            filtered = [p for p in filtered if p.dataset_format == dataset_format]

        if "min_estimated_effort_hours" in filters:
            min_hours = filters["min_estimated_effort_hours"]
            filtered = [p for p in filtered if p.estimated_effort_hours >= min_hours]

        if "max_estimated_effort_hours" in filters:
            max_hours = filters["max_estimated_effort_hours"]
            filtered = [p for p in filtered if p.estimated_effort_hours <= max_hours]

        return filtered

    def _apply_sorting(
        self, plans: List[IntegrationPlan], sort_by: str, sort_order: str
    ) -> List[IntegrationPlan]:
        """Apply sorting to integration plans list."""
        reverse = sort_order == "desc"

        if sort_by == "created_date":
            return sorted(plans, key=lambda p: p.created_date, reverse=reverse)
        elif sort_by == "estimated_effort_hours":
            return sorted(plans, key=lambda p: p.estimated_effort_hours, reverse=reverse)
        elif sort_by == "complexity":
            complexity_order = {"low": 1, "medium": 2, "high": 3}
            return sorted(
                plans,
                key=lambda p: complexity_order.get(p.complexity, 0),
                reverse=reverse,
            )
        elif sort_by == "integration_priority":
            return sorted(plans, key=lambda p: p.integration_priority, reverse=reverse)
        else:
            return plans

    def _plan_to_dict(self, plan: IntegrationPlan) -> Dict[str, Any]:
        """Convert IntegrationPlan to dictionary."""
        return {
            "plan_id": f"plan_{plan.source_id}",
            "source_id": plan.source_id,
            "dataset_format": plan.dataset_format,
            "schema_mapping": plan.schema_mapping,
            "required_transformations": plan.required_transformations,
            "preprocessing_steps": plan.preprocessing_steps,
            "complexity": plan.complexity,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "dependencies": plan.dependencies,
            "integration_priority": plan.integration_priority,
            "created_date": plan.created_date.isoformat(),
        }


class GetIntegrationPlanTool(MCPTool):
    """Tool for getting a specific integration plan."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetIntegrationPlanTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_integration_plan",
            description="Get details of a specific integration plan",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "plan_id": {
                        "type": "string",
                        "description": "Integration plan ID (format: plan_{source_id})",
                    },
                },
                "required": ["session_id", "plan_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_integration_plan tool.

        Args:
            params: Tool parameters

        Returns:
            Integration plan details
        """
        try:
            session_id = params.get("session_id")
            plan_id = params.get("plan_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not plan_id:
                raise ValueError("plan_id is required")

            # Get integration plan
            plan = self.service.get_integration_plan(session_id, plan_id)

            # Convert plan to dict
            return self._plan_to_dict(plan)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting integration plan: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get integration plan: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _plan_to_dict(self, plan: IntegrationPlan) -> Dict[str, Any]:
        """Convert IntegrationPlan to dictionary."""
        return {
            "plan_id": f"plan_{plan.source_id}",
            "source_id": plan.source_id,
            "dataset_format": plan.dataset_format,
            "schema_mapping": plan.schema_mapping,
            "required_transformations": plan.required_transformations,
            "preprocessing_steps": plan.preprocessing_steps,
            "complexity": plan.complexity,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "dependencies": plan.dependencies,
            "integration_priority": plan.integration_priority,
            "created_date": plan.created_date.isoformat(),
        }


class GeneratePreprocessingScriptTool(MCPTool):
    """Tool for generating preprocessing scripts from integration plans."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GeneratePreprocessingScriptTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="generate_preprocessing_script",
            description="Generate a Python preprocessing script from an integration plan. The script includes dataset loading, transformation logic, and schema mapping based on the integration plan specifications.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "plan_id": {
                        "type": "string",
                        "description": "Integration plan ID (format: plan_{source_id})",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output path for the generated script. If not provided, a default path will be used.",
                    },
                },
                "required": ["session_id", "plan_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generate_preprocessing_script tool.

        Args:
            params: Tool parameters

        Returns:
            Script generation result with script content and path
        """
        try:
            session_id = params.get("session_id")
            plan_id = params.get("plan_id")
            output_path = params.get("output_path")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not plan_id:
                raise ValueError("plan_id is required")

            # Get integration plan
            plan = self.service.get_integration_plan(session_id, plan_id)

            # Get integration engine from orchestrator
            orchestrator = self.service.orchestrator
            if not orchestrator.integration_engine:
                raise ValueError("No integration engine configured")

            # Generate default output path if not provided
            if not output_path:
                import os
                from pathlib import Path

                scripts_dir = Path("data/integration_scripts")
                scripts_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(scripts_dir / f"preprocess_{plan.source_id}.py")

            # Generate preprocessing script
            script_path = orchestrator.integration_engine.generate_preprocessing_script(
                plan, output_path
            )

            # Read script content
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()

            return {
                "session_id": session_id,
                "plan_id": plan_id,
                "source_id": plan.source_id,
                "script_path": script_path,
                "script_content": script_content,
                "complexity": plan.complexity,
                "estimated_effort_hours": plan.estimated_effort_hours,
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error generating preprocessing script: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to generate preprocessing script: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

