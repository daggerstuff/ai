"""
Dataset evaluation tools for MCP Server.

This module provides tools for evaluating dataset sources through the MCP protocol.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.tools.base import MCPTool
from ai.sourcing.journal.models.dataset_models import DatasetEvaluation

logger = logging.getLogger(__name__)


class EvaluateSourcesTool(MCPTool):
    """Tool for evaluating dataset sources."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize EvaluateSourcesTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="evaluate_sources",
            description="Evaluate dataset sources for a research session. This tool initiates evaluation of sources using the evaluation engine, which assesses therapeutic relevance, data structure quality, training integration, and ethical accessibility.",
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
                        "description": "Optional list of source IDs to evaluate. If not provided, all sources in the session will be evaluated.",
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluate_sources tool.

        Args:
            params: Tool parameters

        Returns:
            Evaluation result with progress updates
        """
        try:
            session_id = params.get("session_id")
            source_ids = params.get("source_ids")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")

            # Initiate evaluation
            result = self.service.initiate_evaluation(
                session_id=session_id,
                source_ids=source_ids if source_ids else None,
            )

            # Result already contains evaluations as dicts
            return {
                "session_id": session_id,
                "evaluations": result.get("evaluations", []),
                "total_evaluated": len(result.get("evaluations", [])),
                "status": "completed",
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error evaluating sources: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to evaluate sources: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e



class GetEvaluationsTool(MCPTool):
    """Tool for getting all evaluations for a session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetEvaluationsTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_evaluations",
            description="Get all evaluations for a research session with optional filtering and sorting",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters for evaluations",
                        "properties": {
                            "priority_tier": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "min_overall_score": {
                                "type": "number",
                                "description": "Minimum overall score (0.0-10.0)",
                            },
                            "max_overall_score": {
                                "type": "number",
                                "description": "Maximum overall score (0.0-10.0)",
                            },
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["overall_score", "evaluation_date", "therapeutic_relevance"],
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
        Execute get_evaluations tool.

        Args:
            params: Tool parameters

        Returns:
            List of evaluations with optional filtering and sorting
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Get evaluations
            evaluations_list = self.service.get_evaluations(session_id)

            # Apply filters
            filters = params.get("filters", {})
            if filters:
                evaluations_list = self._apply_filters(evaluations_list, filters)

            # Apply sorting
            sort_by = params.get("sort_by")
            sort_order = params.get("sort_order", "desc")
            if sort_by:
                evaluations_list = self._apply_sorting(evaluations_list, sort_by, sort_order)

            # Convert evaluations to dicts
            evaluations_data = []
            for evaluation in evaluations_list:
                evaluations_data.append(self._evaluation_to_dict(evaluation))

            return {
                "session_id": session_id,
                "evaluations": evaluations_data,
                "count": len(evaluations_data),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting evaluations: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get evaluations: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _apply_filters(
        self, evaluations: List[DatasetEvaluation], filters: Dict[str, Any]
    ) -> List[DatasetEvaluation]:
        """Apply filters to evaluations list."""
        filtered = evaluations

        if "priority_tier" in filters:
            filtered = [e for e in filtered if e.priority_tier == filters["priority_tier"]]

        if "min_overall_score" in filters:
            min_score = filters["min_overall_score"]
            filtered = [e for e in filtered if e.overall_score >= min_score]

        if "max_overall_score" in filters:
            max_score = filters["max_overall_score"]
            filtered = [e for e in filtered if e.overall_score <= max_score]

        return filtered

    def _apply_sorting(
        self, evaluations: List[DatasetEvaluation], sort_by: str, sort_order: str
    ) -> List[DatasetEvaluation]:
        """Apply sorting to evaluations list."""
        reverse = sort_order == "desc"

        if sort_by == "overall_score":
            return sorted(evaluations, key=lambda e: e.overall_score, reverse=reverse)
        elif sort_by == "evaluation_date":
            return sorted(evaluations, key=lambda e: e.evaluation_date, reverse=reverse)
        elif sort_by == "therapeutic_relevance":
            return sorted(
                evaluations, key=lambda e: e.therapeutic_relevance, reverse=reverse
            )
        else:
            return evaluations

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict[str, Any]:
        """Convert DatasetEvaluation to dictionary."""
        return {
            "evaluation_id": f"eval_{evaluation.source_id}",
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date.isoformat(),
            "evaluator": evaluation.evaluator,
            "therapeutic_relevance_notes": evaluation.therapeutic_relevance_notes,
            "data_structure_notes": evaluation.data_structure_notes,
            "integration_notes": evaluation.integration_notes,
            "ethical_notes": evaluation.ethical_notes,
            "competitive_advantages": evaluation.competitive_advantages,
            "compliance_checked": evaluation.compliance_checked,
            "compliance_status": evaluation.compliance_status,
            "compliance_score": evaluation.compliance_score,
            "license_compatible": evaluation.license_compatible,
            "privacy_compliant": evaluation.privacy_compliant,
            "hipaa_compliant": evaluation.hipaa_compliant,
        }


class GetEvaluationTool(MCPTool):
    """Tool for getting a specific evaluation."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetEvaluationTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_evaluation",
            description="Get details of a specific dataset evaluation",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "evaluation_id": {
                        "type": "string",
                        "description": "Evaluation ID (format: eval_{source_id})",
                    },
                },
                "required": ["session_id", "evaluation_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_evaluation tool.

        Args:
            params: Tool parameters

        Returns:
            Evaluation details
        """
        try:
            session_id = params.get("session_id")
            evaluation_id = params.get("evaluation_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not evaluation_id:
                raise ValueError("evaluation_id is required")

            # Get evaluation
            evaluation = self.service.get_evaluation(session_id, evaluation_id)

            # Convert evaluation to dict
            return self._evaluation_to_dict(evaluation)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting evaluation: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get evaluation: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict[str, Any]:
        """Convert DatasetEvaluation to dictionary."""
        return {
            "evaluation_id": f"eval_{evaluation.source_id}",
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date.isoformat(),
            "evaluator": evaluation.evaluator,
            "therapeutic_relevance_notes": evaluation.therapeutic_relevance_notes,
            "data_structure_notes": evaluation.data_structure_notes,
            "integration_notes": evaluation.integration_notes,
            "ethical_notes": evaluation.ethical_notes,
            "competitive_advantages": evaluation.competitive_advantages,
            "compliance_checked": evaluation.compliance_checked,
            "compliance_status": evaluation.compliance_status,
            "compliance_score": evaluation.compliance_score,
            "license_compatible": evaluation.license_compatible,
            "privacy_compliant": evaluation.privacy_compliant,
            "hipaa_compliant": evaluation.hipaa_compliant,
        }


class UpdateEvaluationTool(MCPTool):
    """Tool for updating evaluation scores."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize UpdateEvaluationTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="update_evaluation",
            description="Update evaluation scores and priority tier for a dataset evaluation. The overall score will be automatically recalculated based on the updated scores.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "evaluation_id": {
                        "type": "string",
                        "description": "Evaluation ID (format: eval_{source_id})",
                    },
                    "therapeutic_relevance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Therapeutic relevance score (1-10)",
                    },
                    "data_structure_quality": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Data structure quality score (1-10)",
                    },
                    "training_integration": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Training integration score (1-10)",
                    },
                    "ethical_accessibility": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Ethical accessibility score (1-10)",
                    },
                    "priority_tier": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority tier",
                    },
                },
                "required": ["session_id", "evaluation_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute update_evaluation tool.

        Args:
            params: Tool parameters

        Returns:
            Updated evaluation details
        """
        try:
            session_id = params.get("session_id")
            evaluation_id = params.get("evaluation_id")
            therapeutic_relevance = params.get("therapeutic_relevance")
            data_structure_quality = params.get("data_structure_quality")
            training_integration = params.get("training_integration")
            ethical_accessibility = params.get("ethical_accessibility")
            priority_tier = params.get("priority_tier")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not evaluation_id:
                raise ValueError("evaluation_id is required")

            # Validate score ranges if provided
            if therapeutic_relevance is not None and not (1 <= therapeutic_relevance <= 10):
                raise ValueError("therapeutic_relevance must be between 1 and 10")
            if data_structure_quality is not None and not (1 <= data_structure_quality <= 10):
                raise ValueError("data_structure_quality must be between 1 and 10")
            if training_integration is not None and not (1 <= training_integration <= 10):
                raise ValueError("training_integration must be between 1 and 10")
            if ethical_accessibility is not None and not (1 <= ethical_accessibility <= 10):
                raise ValueError("ethical_accessibility must be between 1 and 10")

            # Update evaluation
            evaluation = self.service.update_evaluation(
                session_id=session_id,
                evaluation_id=evaluation_id,
                therapeutic_relevance=therapeutic_relevance,
                data_structure_quality=data_structure_quality,
                training_integration=training_integration,
                ethical_accessibility=ethical_accessibility,
                priority_tier=priority_tier,
            )

            # Convert evaluation to dict
            return self._evaluation_to_dict(evaluation)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error updating evaluation: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to update evaluation: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict[str, Any]:
        """Convert DatasetEvaluation to dictionary."""
        return {
            "evaluation_id": f"eval_{evaluation.source_id}",
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date.isoformat(),
            "evaluator": evaluation.evaluator,
            "therapeutic_relevance_notes": evaluation.therapeutic_relevance_notes,
            "data_structure_notes": evaluation.data_structure_notes,
            "integration_notes": evaluation.integration_notes,
            "ethical_notes": evaluation.ethical_notes,
            "competitive_advantages": evaluation.competitive_advantages,
            "compliance_checked": evaluation.compliance_checked,
            "compliance_status": evaluation.compliance_status,
            "compliance_score": evaluation.compliance_score,
            "license_compatible": evaluation.license_compatible,
            "privacy_compliant": evaluation.privacy_compliant,
            "hipaa_compliant": evaluation.hipaa_compliant,
        }

