"""
Dataset acquisition tools for MCP Server.

This module provides tools for acquiring dataset sources through the MCP protocol.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.integration.mcp_pipeline_bridge import (
    MCPPipelineBridge,
)
from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.tools.base import MCPTool
from ai.sourcing.journal.models.dataset_models import AcquiredDataset

logger = logging.getLogger(__name__)


class AcquireDatasetsTool(MCPTool):
    """Tool for acquiring dataset sources."""

    def __init__(
        self,
        service: CommandHandlerService,
        pipeline_bridge: Optional[MCPPipelineBridge] = None,
    ) -> None:
        """
        Initialize AcquireDatasetsTool.

        Args:
            service: CommandHandlerService instance
            pipeline_bridge: Optional MCPPipelineBridge for automatic pipeline integration
        """
        self.service = service
        self.pipeline_bridge = pipeline_bridge
        super().__init__(
            name="acquire_datasets",
            description="Acquire dataset sources for a research session. This tool initiates the acquisition process by submitting access requests and downloading datasets from the specified sources. If pipeline bridge is configured, acquired datasets are automatically integrated into the training pipeline.",
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
                        "description": "Optional list of source IDs to acquire. If not provided, all sources in the session will be acquired.",
                    },
                    "auto_integrate": {
                        "type": "boolean",
                        "description": "Whether to automatically integrate acquired datasets into training pipeline (default: true if bridge configured)",
                        "default": True,
                    },
                },
                "required": ["session_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute acquire_datasets tool.

        Args:
            params: Tool parameters

        Returns:
            Acquisition result with progress updates
        """
        try:
            session_id = params.get("session_id")
            source_ids = params.get("source_ids")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")

            # Initiate acquisition
            result = self.service.initiate_acquisition(
                session_id=session_id,
                source_ids=source_ids if source_ids else None,
            )

            # Get acquired datasets
            acquisitions_list = self.service.get_acquisitions(session_id)
            
            # Auto-integrate with training pipeline if bridge is configured
            integration_results = []
            auto_integrate = params.get("auto_integrate", True)
            
            if self.pipeline_bridge and auto_integrate:
                logger.info(
                    f"Auto-integrating {len(acquisitions_list)} acquired datasets "
                    "into training pipeline"
                )
                
                # Get orchestrator to access evaluations and integration plans
                orchestrator = self.service.orchestrator
                state = orchestrator.get_session_state(session_id)
                
                for acquisition in acquisitions_list:
                    try:
                        # Get evaluation if available
                        evaluation = None
                        for eval_item in state.evaluations:
                            if eval_item.source_id == acquisition.source_id:
                                evaluation = eval_item
                                break
                        
                        # Get integration plan if available
                        integration_plan = None
                        for plan in state.integration_plans:
                            if plan.source_id == acquisition.source_id:
                                integration_plan = plan
                                break
                        
                        # Trigger pipeline integration
                        integration_result = self.pipeline_bridge.on_dataset_acquired(
                            dataset=acquisition,
                            evaluation=evaluation,
                            integration_plan=integration_plan,
                        )
                        integration_results.append({
                            "source_id": acquisition.source_id,
                            "integration_status": integration_result.get("status"),
                            "auto_integrated": integration_result.get("auto_integrated", False),
                        })
                        
                    except Exception as e:
                        logger.error(
                            f"Error auto-integrating dataset {acquisition.source_id}: {e}",
                            exc_info=True,
                        )
                        integration_results.append({
                            "source_id": acquisition.source_id,
                            "integration_status": "failed",
                            "auto_integrated": False,
                            "error": str(e),
                        })

            # Convert acquisitions to dicts
            acquisitions_data = []
            for acquisition in acquisitions_list:
                acquisitions_data.append(self._acquisition_to_dict(acquisition))

            return {
                "session_id": session_id,
                "acquired": result.get("acquired", []),
                "acquisitions": acquisitions_data,
                "total_acquired": len(acquisitions_data),
                "status": "completed",
                "pipeline_integration": {
                    "enabled": self.pipeline_bridge is not None and auto_integrate,
                    "results": integration_results,
                    "total_integrated": sum(
                        1 for r in integration_results
                        if r.get("integration_status") == "completed"
                    ),
                },
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error acquiring datasets: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to acquire datasets: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _acquisition_to_dict(self, acquisition: AcquiredDataset) -> Dict[str, Any]:
        """Convert AcquiredDataset to dictionary."""
        return {
            "acquisition_id": f"acq_{acquisition.source_id}",
            "source_id": acquisition.source_id,
            "status": "completed" if acquisition.storage_path else "pending",
            "download_progress": 100.0 if acquisition.storage_path else 0.0,
            "file_path": acquisition.storage_path,
            "file_size_mb": acquisition.file_size_mb,
            "file_format": acquisition.file_format,
            "acquired_date": acquisition.acquisition_date.isoformat(),
            "license": acquisition.license,
            "usage_restrictions": acquisition.usage_restrictions,
            "attribution_required": acquisition.attribution_required,
            "checksum": acquisition.checksum,
            "encrypted": acquisition.encrypted,
            "compliance_status": acquisition.compliance_status,
            "compliance_score": acquisition.compliance_score,
            "hipaa_compliant": acquisition.hipaa_compliant,
            "privacy_assessed": acquisition.privacy_assessed,
        }


class GetAcquisitionsTool(MCPTool):
    """Tool for getting all acquisitions for a session."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetAcquisitionsTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_acquisitions",
            description="Get all acquisitions for a research session with optional filtering and sorting",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional filters for acquisitions",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["completed", "pending"],
                            },
                            "file_format": {
                                "type": "string",
                                "description": "Filter by file format (e.g., 'csv', 'json', 'parquet')",
                            },
                            "min_file_size_mb": {
                                "type": "number",
                                "description": "Minimum file size in MB",
                            },
                            "max_file_size_mb": {
                                "type": "number",
                                "description": "Maximum file size in MB",
                            },
                            "compliance_status": {
                                "type": "string",
                                "enum": ["compliant", "partially_compliant", "non_compliant", "unknown"],
                            },
                        },
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["acquired_date", "file_size_mb", "source_id"],
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
        Execute get_acquisitions tool.

        Args:
            params: Tool parameters

        Returns:
            List of acquisitions with optional filtering and sorting
        """
        try:
            session_id = params.get("session_id")
            if not session_id:
                raise ValueError("session_id is required")

            # Get acquisitions
            acquisitions_list = self.service.get_acquisitions(session_id)

            # Apply filters
            filters = params.get("filters", {})
            if filters:
                acquisitions_list = self._apply_filters(acquisitions_list, filters)

            # Apply sorting
            sort_by = params.get("sort_by")
            sort_order = params.get("sort_order", "desc")
            if sort_by:
                acquisitions_list = self._apply_sorting(acquisitions_list, sort_by, sort_order)

            # Convert acquisitions to dicts
            acquisitions_data = []
            for acquisition in acquisitions_list:
                acquisitions_data.append(self._acquisition_to_dict(acquisition))

            return {
                "session_id": session_id,
                "acquisitions": acquisitions_data,
                "count": len(acquisitions_data),
            }
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting acquisitions: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get acquisitions: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _apply_filters(
        self, acquisitions: List[AcquiredDataset], filters: Dict[str, Any]
    ) -> List[AcquiredDataset]:
        """Apply filters to acquisitions list."""
        filtered = acquisitions

        if "status" in filters:
            status = filters["status"]
            if status == "completed":
                filtered = [a for a in filtered if a.storage_path]
            elif status == "pending":
                filtered = [a for a in filtered if not a.storage_path]

        if "file_format" in filters:
            file_format = filters["file_format"]
            filtered = [a for a in filtered if a.file_format == file_format]

        if "min_file_size_mb" in filters:
            min_size = filters["min_file_size_mb"]
            filtered = [a for a in filtered if a.file_size_mb >= min_size]

        if "max_file_size_mb" in filters:
            max_size = filters["max_file_size_mb"]
            filtered = [a for a in filtered if a.file_size_mb <= max_size]

        if "compliance_status" in filters:
            compliance_status = filters["compliance_status"]
            filtered = [a for a in filtered if a.compliance_status == compliance_status]

        return filtered

    def _apply_sorting(
        self, acquisitions: List[AcquiredDataset], sort_by: str, sort_order: str
    ) -> List[AcquiredDataset]:
        """Apply sorting to acquisitions list."""
        reverse = sort_order == "desc"

        if sort_by == "acquired_date":
            return sorted(acquisitions, key=lambda a: a.acquisition_date, reverse=reverse)
        elif sort_by == "file_size_mb":
            return sorted(acquisitions, key=lambda a: a.file_size_mb, reverse=reverse)
        elif sort_by == "source_id":
            return sorted(acquisitions, key=lambda a: a.source_id, reverse=reverse)
        else:
            return acquisitions

    def _acquisition_to_dict(self, acquisition: AcquiredDataset) -> Dict[str, Any]:
        """Convert AcquiredDataset to dictionary."""
        return {
            "acquisition_id": f"acq_{acquisition.source_id}",
            "source_id": acquisition.source_id,
            "status": "completed" if acquisition.storage_path else "pending",
            "download_progress": 100.0 if acquisition.storage_path else 0.0,
            "file_path": acquisition.storage_path,
            "file_size_mb": acquisition.file_size_mb,
            "file_format": acquisition.file_format,
            "acquired_date": acquisition.acquisition_date.isoformat(),
            "license": acquisition.license,
            "usage_restrictions": acquisition.usage_restrictions,
            "attribution_required": acquisition.attribution_required,
            "checksum": acquisition.checksum,
            "encrypted": acquisition.encrypted,
            "compliance_status": acquisition.compliance_status,
            "compliance_score": acquisition.compliance_score,
            "hipaa_compliant": acquisition.hipaa_compliant,
            "privacy_assessed": acquisition.privacy_assessed,
        }


class GetAcquisitionTool(MCPTool):
    """Tool for getting a specific acquisition."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize GetAcquisitionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="get_acquisition",
            description="Get details of a specific dataset acquisition",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "acquisition_id": {
                        "type": "string",
                        "description": "Acquisition ID (format: acq_{source_id})",
                    },
                },
                "required": ["session_id", "acquisition_id"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute get_acquisition tool.

        Args:
            params: Tool parameters

        Returns:
            Acquisition details
        """
        try:
            session_id = params.get("session_id")
            acquisition_id = params.get("acquisition_id")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not acquisition_id:
                raise ValueError("acquisition_id is required")

            # Get acquisition
            acquisition = self.service.get_acquisition(session_id, acquisition_id)

            # Convert acquisition to dict
            return self._acquisition_to_dict(acquisition)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error getting acquisition: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to get acquisition: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _acquisition_to_dict(self, acquisition: AcquiredDataset) -> Dict[str, Any]:
        """Convert AcquiredDataset to dictionary."""
        return {
            "acquisition_id": f"acq_{acquisition.source_id}",
            "source_id": acquisition.source_id,
            "status": "completed" if acquisition.storage_path else "pending",
            "download_progress": 100.0 if acquisition.storage_path else 0.0,
            "file_path": acquisition.storage_path,
            "file_size_mb": acquisition.file_size_mb,
            "file_format": acquisition.file_format,
            "acquired_date": acquisition.acquisition_date.isoformat(),
            "license": acquisition.license,
            "usage_restrictions": acquisition.usage_restrictions,
            "attribution_required": acquisition.attribution_required,
            "checksum": acquisition.checksum,
            "encrypted": acquisition.encrypted,
            "compliance_status": acquisition.compliance_status,
            "compliance_score": acquisition.compliance_score,
            "hipaa_compliant": acquisition.hipaa_compliant,
            "privacy_assessed": acquisition.privacy_assessed,
        }


class UpdateAcquisitionTool(MCPTool):
    """Tool for updating acquisition status."""

    def __init__(self, service: CommandHandlerService) -> None:
        """
        Initialize UpdateAcquisitionTool.

        Args:
            service: CommandHandlerService instance
        """
        self.service = service
        super().__init__(
            name="update_acquisition",
            description="Update acquisition status. Note: Currently, the underlying model does not support status updates, so this tool returns the acquisition without modification.",
            parameters={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID",
                    },
                    "acquisition_id": {
                        "type": "string",
                        "description": "Acquisition ID (format: acq_{source_id})",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "approved", "denied", "downloaded", "error"],
                        "description": "New status for the acquisition",
                    },
                },
                "required": ["session_id", "acquisition_id", "status"],
            },
        )

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute update_acquisition tool.

        Args:
            params: Tool parameters

        Returns:
            Updated acquisition details
        """
        try:
            session_id = params.get("session_id")
            acquisition_id = params.get("acquisition_id")
            status = params.get("status")

            # Validate parameters
            if not session_id:
                raise ValueError("session_id is required")
            if not acquisition_id:
                raise ValueError("acquisition_id is required")
            if not status:
                raise ValueError("status is required")

            # Update acquisition
            acquisition = self.service.update_acquisition(
                session_id=session_id,
                acquisition_id=acquisition_id,
                status=status,
            )

            # Convert acquisition to dict
            return self._acquisition_to_dict(acquisition)
        except ValueError as e:
            raise MCPError(
                MCPErrorCode.TOOL_VALIDATION_ERROR,
                f"Invalid parameters: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e
        except Exception as e:
            logger.exception(f"Error updating acquisition: {e}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Failed to update acquisition: {str(e)}",
                {"params": params, "error": str(e)},
            ) from e

    def _acquisition_to_dict(self, acquisition: AcquiredDataset) -> Dict[str, Any]:
        """Convert AcquiredDataset to dictionary."""
        return {
            "acquisition_id": f"acq_{acquisition.source_id}",
            "source_id": acquisition.source_id,
            "status": "completed" if acquisition.storage_path else "pending",
            "download_progress": 100.0 if acquisition.storage_path else 0.0,
            "file_path": acquisition.storage_path,
            "file_size_mb": acquisition.file_size_mb,
            "file_format": acquisition.file_format,
            "acquired_date": acquisition.acquisition_date.isoformat(),
            "license": acquisition.license,
            "usage_restrictions": acquisition.usage_restrictions,
            "attribution_required": acquisition.attribution_required,
            "checksum": acquisition.checksum,
            "encrypted": acquisition.encrypted,
            "compliance_status": acquisition.compliance_status,
            "compliance_score": acquisition.compliance_score,
            "hipaa_compliant": acquisition.hipaa_compliant,
            "privacy_assessed": acquisition.privacy_assessed,
        }

