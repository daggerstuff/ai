"""
MCP Server implementation.

This module provides the main MCP server class that handles protocol requests,
tool execution, resource access, and prompt rendering.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from ai.sourcing.journal.mcp.config import MCPConfig, load_mcp_config
from ai.sourcing.journal.mcp.protocol import (
    JSONRPCErrorCode,
    MCPError,
    MCPErrorCode,
    MCPProtocolHandler,
    MCPRequest,
    MCPResponse,
)
from ai.sourcing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)
from ai.sourcing.journal.mcp.resources import (
  ProgressHistoryResource,
  ProgressMetricsResource,
  ResourceRegistry,
  SessionMetricsResource,
  SessionStateResource,
)
from ai.sourcing.journal.mcp.prompts import PromptRegistry
from ai.sourcing.journal.mcp.prompts.discovery import DiscoverSourcesPrompt
from ai.sourcing.journal.mcp.prompts.evaluation import EvaluateSourcesPrompt
from ai.sourcing.journal.mcp.prompts.acquisition import AcquireDatasetsPrompt
from ai.sourcing.journal.mcp.prompts.integration import CreateIntegrationPlansPrompt
from ai.sourcing.journal.mcp.tools.executor import ToolExecutor
from ai.sourcing.journal.mcp.tools.registry import ToolRegistry
from ai.sourcing.journal.mcp.utils.progress_streaming import ProgressStreamer
from ai.sourcing.journal.mcp.tools.acquisition import (
    AcquireDatasetsTool,
    GetAcquisitionTool,
    GetAcquisitionsTool,
    UpdateAcquisitionTool,
)

# Optional pipeline bridge integration
try:
    from ai.sourcing.journal.integration.mcp_pipeline_bridge import (
        MCPPipelineBridge,
    )
    PIPELINE_BRIDGE_AVAILABLE = True
except ImportError:
    PIPELINE_BRIDGE_AVAILABLE = False
    MCPPipelineBridge = None  # type: ignore
from ai.sourcing.journal.mcp.tools.discovery import (
    DiscoverSourcesTool,
    FilterSourcesTool,
    GetSourceTool,
    GetSourcesTool,
)
from ai.sourcing.journal.mcp.tools.evaluation import (
    EvaluateSourcesTool,
    GetEvaluationTool,
    GetEvaluationsTool,
    UpdateEvaluationTool,
)
from ai.sourcing.journal.mcp.tools.integration import (
    CreateIntegrationPlansTool,
    GeneratePreprocessingScriptTool,
    GetIntegrationPlanTool,
    GetIntegrationPlansTool,
)
from ai.sourcing.journal.mcp.tools.reports import (
    GenerateReportTool,
    GetReportTool,
    ListReportsTool,
)
from ai.sourcing.journal.mcp.tools.sessions import (
    CreateSessionTool,
    DeleteSessionTool,
    GetSessionTool,
    ListSessionsTool,
    UpdateSessionTool,
)
from ai.sourcing.journal.mcp.utils.error_handling import MCPErrorHandler
from ai.sourcing.journal.mcp.utils.logging import get_logger, setup_logging
from ai.sourcing.journal.mcp.utils.rate_limiting import (
    RateLimitManager,
    check_rate_limit as check_rate_limit_request,
)
from ai.sourcing.journal.mcp.utils.security import (
    sanitize_input,
    sanitize_json_output,
    validate_and_sanitize_input,
    SecurityError,
)
from ai.sourcing.journal.mcp.utils.audit_logging import (
    AuditLogger,
    create_audit_logger,
)
from ai.sourcing.journal.mcp.auth import (
    create_auth_handler,
    create_authorization_handler,
)

logger = get_logger(__name__)


class MCPServer:
    """Main MCP server implementation."""

    def __init__(self, config: Optional[MCPConfig] = None) -> None:
        """
        Initialize MCP server.

        Args:
            config: Optional MCP configuration (loads from environment if not provided)
        """
        self.config = config or load_mcp_config()

        # Set up logging
        setup_logging(self.config.logging)

        # Initialize CommandHandlerService for backend integration
        self.command_handler_service = CommandHandlerService(
            config=self.config.command_handler_config
        )

        # Initialize progress streaming (Phase 12)
        self.progress_streamer = ProgressStreamer()

        # Initialize registries
        self.tools = ToolRegistry()
        self.tool_executor = ToolExecutor(
            self.tools, progress_streamer=self.progress_streamer
        )
        self.resources = ResourceRegistry()
        self.prompts = PromptRegistry()

        # Initialize protocol handler
        self.protocol_handler = MCPProtocolHandler()

        # Initialize authentication and authorization (Phase 11)
        self.auth_handler = create_auth_handler(self.config.auth)
        self.authorization_handler = create_authorization_handler()
        self.current_user: Optional[Dict[str, Any]] = None

        # Initialize rate limiting (Phase 14)
        self.rate_limiter: Optional[RateLimitManager] = None
        if self.config.rate_limits.enabled:
            self.rate_limiter = RateLimitManager(self.config.rate_limits)

        # Initialize audit logging (Phase 14)
        self.audit_logger = create_audit_logger(self.config.logging)
        
        # Initialize pipeline bridge (optional, for training pipeline integration)
        self.pipeline_bridge: Optional[MCPPipelineBridge] = None
        if PIPELINE_BRIDGE_AVAILABLE:
            try:
                self.pipeline_bridge = MCPPipelineBridge(auto_integrate=True)
                logger.info("Pipeline bridge initialized (auto-integration enabled)")
            except Exception as e:
                logger.warning(f"Failed to initialize pipeline bridge: {e}")

        # Register session management tools (Phase 3)
        self._register_session_tools()

        # Register source discovery tools (Phase 4)
        self._register_discovery_tools()

        # Register dataset evaluation tools (Phase 5)
        self._register_evaluation_tools()

        # Register dataset acquisition tools (Phase 6)
        self._register_acquisition_tools()

        # Register integration planning tools (Phase 7)
        self._register_integration_tools()

        # Register report generation tools (Phase 8)
        self._register_report_tools()

        # Register resources (Phase 9)
        self._register_resources()

        # Register prompts (Phase 10)
        self._register_prompts()

        logger.info(
            f"MCP Server initialized: {self.config.server_name} v{self.config.server_version}"
        )

    async def handle_request(self, request_data: Any) -> str:
        """
        Handle MCP protocol request.

        Args:
            request_data: Request data (JSON string, bytes, or dict)

        Returns:
            JSON string response
        """
        request_id: Optional[Any] = None

        try:
            # Parse request
            try:
                request = self.protocol_handler.parse_request(request_data)
                request_id = request.id
            except Exception as e:
                # Parse error - return error response without request ID
                error_response = MCPErrorHandler.format_error_response(
                    e,
                    request_id=None,
                    include_traceback=self.config.debug,
                )
                return self.protocol_handler.format_response(
                    MCPResponse(
                        error=error_response.get("error"),
                        id=error_response.get("id"),
                        jsonrpc=error_response.get("jsonrpc", "2.0"),
                    )
                )

            # Sanitize input (Phase 14 - Security)
            try:
                if request.params:
                    request.params = validate_and_sanitize_input(
                        request.params,
                        detect_injections=True,
                    )
            except SecurityError as e:
                # Log security violation
                self.audit_logger.log_security_violation(
                    violation_type="input_sanitization",
                    description=str(e),
                    request_id=str(request.id) if request.id else None,
                )
                error_response = MCPErrorHandler.format_error_response(
                    e,
                    request_id=request.id,
                    include_traceback=self.config.debug,
                )
                return self.protocol_handler.format_response(
                    MCPResponse(
                        error=error_response.get("error"),
                        id=error_response.get("id"),
                        jsonrpc=error_response.get("jsonrpc", "2.0"),
                    )
                )

            # Authenticate request (if authentication enabled)
            if self.config.auth.enabled and self.auth_handler:
                try:
                    await self._authenticate_request(request)
                except Exception as e:
                    # Log auth failure
                    user_id = None
                    if hasattr(e, "data") and isinstance(e.data, dict):
                        user_id = e.data.get("user_id")
                    self.audit_logger.log_auth_failure(
                        reason=str(e),
                        user_id=user_id,
                        request_id=str(request.id) if request.id else None,
                    )
                    error_response = MCPErrorHandler.format_error_response(
                        e,
                        request_id=request.id,
                        include_traceback=self.config.debug,
                    )
                    return self.protocol_handler.format_response(
                        MCPResponse(
                            error=error_response.get("error"),
                            id=error_response.get("id"),
                            jsonrpc=error_response.get("jsonrpc", "2.0"),
                        )
                    )

            # Rate limit check (if rate limiting enabled)
            if self.config.rate_limits.enabled and self.rate_limiter:
                try:
                    await self._check_rate_limit(request)
                except Exception as e:
                    # Log rate limit exceeded
                    user_id = self.current_user.get("user_id") if self.current_user else None
                    self.audit_logger.log_rate_limit_exceeded(
                        identifier=getattr(e, "data", {}).get("identifier", "unknown") if hasattr(e, "data") else "unknown",
                        user_id=user_id,
                        request_id=str(request.id) if request.id else None,
                    )
                    error_response = MCPErrorHandler.format_error_response(
                        e,
                        request_id=request.id,
                        include_traceback=self.config.debug,
                    )
                    return self.protocol_handler.format_response(
                        MCPResponse(
                            error=error_response.get("error"),
                            id=error_response.get("id"),
                            jsonrpc=error_response.get("jsonrpc", "2.0"),
                        )
                    )

            # Route request to appropriate handler
            response = await self._route_request(request)

            # Sanitize output (Phase 14 - Security)
            if response.result:
                try:
                    response.result = sanitize_json_output(response.result)
                except Exception as e:
                    logger.warning(f"Error sanitizing output: {e}")

            # Format and return response
            return self.protocol_handler.format_response(response)

        except Exception as e:
            # Unexpected error
            logger.exception("Unexpected error handling request")
            error_response = MCPErrorHandler.format_error_response(
                e,
                request_id=request_id,
                include_traceback=self.config.debug,
            )
            return self.protocol_handler.format_response(
                MCPResponse(
                    error=error_response.get("error"),
                    id=error_response.get("id"),
                    jsonrpc=error_response.get("jsonrpc", "2.0"),
                )
            )

    async def _route_request(self, request: MCPRequest) -> MCPResponse:
        """
        Route request to appropriate handler based on method.

        Args:
            request: MCP request

        Returns:
            MCP response
        """
        method = request.method

        # Handle MCP initialization methods
        if method == "initialize":
            return await self._handle_initialize(request)
        elif method == "notifications/initialized":
            return await self._handle_initialized(request)

        # Handle tool methods
        elif method.startswith("tools/"):
            return await self._handle_tool_request(request)

        # Handle resource methods
        elif method.startswith("resources/"):
            return await self._handle_resource_request(request)

        # Handle prompt methods
        elif method.startswith("prompts/"):
            return await self._handle_prompt_request(request)

        # Unknown method
        else:
            return MCPResponse.error(
                JSONRPCErrorCode.METHOD_NOT_FOUND,
                f"Method not found: {method}",
                id=request.id,
            )

    async def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """
        Handle initialize request.

        Args:
            request: Initialize request

        Returns:
            Initialize response
        """
        # Return server capabilities
        result = {
            "protocolVersion": self.config.protocol_version,
            "capabilities": {
                "tools": {
                    "listChanged": True,  # Indicates tools can change
                },
                "resources": {},
                "prompts": {},
            },
            "serverInfo": {
                "name": self.config.server_name,
                "version": self.config.server_version,
            },
        }

        return MCPResponse.success(result, id=request.id)

    async def _handle_initialized(self, request: MCPRequest) -> MCPResponse:
        """
        Handle initialized notification.

        Args:
            request: Initialized notification

        Returns:
            Empty response (notifications don't have responses)
        """
        logger.info("Client initialized")
        # Notifications don't return responses
        return MCPResponse.success(None, id=None)

    async def _handle_tool_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle tool request.

        Args:
            request: Tool request

        Returns:
            Tool response
        """
        method = request.method
        params = request.params or {}

        try:
            if method == "tools/list":
                # List all available tools
                tools = self.tool_executor.list_tools()
                return MCPResponse.success({"tools": tools}, id=request.id)

            elif method == "tools/call":
                # Execute a tool
                tool_name = params.get("name")
                if not tool_name:
                    return MCPResponse.error(
                        JSONRPCErrorCode.INVALID_PARAMS,
                        "Missing required parameter: name",
                        id=request.id,
                    )

                # Check authorization for tool execution
                try:
                    await self.authorization_handler.require_authorization(
                        self.current_user or {},
                        tool_name,
                        "execute",
                    )
                    # Log authorization granted
                    if self.current_user:
                        self.audit_logger.log_authorization_granted(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource=tool_name,
                            action="execute",
                            request_id=str(request.id) if request.id else None,
                        )
                except MCPError as e:
                    # Log authorization denied
                    if self.current_user:
                        self.audit_logger.log_authorization_denied(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource=tool_name,
                            action="execute",
                            reason=e.message,
                            request_id=str(request.id) if request.id else None,
                        )
                    return MCPResponse.error(
                        e.code,
                        e.message,
                        e.data,
                        id=request.id,
                    )

                tool_params = params.get("arguments", {})
                timeout = params.get("timeout")

                # Extract session_id from tool_params for audit logging
                session_id = tool_params.get("session_id") if isinstance(tool_params, dict) else None

                # Log tool execution start
                if self.current_user:
                    self.audit_logger.log_tool_execution_start(
                        user_id=self.current_user.get("user_id", "unknown"),
                        user_role=self.current_user.get("role", "unknown"),
                        tool_name=tool_name,
                        tool_params=tool_params,
                        request_id=str(request.id) if request.id else None,
                        session_id=session_id,
                    )

                try:
                    import time
                    start_time = time.time()
                    result = await self.tool_executor.execute_tool(
                        tool_name,
                        tool_params,
                        timeout=timeout,
                    )
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Log tool execution success
                    if self.current_user:
                        self.audit_logger.log_tool_execution_success(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            tool_name=tool_name,
                            execution_time_ms=execution_time_ms,
                            request_id=str(request.id) if request.id else None,
                            session_id=session_id,
                        )
                    # Format result according to MCP protocol
                    # Result should be a dict with content array
                    if isinstance(result, dict) and "content" in result:
                        # Already formatted
                        formatted_result = result
                    else:
                        # Format as text content
                        formatted_result = {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result),
                                }
                            ]
                        }
                    return MCPResponse.success(formatted_result, id=request.id)
                except MCPError as e:
                    # Log tool execution failure
                    if self.current_user:
                        self.audit_logger.log_tool_execution_failure(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            tool_name=tool_name,
                            error_message=e.message,
                            request_id=str(request.id) if request.id else None,
                            session_id=session_id,
                        )
                    return MCPResponse.error(
                        e.code,
                        e.message,
                        e.data,
                        id=request.id,
                    )

            else:
                return MCPResponse.error(
                    JSONRPCErrorCode.METHOD_NOT_FOUND,
                    f"Unknown tool method: {method}",
                    id=request.id,
                )

        except Exception as e:
            logger.exception(f"Error handling tool request: {method}")
            error_response = MCPErrorHandler.format_error_response(
                e,
                request_id=request.id,
                include_traceback=self.config.debug,
            )
            return MCPResponse(
                error=error_response.get("error"),
                id=error_response.get("id"),
                jsonrpc=error_response.get("jsonrpc", "2.0"),
            )

    async def _handle_resource_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle resource request.

        Args:
            request: Resource request

        Returns:
            Resource response
        """
        method = request.method
        params = request.params or {}

        try:
            if method == "resources/list":
                # List all available resources
                resources = self.resources.get_resource_schemas()
                return MCPResponse.success({"resources": resources}, id=request.id)

            elif method == "resources/read":
                # Read a resource
                uri = params.get("uri")
                if not uri:
                    return MCPResponse.error(
                        JSONRPCErrorCode.INVALID_PARAMS,
                        "Missing required parameter: uri",
                        id=request.id,
                    )

                # Check authorization for resource access
                try:
                    await self.authorization_handler.require_authorization(
                        self.current_user or {},
                        uri,
                        "read",
                    )
                    # Log authorization granted
                    if self.current_user:
                        self.audit_logger.log_authorization_granted(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource=uri,
                            action="read",
                            request_id=str(request.id) if request.id else None,
                        )
                except MCPError as e:
                    # Log authorization denied
                    if self.current_user:
                        self.audit_logger.log_authorization_denied(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource=uri,
                            action="read",
                            reason=e.message,
                            request_id=str(request.id) if request.id else None,
                        )
                    return MCPResponse.error(
                        e.code,
                        e.message,
                        e.data,
                        id=request.id,
                    )

                # Find resource by URI pattern
                resource = None
                for registered_resource in self.resources.list_resources():
                    if registered_resource.validate_uri(uri):
                        resource = registered_resource
                        break

                if not resource:
                    return MCPResponse.error(
                        MCPErrorCode.RESOURCE_NOT_FOUND,
                        f"Resource not found: {uri}",
                        id=request.id,
                    )

                # Extract parameters from URI if needed
                read_params = params.get("params", {})

                # If resource has extract_session_id method, try to extract session_id from URI
                if hasattr(resource, "extract_session_id"):
                    session_id = resource.extract_session_id(uri)
                    if session_id:
                        read_params["session_id"] = session_id

                # Validate resource parameters if resource has validation
                if hasattr(resource, "validate_parameters") and read_params:
                    try:
                        resource.validate_parameters(read_params)
                    except Exception as e:
                        from ai.sourcing.journal.mcp.utils.validation import (
                            ValidationError,
                        )

                        if isinstance(e, ValidationError):
                            return MCPResponse.error(
                                e.code,
                                e.message,
                                e.data,
                                id=request.id,
                            )
                        else:
                            return MCPResponse.error(
                                MCPErrorCode.RESOURCE_NOT_FOUND,
                                f"Invalid resource parameters: {str(e)}",
                                id=request.id,
                            )

                # Extract session_id from URI or params for audit logging
                session_id = read_params.get("session_id") if read_params else None
                if not session_id and hasattr(resource, "extract_session_id"):
                    session_id = resource.extract_session_id(uri)

                # Read resource content
                try:
                    result = await resource.read(read_params if read_params else None)
                    # Log successful resource access
                    if self.current_user:
                        self.audit_logger.log_resource_access(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource_uri=uri,
                            success=True,
                            request_id=str(request.id) if request.id else None,
                            session_id=session_id,
                        )
                    return MCPResponse.success(result, id=request.id)
                except MCPError as e:
                    # Log failed resource access
                    if self.current_user:
                        self.audit_logger.log_resource_access(
                            user_id=self.current_user.get("user_id", "unknown"),
                            user_role=self.current_user.get("role", "unknown"),
                            resource_uri=uri,
                            success=False,
                            error_message=e.message,
                            request_id=str(request.id) if request.id else None,
                            session_id=session_id,
                        )
                    return MCPResponse.error(
                        e.code,
                        e.message,
                        e.data,
                        id=request.id,
                    )
                except Exception as e:
                    # Handle unexpected errors in resource access
                    error_response = MCPErrorHandler.handle_error(
                        e,
                        request_id=request.id,
                        context={"method": method, "uri": uri},
                        include_traceback=self.config.debug,
                    )
                    return MCPResponse(
                        error=error_response.get("error"),
                        id=error_response.get("id"),
                        jsonrpc=error_response.get("jsonrpc", "2.0"),
                    )

            else:
                return MCPResponse.error(
                    JSONRPCErrorCode.METHOD_NOT_FOUND,
                    f"Unknown resource method: {method}",
                    id=request.id,
                )

        except Exception as e:
            logger.exception(f"Error handling resource request: {method}")
            error_response = MCPErrorHandler.format_error_response(
                e,
                request_id=request.id,
                include_traceback=self.config.debug,
            )
            return MCPResponse(
                error=error_response.get("error"),
                id=error_response.get("id"),
                jsonrpc=error_response.get("jsonrpc", "2.0"),
            )

    async def _handle_prompt_request(self, request: MCPRequest) -> MCPResponse:
        """
        Handle prompt request.

        Args:
            request: Prompt request

        Returns:
            Prompt response
        """
        method = request.method
        params = request.params or {}

        try:
            if method == "prompts/list":
                # List all available prompts
                prompts = self.prompts.get_prompt_schemas()
                return MCPResponse.success({"prompts": prompts}, id=request.id)

            elif method == "prompts/get":
                # Get a specific prompt
                prompt_name = params.get("name")
                if not prompt_name:
                    return MCPResponse.error(
                        JSONRPCErrorCode.INVALID_PARAMS,
                        "Missing required parameter: name",
                        id=request.id,
                    )

                prompt = self.prompts.get(prompt_name)
                if not prompt:
                    return MCPResponse.error(
                        MCPErrorCode.RESOURCE_NOT_FOUND,
                        f"Prompt not found: {prompt_name}",
                        id=request.id,
                    )

                # Get prompt arguments
                prompt_args = params.get("arguments", {})

                # Validate prompt arguments
                try:
                    prompt.validate_arguments(prompt_args)
                except Exception as e:
                    from ai.sourcing.journal.mcp.utils.validation import (
                        ValidationError,
                    )

                    if isinstance(e, ValidationError):
                        return MCPResponse.error(
                            e.code,
                            e.message,
                            e.data,
                            id=request.id,
                        )
                    else:
                        return MCPResponse.error(
                            JSONRPCErrorCode.INVALID_PARAMS,
                            f"Invalid prompt arguments: {str(e)}",
                            id=request.id,
                        )

                # Render prompt with arguments
                try:
                    rendered_prompt = prompt.render(prompt_args)
                    return MCPResponse.success(
                        {
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {
                                        "type": "text",
                                        "text": rendered_prompt,
                                    },
                                }
                            ],
                        },
                        id=request.id,
                    )
                except Exception as e:
                    # Handle unexpected errors in prompt rendering
                    error_response = MCPErrorHandler.handle_error(
                        e,
                        request_id=request.id,
                        context={"method": method, "prompt_name": prompt_name},
                        include_traceback=self.config.debug,
                    )
                    return MCPResponse(
                        error=error_response.get("error"),
                        id=error_response.get("id"),
                        jsonrpc=error_response.get("jsonrpc", "2.0"),
                    )

            else:
                return MCPResponse.error(
                    JSONRPCErrorCode.METHOD_NOT_FOUND,
                    f"Unknown prompt method: {method}",
                    id=request.id,
                )

        except Exception as e:
            logger.exception(f"Error handling prompt request: {method}")
            error_response = MCPErrorHandler.format_error_response(
                e,
                request_id=request.id,
                include_traceback=self.config.debug,
            )
            return MCPResponse(
                error=error_response.get("error"),
                id=error_response.get("id"),
                jsonrpc=error_response.get("jsonrpc", "2.0"),
            )

    async def _authenticate_request(self, request: MCPRequest) -> None:
        """
        Authenticate request.

        Args:
            request: Request to authenticate

        Raises:
            MCPError: If authentication fails
        """
        if not self.auth_handler:
            # Authentication disabled, set default user
            self.current_user = {
                "user_id": "anonymous",
                "email": None,
                "role": "viewer",
                "permissions": [],
            }
            return

        try:
            # Authenticate request and get user information
            user = await self.auth_handler.authenticate(request)
            self.current_user = user
            logger.debug(f"Request authenticated for user: {user.get('user_id')}")

            # Log successful authentication
            self.audit_logger.log_auth_success(
                user_id=user.get("user_id", "unknown"),
                user_role=user.get("role"),
                request_id=str(request.id) if request.id else None,
            )
        except MCPError:
            # Re-raise MCP errors as-is (auth failure already logged in handle_request)
            raise
        except Exception as e:
            logger.exception("Unexpected error during authentication")
            # Log auth failure
            self.audit_logger.log_auth_failure(
                reason=str(e),
                request_id=str(request.id) if request.id else None,
            )
            raise MCPError(
                MCPErrorCode.AUTHENTICATION_ERROR,
                f"Authentication failed: {str(e)}",
            )

    async def _check_rate_limit(self, request: MCPRequest) -> None:
        """
        Check rate limit for request.

        Args:
            request: Request to check

        Raises:
            MCPError: If rate limit exceeded
        """
        if not self.rate_limiter:
            return

        await check_rate_limit_request(
            request,
            self.rate_limiter,
            self.current_user,
        )

    def register_tool(self, tool: Any) -> None:
        """
        Register an MCP tool.

        Args:
            tool: Tool to register
        """
        self.tools.register(tool)

    def register_resource(self, resource: Any) -> None:
        """
        Register an MCP resource.

        Args:
            resource: Resource to register
        """
        self.resources.register(resource)

    def register_prompt(self, prompt: Any) -> None:
        """
        Register an MCP prompt.

        Args:
            prompt: Prompt to register
        """
        # Will be implemented in Phase 10
        logger.warning("Prompt registration not yet implemented")

    def _register_session_tools(self) -> None:
        """Register session management tools."""
        # Register session management tools
        self.tools.register(CreateSessionTool(self.command_handler_service))
        self.tools.register(ListSessionsTool(self.command_handler_service))
        self.tools.register(GetSessionTool(self.command_handler_service))
        self.tools.register(UpdateSessionTool(self.command_handler_service))
        self.tools.register(DeleteSessionTool(self.command_handler_service))
        logger.info("Registered session management tools")

    def _register_discovery_tools(self) -> None:
        """Register source discovery tools."""
        # Register source discovery tools
        self.tools.register(DiscoverSourcesTool(self.command_handler_service))
        self.tools.register(GetSourcesTool(self.command_handler_service))
        self.tools.register(GetSourceTool(self.command_handler_service))
        self.tools.register(FilterSourcesTool(self.command_handler_service))
        logger.info("Registered source discovery tools")

    def _register_evaluation_tools(self) -> None:
        """Register dataset evaluation tools."""
        # Register dataset evaluation tools
        self.tools.register(EvaluateSourcesTool(self.command_handler_service))
        self.tools.register(GetEvaluationsTool(self.command_handler_service))
        self.tools.register(GetEvaluationTool(self.command_handler_service))
        self.tools.register(UpdateEvaluationTool(self.command_handler_service))
        logger.info("Registered dataset evaluation tools")

    def _register_acquisition_tools(self) -> None:
        """Register dataset acquisition tools."""
        # Register dataset acquisition tools with optional pipeline bridge
        self.tools.register(
            AcquireDatasetsTool(
                self.command_handler_service,
                pipeline_bridge=self.pipeline_bridge,
            )
        )
        self.tools.register(GetAcquisitionsTool(self.command_handler_service))
        self.tools.register(GetAcquisitionTool(self.command_handler_service))
        self.tools.register(UpdateAcquisitionTool(self.command_handler_service))
        logger.info("Registered dataset acquisition tools")
    
    def register_pipeline_orchestrator(self, orchestrator: Any) -> None:
        """
        Register pipeline orchestrator with the pipeline bridge.
        
        This enables automatic integration of acquired datasets into the training pipeline.
        
        Args:
            orchestrator: PipelineOrchestrator instance
        """
        if self.pipeline_bridge:
            self.pipeline_bridge.register_pipeline_orchestrator(orchestrator)
            logger.info("Pipeline orchestrator registered with MCP server bridge")
        else:
            logger.warning(
                "Pipeline bridge not available, cannot register orchestrator"
            )

    def _register_integration_tools(self) -> None:
        """Register integration planning tools."""
        # Register integration planning tools
        self.tools.register(CreateIntegrationPlansTool(self.command_handler_service))
        self.tools.register(GetIntegrationPlansTool(self.command_handler_service))
        self.tools.register(GetIntegrationPlanTool(self.command_handler_service))
        self.tools.register(GeneratePreprocessingScriptTool(self.command_handler_service))
        logger.info("Registered integration planning tools")

    def _register_report_tools(self) -> None:
        """Register report generation tools."""
        # Register report generation tools
        self.tools.register(GenerateReportTool(self.command_handler_service))
        self.tools.register(GetReportTool(self.command_handler_service))
        self.tools.register(ListReportsTool(self.command_handler_service))
        logger.info("Registered report generation tools")

    def _register_resources(self) -> None:
        """Register MCP resources."""
        # Register progress resources
        self.resources.register(
          ProgressMetricsResource(self.command_handler_service)
        )
        self.resources.register(
          ProgressHistoryResource(self.command_handler_service)
        )

        # Register session resources
        self.resources.register(
          SessionStateResource(self.command_handler_service)
        )

        # Register metrics resources
        self.resources.register(
          SessionMetricsResource(self.command_handler_service)
        )

        logger.info("Registered MCP resources")

    def _register_prompts(self) -> None:
        """Register MCP prompts."""
        # Register discovery workflow prompt
        self.prompts.register(DiscoverSourcesPrompt())

        # Register evaluation workflow prompt
        self.prompts.register(EvaluateSourcesPrompt())

        # Register acquisition workflow prompt
        self.prompts.register(AcquireDatasetsPrompt())

        # Register integration workflow prompt
        self.prompts.register(CreateIntegrationPlansPrompt())

        logger.info("Registered MCP prompts")



