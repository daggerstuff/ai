"""
MCP Protocol Handler.

This module implements the Model Context Protocol (MCP) over JSON-RPC 2.0.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Base exception for MCP protocol errors."""

    def __init__(
        self,
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> None:
        """
        Initialize MCP error.

        Args:
            code: Error code (JSON-RPC error codes or MCP-specific codes)
            message: Error message
            data: Optional error data
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# JSON-RPC 2.0 error codes
class JSONRPCErrorCode:
    """JSON-RPC 2.0 standard error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# MCP-specific error codes (extend JSON-RPC codes)
class MCPErrorCode:
    """MCP-specific error codes."""

    # Tool execution errors (start at -32000)
    TOOL_EXECUTION_ERROR = -32000
    TOOL_VALIDATION_ERROR = -32001
    TOOL_TIMEOUT = -32002

    # Resource access errors
    RESOURCE_NOT_FOUND = -32010
    RESOURCE_ACCESS_DENIED = -32011

    # Authentication/Authorization errors
    AUTHENTICATION_ERROR = -32020
    AUTHORIZATION_ERROR = -32021

    # Rate limiting errors
    RATE_LIMIT_EXCEEDED = -32030


class MCPRequest:
    """Represents an MCP protocol request."""

    def __init__(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        id: Optional[Union[str, int]] = None,
        jsonrpc: str = "2.0",
    ) -> None:
        """
        Initialize MCP request.

        Args:
            method: Method name (e.g., "tools/call", "resources/read")
            params: Method parameters
            id: Request ID (for matching responses)
            jsonrpc: JSON-RPC version (must be "2.0")
        """
        self.method = method
        self.params = params or {}
        self.id = id
        self.jsonrpc = jsonrpc

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        """
        Create MCPRequest from dictionary.

        Args:
            data: Request dictionary

        Returns:
            MCPRequest instance

        Raises:
            MCPError: If request is invalid
        """
        # Validate JSON-RPC version
        if data.get("jsonrpc") != "2.0":
            raise MCPError(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Invalid JSON-RPC version. Must be '2.0'",
            )

        # Validate method
        if "method" not in data:
            raise MCPError(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Missing required field: method",
            )

        method = data["method"]
        if not isinstance(method, str):
            raise MCPError(
                JSONRPCErrorCode.INVALID_REQUEST,
                "Method must be a string",
            )

        return cls(
            method=method,
            params=data.get("params"),
            id=data.get("id"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        result: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
        }
        if self.params:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result


class MCPResponse:
    """Represents an MCP protocol response."""

    def __init__(
        self,
        result: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
        id: Optional[Union[str, int]] = None,
        jsonrpc: str = "2.0",
    ) -> None:
        """
        Initialize MCP response.

        Args:
            result: Response result (for success)
            error: Error information (for failure)
            id: Request ID (must match request)
            jsonrpc: JSON-RPC version (must be "2.0")
        """
        if result is not None and error is not None:
            raise ValueError("Response cannot have both result and error")

        self.result = result
        self.error = error
        self.id = id
        self.jsonrpc = jsonrpc

    @classmethod
    def success(
        cls,
        result: Any,
        id: Optional[Union[str, int]] = None,
    ) -> "MCPResponse":
        """
        Create success response.

        Args:
            result: Response result
            id: Request ID

        Returns:
            MCPResponse instance
        """
        return cls(result=result, id=id)

    @classmethod
    def error(
        cls,
        code: int,
        message: str,
        data: Optional[Any] = None,
        id: Optional[Union[str, int]] = None,
    ) -> "MCPResponse":
        """
        Create error response.

        Args:
            code: Error code
            message: Error message
            data: Optional error data
            id: Request ID

        Returns:
            MCPResponse instance
        """
        error_dict: Dict[str, Any] = {
            "code": code,
            "message": message,
        }
        if data is not None:
            error_dict["data"] = data

        return cls(error=error_dict, id=id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        result: Dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
        }
        if self.error is not None:
            result["error"] = self.error
        else:
            result["result"] = self.result
        if self.id is not None:
            result["id"] = self.id
        return result


class MCPProtocolHandler:
    """Handles MCP protocol parsing and formatting."""

    @staticmethod
    def parse_request(data: Union[str, bytes, Dict[str, Any]]) -> MCPRequest:
        """
        Parse MCP request from JSON string, bytes, or dictionary.

        Args:
            data: Request data (JSON string, bytes, or dict)

        Returns:
            MCPRequest instance

        Raises:
            MCPError: If request cannot be parsed
        """
        try:
            # Parse JSON if needed
            if isinstance(data, (str, bytes)):
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    raise MCPError(
                        JSONRPCErrorCode.PARSE_ERROR,
                        f"Invalid JSON: {str(e)}",
                    ) from e

            if not isinstance(data, dict):
                raise MCPError(
                    JSONRPCErrorCode.INVALID_REQUEST,
                    "Request must be a JSON object",
                )

            return MCPRequest.from_dict(data)

        except MCPError:
            raise
        except Exception as e:
            logger.exception("Unexpected error parsing request")
            raise MCPError(
                JSONRPCErrorCode.INTERNAL_ERROR,
                f"Internal error: {str(e)}",
            ) from e

    @staticmethod
    def format_response(response: MCPResponse) -> str:
        """
        Format MCP response as JSON string.

        Args:
            response: MCPResponse instance

        Returns:
            JSON string representation
        """
        try:
            return json.dumps(response.to_dict())
        except Exception as e:
            logger.exception("Error formatting response")
            # Return error response
            error_response = MCPResponse.error(
                JSONRPCErrorCode.INTERNAL_ERROR,
                f"Error formatting response: {str(e)}",
            )
            return json.dumps(error_response.to_dict())

    @staticmethod
    def format_error(
        code: int,
        message: str,
        data: Optional[Any] = None,
        request_id: Optional[Union[str, int]] = None,
    ) -> str:
        """
        Format error response as JSON string.

        Args:
            code: Error code
            message: Error message
            data: Optional error data
            request_id: Request ID

        Returns:
            JSON string representation
        """
        response = MCPResponse.error(code, message, data, request_id)
        return MCPProtocolHandler.format_response(response)

