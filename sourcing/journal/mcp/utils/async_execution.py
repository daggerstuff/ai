"""
Async tool execution handler for MCP Server.

This module provides async tool execution with operation status tracking,
cancellation support, and timeout handling.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ai.sourcing.journal.mcp.protocol import MCPError, MCPErrorCode
from ai.sourcing.journal.mcp.utils.progress_streaming import (
    ProgressStatus,
    ProgressStreamer,
    ProgressUpdate,
)

logger = logging.getLogger(__name__)


class OperationCancelledError(Exception):
    """Raised when an operation is cancelled."""

    pass


class AsyncToolExecutor:
    """
    Async tool execution handler with operation tracking.

    This class manages async tool execution with:
    - Operation status tracking
    - Operation cancellation
    - Timeout handling
    - Progress update integration
    """

    def __init__(
        self,
        progress_streamer: ProgressStreamer,
        default_timeout: Optional[float] = None,
    ) -> None:
        """
        Initialize async tool executor.

        Args:
            progress_streamer: Progress streamer instance
            default_timeout: Default timeout in seconds (None for no timeout)
        """
        self.progress_streamer = progress_streamer
        self.default_timeout = default_timeout
        # Active operations: operation_id -> asyncio.Task
        self._active_operations: Dict[str, asyncio.Task] = {}
        # Operation metadata: operation_id -> dict
        self._operation_metadata: Dict[str, Dict[str, Any]] = {}
        # Cancellation flags: operation_id -> asyncio.Event
        self._cancellation_flags: Dict[str, asyncio.Event] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def execute_async(
        self,
        operation_id: str,
        tool_name: str,
        tool_executor: Callable,
        params: Dict[str, Any],
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute tool asynchronously with operation tracking.

        Args:
            operation_id: Unique operation ID
            tool_name: Name of tool being executed
            tool_executor: Async function to execute the tool
            params: Tool parameters
            session_id: Optional session ID
            timeout: Optional timeout in seconds (uses default if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            Tool execution result

        Raises:
            MCPError: If execution fails or is cancelled
            OperationCancelledError: If operation is cancelled
        """
        # Use provided timeout or default
        timeout = timeout or self.default_timeout

        # Create cancellation flag
        cancellation_flag = asyncio.Event()
        async with self._lock:
            self._cancellation_flags[operation_id] = cancellation_flag
            self._operation_metadata[operation_id] = {
                "tool_name": tool_name,
                "session_id": session_id,
                "params": params,
                "started_at": datetime.now(),
                "status": ProgressStatus.PENDING,
            }

        # Subscribe to progress updates if callback provided
        if progress_callback:
            await self.progress_streamer.subscribe(
                operation_id, progress_callback, session_id=session_id
            )

        try:
            # Send initial progress update
            await self._send_progress_update(
                operation_id,
                session_id or "unknown",
                ProgressStatus.RUNNING,
                0.0,
                f"Starting {tool_name}",
            )

            # Create execution task
            execution_task = asyncio.create_task(
                self._execute_with_progress(
                    operation_id,
                    tool_name,
                    tool_executor,
                    params,
                    session_id,
                    cancellation_flag,
                )
            )

            async with self._lock:
                self._active_operations[operation_id] = execution_task

            # Wait for completion with optional timeout
            try:
                if timeout:
                    result = await asyncio.wait_for(execution_task, timeout=timeout)
                else:
                    result = await execution_task

                # Send completion update
                await self._send_progress_update(
                    operation_id,
                    session_id or "unknown",
                    ProgressStatus.COMPLETED,
                    100.0,
                    f"{tool_name} completed successfully",
                )

                return result

            except asyncio.TimeoutError:
                # Cancel operation on timeout
                await self.cancel_operation(operation_id)
                raise MCPError(
                    MCPErrorCode.TOOL_TIMEOUT,
                    f"Operation {operation_id} timed out after {timeout} seconds",
                    {"operation_id": operation_id, "timeout": timeout},
                )

            except OperationCancelledError:
                await self._send_progress_update(
                    operation_id,
                    session_id or "unknown",
                    ProgressStatus.CANCELLED,
                    0.0,
                    f"{tool_name} was cancelled",
                )
                raise MCPError(
                    MCPErrorCode.TOOL_EXECUTION_ERROR,
                    f"Operation {operation_id} was cancelled",
                    {"operation_id": operation_id},
                )

        except MCPError:
            # Re-raise MCP errors
            raise
        except Exception as e:
            # Send failure update
            await self._send_progress_update(
                operation_id,
                session_id or "unknown",
                ProgressStatus.FAILED,
                0.0,
                f"{tool_name} failed: {str(e)}",
                {"error": str(e), "error_type": type(e).__name__},
            )
            logger.exception(f"Error executing async tool {tool_name}")
            raise MCPError(
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                f"Tool execution failed: {str(e)}",
                {"tool_name": tool_name, "operation_id": operation_id, "error": str(e)},
            ) from e

        finally:
            # Cleanup
            async with self._lock:
                self._active_operations.pop(operation_id, None)
                self._cancellation_flags.pop(operation_id, None)
                if progress_callback:
                    await self.progress_streamer.unsubscribe(
                        operation_id, progress_callback, session_id=session_id
                    )

    async def _execute_with_progress(
        self,
        operation_id: str,
        tool_name: str,
        tool_executor: Callable,
        params: Dict[str, Any],
        session_id: Optional[str],
        cancellation_flag: asyncio.Event,
    ) -> Dict[str, Any]:
        """
        Execute tool with progress tracking and cancellation support.

        Args:
            operation_id: Operation ID
            tool_name: Tool name
            tool_executor: Tool executor function
            params: Tool parameters
            session_id: Optional session ID
            cancellation_flag: Cancellation flag

        Returns:
            Tool execution result

        Raises:
            OperationCancelledError: If operation is cancelled
        """
        # Check for cancellation before starting
        if cancellation_flag.is_set():
            raise OperationCancelledError(f"Operation {operation_id} was cancelled")

        # Execute tool
        # If tool_executor accepts a progress_callback, pass it
        # Otherwise, just call it directly
        try:
            # Check if tool_executor accepts progress_callback parameter
            import inspect

            sig = inspect.signature(tool_executor)
            if "progress_callback" in sig.parameters:
                # Create progress callback wrapper
                async def progress_wrapper(progress: float, message: str = "") -> None:
                    if cancellation_flag.is_set():
                        raise OperationCancelledError(f"Operation {operation_id} was cancelled")
                    await self._send_progress_update(
                        operation_id,
                        session_id or "unknown",
                        ProgressStatus.RUNNING,
                        progress,
                        message or f"{tool_name} in progress",
                    )

                result = await tool_executor(params, progress_callback=progress_wrapper)
            else:
                # No progress callback support, execute directly
                result = await tool_executor(params)

            # Check for cancellation after execution
            if cancellation_flag.is_set():
                raise OperationCancelledError(f"Operation {operation_id} was cancelled")

            return result

        except OperationCancelledError:
            raise
        except Exception as e:
            # Check if cancellation was requested during execution
            if cancellation_flag.is_set():
                raise OperationCancelledError(f"Operation {operation_id} was cancelled") from e
            raise

    async def _send_progress_update(
        self,
        operation_id: str,
        session_id: str,
        status: ProgressStatus,
        progress_percent: float,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Send progress update.

        Args:
            operation_id: Operation ID
            session_id: Session ID
            status: Progress status
            progress_percent: Progress percentage (0-100)
            message: Progress message
            metadata: Optional metadata
        """
        update = ProgressUpdate(
            operation_id=operation_id,
            session_id=session_id,
            status=status,
            progress_percent=progress_percent,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        await self.progress_streamer.broadcast_update(update)

        # Update operation metadata
        async with self._lock:
            if operation_id in self._operation_metadata:
                self._operation_metadata[operation_id]["status"] = status
                self._operation_metadata[operation_id]["last_update"] = datetime.now()

    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel an active operation.

        Args:
            operation_id: Operation ID to cancel

        Returns:
            True if operation was cancelled, False if not found
        """
        async with self._lock:
            if operation_id not in self._cancellation_flags:
                return False

            # Set cancellation flag
            self._cancellation_flags[operation_id].set()

            # Cancel task if running
            if operation_id in self._active_operations:
                task = self._active_operations[operation_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        logger.info(f"Cancelled operation {operation_id}")
        return True

    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get operation status and metadata.

        Args:
            operation_id: Operation ID

        Returns:
            Operation metadata or None if not found
        """
        async with self._lock:
            metadata = self._operation_metadata.get(operation_id)
            if not metadata:
                return None

            # Get current status from progress streamer
            status = await self.progress_streamer.get_operation_status(operation_id)
            if status:
                metadata["status"] = status

            # Check if task is still running
            if operation_id in self._active_operations:
                task = self._active_operations[operation_id]
                metadata["task_done"] = task.done()
                if task.done():
                    try:
                        metadata["result"] = task.result()
                    except Exception as e:
                        metadata["error"] = str(e)

            return metadata.copy()

    async def list_active_operations(self) -> List[str]:
        """
        List all active operation IDs.

        Returns:
            List of active operation IDs
        """
        async with self._lock:
            return [
                op_id
                for op_id, task in self._active_operations.items()
                if not task.done()
            ]

    def generate_operation_id(self, prefix: str = "op") -> str:
        """
        Generate a unique operation ID.

        Args:
            prefix: Prefix for operation ID

        Returns:
            Unique operation ID
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

