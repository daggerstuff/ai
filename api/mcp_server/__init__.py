"""
MCP (Management Control Panel) Server for TechDeck-Python Pipeline Integration.

This module provides the main entry point for the MCP server, which serves as the
central orchestration hub for agent interactions in the TechDeck-Python pipeline
integration project.

Key Features:
- Agent registration and discovery
- Task delegation and status tracking
- Pipeline orchestration for 6-stage process
- WebSocket support for real-time communication
- Authentication and authorization for agent access
- Error handling and retry mechanisms
- Monitoring and logging capabilities
"""

__version__ = "1.0.0"
__author__ = "TechDeck Team"
__description__ = "MCP Server for Agent Interaction Management"

from .app import create_mcp_app
from .config import MCPConfig, get_mcp_config
from .core.agent_manager import AgentManager
from .core.task_orchestrator import TaskOrchestrator
from .core.pipeline_integration import PipelineIntegrationManager

__all__ = [
    "create_mcp_app",
    "MCPConfig",
    "get_mcp_config",
    "AgentManager",
    "TaskOrchestrator",
    "PipelineIntegrationManager",
]