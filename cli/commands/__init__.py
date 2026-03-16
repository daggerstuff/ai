"""
Command modules for Pixelated AI CLI

This package contains all the command groups for the CLI tool:
- web_frontend_group: Web-based interface commands
- cli_interface_group: Direct CLI operations
- mcp_connect_group: Agent connection commands
- pipeline_group: Pipeline management commands
- config_group: Configuration management commands
- auth_group: Authentication commands
"""

from .web_frontend import web_frontend_group
from .cli_interface import cli_interface_group
from .mcp_connect import mcp_connect_group
from .pipeline import pipeline_group
from .config import config_group
from .auth import auth_group

__all__ = [
    'web_frontend_group',
    'cli_interface_group',
    'mcp_connect_group',
    'pipeline_group',
    'config_group',
    'auth_group',
]