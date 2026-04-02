# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "mcp>=1.26.0",
#   "fastmcp>=2.3.3",
#   "pydantic>=2.11.7",
# ]
# ///
"""
FastMCP entrypoint for Pixelated Memory.

This file intentionally stays thin: it wires together the shared memory MCP
surface from dedicated modules instead of mixing storage primitives, prompts,
resources, and extra orchestration in one place.
"""

import logging

from mcp.server.fastmcp import FastMCP

from ai.api.mcp_server.fastmcp_context import register_context_surfaces
from ai.api.mcp_server.fastmcp_tools import register_memory_tools

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("Pixelated Memory", dependencies=["pydantic"])

register_context_surfaces(mcp)
register_memory_tools(mcp)


if __name__ == "__main__":
    mcp.run()
