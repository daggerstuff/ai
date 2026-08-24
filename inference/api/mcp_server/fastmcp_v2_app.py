"""FastMCP v2 entrypoint for Pixelated Memory."""

from __future__ import annotations

import logging
import sys

from fastmcp import FastMCP

from ai.inference.api.mcp_server.fastmcp_context import register_context_surfaces
from ai.inference.api.mcp_server.fastmcp_v2_tools import register_memory_tools_v2

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("Pixelated Memory")

register_context_surfaces(mcp)
register_memory_tools_v2(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")
