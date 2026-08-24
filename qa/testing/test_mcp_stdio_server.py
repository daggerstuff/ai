import asyncio

from ai.inference.api.mcp_server import mcp_stdio_server
from ai.inference.api.memory.null_memory import NullMemoryManager


def test_mcp_stdio_server_uses_shared_memory_manager(monkeypatch) -> None:
    manager = NullMemoryManager()

    monkeypatch.setattr(mcp_stdio_server, "_memory_client", None, raising=False)
    monkeypatch.setattr(mcp_stdio_server, "get_required_memory_manager", lambda: manager)

    add_result = asyncio.run(
        mcp_stdio_server.call_tool(
            "add_memory",
            {"content": "shared memory path", "user_id": "vivi"},
        )
    )
    assert "Memory stored successfully" in add_result[0].text

    search_result = asyncio.run(
        mcp_stdio_server.call_tool(
            "search_memory",
            {"query": "shared", "user_id": "vivi", "limit": 5},
        )
    )
    assert "Found 1 memories" in search_result[0].text
