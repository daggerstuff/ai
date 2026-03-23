import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from ai.api.memory.null_memory import NullMemoryManager


def _load_fastmcp_app_module():
    module_name = "test_fastmcp_app_module"
    module_path = Path(__file__).resolve().parents[1] / "api" / "mcp_server" / "fastmcp_app.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


fastmcp_app = _load_fastmcp_app_module()


def test_memory_store_persists_across_tool_calls_with_fallback_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback state should survive across tool calls within one server process."""
    created_managers: list[NullMemoryManager] = []

    def make_manager() -> NullMemoryManager:
        manager = NullMemoryManager()
        created_managers.append(manager)
        return manager

    monkeypatch.setattr(fastmcp_app, "get_memory_manager", make_manager)
    monkeypatch.setattr(fastmcp_app, "_manager_instance", None, raising=False)

    store_result = asyncio.run(
        fastmcp_app.memory_store(
            content="Baseline project checkpoint",
            user_id="vivi",
            category="project_context",
        )
    )
    assert "Memory Secured" in store_result

    status_result = asyncio.run(fastmcp_app.memory_status("vivi"))

    assert "Total Anchors:** 1" in status_result
    assert len(created_managers) == 1


def test_memory_query_applies_limit_without_manager_limit_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managers matching BaseMemoryManager should still support query limits."""
    manager = NullMemoryManager()
    manager.add_memory("project alpha", "vivi")
    manager.add_memory("project beta", "vivi")

    monkeypatch.setattr(fastmcp_app, "get_memory_manager", lambda: manager)
    monkeypatch.setattr(fastmcp_app, "_manager_instance", None, raising=False)

    result = asyncio.run(fastmcp_app.memory_query("project", "vivi", limit=1))

    assert "Error querying memory" not in result
    assert result.count("\n- [") == 1


def test_memory_analyze_supports_async_ai_capable_manager_without_sync_client_assumptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analysis should work with async-capable managers such as enhanced NVIDIA."""

    class AsyncAnalysisManager(NullMemoryManager):
        async def generate(self, prompt: str, **kwargs) -> str:
            return f"analysis generated for: {prompt[:40]}"

    manager = AsyncAnalysisManager()
    manager.add_memory(
        "We are fixing the NVIDIA memory manager interface mismatch.",
        "vivi",
        {"category": "project_context"},
    )

    monkeypatch.setattr(fastmcp_app, "get_memory_manager", lambda: manager)
    monkeypatch.setattr(fastmcp_app, "_manager_instance", None, raising=False)

    result = asyncio.run(fastmcp_app.memory_analyze("vivi", mode="themes"))

    assert "Analysis requires an AI-capable memory manager" not in result
    assert "Memory Analysis (themes): vivi" in result
    assert "analysis generated for:" in result
