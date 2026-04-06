import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ai.api.mcp_server import fastmcp_context, fastmcp_tools
from ai.api.mcp_server.fastmcp_shared import AuthorizedToolContext
from ai.api.mcp_server.memory_scope import scope_from_kwargs
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


def _auth_context() -> str:
    return json.dumps({
        "actor_id": "api-server",
        "timestamp": "1710000000",
        "nonce": "test-nonce",
        "signature": "test-signature",
    })


def _scope_context(*, project_id: str | None = None) -> str:
    payload: dict[str, object] = {}
    if project_id is not None:
        payload["project_id"] = project_id
    return json.dumps(payload)


def _authorized_context(manager, *, user_id: str, project_id: str | None = None) -> AuthorizedToolContext:
    return AuthorizedToolContext(
        manager=manager,
        scope=scope_from_kwargs(
            user_id=user_id,
            project_id=project_id,
        ),
    )


def test_memory_store_persists_across_tool_calls_with_fallback_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback state should survive across tool calls within one server process."""
    manager = NullMemoryManager()
    monkeypatch.setattr(
        fastmcp_tools,
        "authorized_tool_context_from_json",
        lambda **kwargs: _authorized_context(manager, user_id=kwargs["user_id"]),
    )
    monkeypatch.setattr(
        fastmcp_tools,
        "authorized_tool_context_from_parts",
        lambda **kwargs: _authorized_context(manager, user_id=kwargs["user_id"]),
    )
    monkeypatch.setattr(
        fastmcp_context,
        "authorized_tool_context_from_json",
        lambda **kwargs: _authorized_context(manager, user_id=kwargs["user_id"]),
    )

    store_result = asyncio.run(
        fastmcp_app.memory_store(
            content="Baseline project checkpoint",
            user_id="vivi",
            category="project_context",
            auth_context=_auth_context(),
        )
    )
    assert "Memory Secured" in store_result

    status_result = asyncio.run(
        fastmcp_app.memory_status(
            user_id="vivi",
            auth_context=_auth_context(),
        )
    )

    assert "Total Memories:** 1" in status_result
    assert "**Health:** Healthy" in status_result


def test_memory_query_applies_limit_without_manager_limit_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managers matching BaseMemoryManager should still support query limits."""
    manager = NullMemoryManager()
    manager.add_memory("project alpha", "vivi")
    manager.add_memory("project beta", "vivi")

    monkeypatch.setattr(
        fastmcp_tools,
        "authorized_tool_context_from_json",
        lambda **kwargs: _authorized_context(manager, user_id=kwargs["user_id"]),
    )

    result = asyncio.run(
        fastmcp_app.memory_query(
            query="project",
            user_id="vivi",
            limit=1,
            auth_context=_auth_context(),
        )
    )

    assert "Error querying memory" not in result
    assert result.count("\n- [") == 1


def test_memory_query_refills_candidates_after_scope_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyManager:
        pass

    attempts: list[int] = []

    def fake_search_with_overfetch(*, manager, query: str, user_id: str, requested_limit: int, scope):
        del manager, query, user_id
        attempts.append(requested_limit)
        assert scope.project_id == "pixelated"
        if requested_limit == 2:
            return []
        return [
            {"memory": "project one", "score": 0.8, "metadata": {"project_id": "pixelated"}},
            {"memory": "project two", "score": 0.7, "metadata": {"project_id": "pixelated"}},
        ]

    monkeypatch.setattr(
        fastmcp_tools,
        "authorized_tool_context_from_json",
        lambda **kwargs: _authorized_context(
            DummyManager(),
            user_id=kwargs["user_id"],
            project_id=json.loads(kwargs["scope_context"]).get("project_id") if kwargs.get("scope_context") else None,
        ),
    )
    monkeypatch.setattr("ai.api.mcp_server.fastmcp_search.search_with_overfetch", fake_search_with_overfetch)

    result = asyncio.run(
        fastmcp_app.memory_query(
            query="project",
            user_id="vivi",
            limit=2,
            scope_context=_scope_context(project_id="pixelated"),
            auth_context=_auth_context(),
        )
    )

    assert "Error querying memory" not in result
    assert "project one" in result
    assert "project two" in result
    assert attempts == [10]


def test_fastmcp_surface_exposes_only_memory_primitives() -> None:
    assert hasattr(fastmcp_app, "memory_store")
    assert hasattr(fastmcp_app, "memory_query")
    assert hasattr(fastmcp_app, "memory_update")
    assert hasattr(fastmcp_app, "memory_delete")
    assert hasattr(fastmcp_app, "memory_status")
    assert not hasattr(fastmcp_app, "memory_analyze")
    assert not hasattr(fastmcp_app, "memory_sync_workspace")
    assert not hasattr(fastmcp_app, "get_memory_context")
    assert not hasattr(fastmcp_app, "session_start")
