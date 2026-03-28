from fastapi.testclient import TestClient

from ai.api.memory.null_memory import NullMemoryManager
from ai.api.mcp_server.memory_server import create_memory_server


def test_memory_server_get_all_route_returns_scoped_memories() -> None:
    app = create_memory_server()
    manager = NullMemoryManager()
    manager.add_memory(
        "project alpha",
        "vivi",
        metadata={"visibility": "private", "project_id": "pixelated"},
    )
    manager.add_memory(
        "project beta",
        "other",
        metadata={"visibility": "private", "project_id": "pixelated"},
    )
    app.state.memory_manager = manager

    client = TestClient(app)
    response = client.get(
        "/api/memory/all/vivi",
        params={"project_id": "pixelated"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["memories"][0]["content"] == "project alpha"
