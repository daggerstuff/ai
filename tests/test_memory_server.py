from fastapi.testclient import TestClient

from ai.api.memory.null_memory import NullMemoryManager
from ai.api.mcp_server import memory_auth
from ai.api.mcp_server.routes import _sanitize_user_profile_metadata
from ai.api.mcp_server.memory_server import create_memory_server
from ai.memory.local_hindsight_manager import LocalHindsightMemoryManager


def _configure_memory_auth(monkeypatch) -> None:
    memory_auth.configured_actor_tokens.cache_clear()
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv("LOCAL_MEMORY_ACTOR_TOKENS_JSON", '{"api-server":"actor-token"}')
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_POLICIES_JSON",
        '{"api-server":{"allowed_user_prefixes":["vivi","mallory","service-"]}}',
    )


def test_memory_server_get_all_route_returns_scoped_memories(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
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
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "vivi",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["memories"][0]["content"] == "project alpha"


def test_memory_server_exposes_hindsight_compatible_routes(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    headers = {
        "Authorization": "Bearer actor-token",
        "X-Memory-Actor-Id": "api-server",
        "X-Memory-User-Id": "vivi",
    }

    retain = client.post(
        "/v1/default/banks/pixelated/memories",
        json={
            "items": [
                {
                    "content": "Vivi prefers direct operational summaries",
                    "document_id": "doc-1",
                    "context": '{"user_id":"vivi","metadata":{"project_id":"pixelated"}}',
                    "tags": ["user:vivi", "project_id:pixelated"],
                }
            ]
        },
        headers=headers,
    )
    assert retain.status_code == 200
    assert retain.json()["results"][0]["id"] == "doc-1"

    recall = client.post(
        "/v1/default/banks/pixelated/memories/recall",
        json={"query": "operational summaries", "tags": ["user:vivi"]},
        headers=headers,
    )
    assert recall.status_code == 200
    payload = recall.json()
    assert payload["results"][0]["document_id"] == "doc-1"
    assert payload["results"][0]["text"] == "Vivi prefers direct operational summaries"

    listing = client.get("/v1/default/banks/pixelated/documents", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["items"][0]["id"] == "doc-1"

    document = client.get("/v1/default/banks/pixelated/documents/doc-1", headers=headers)
    assert document.status_code == 200
    assert document.json()["original_text"] == "Vivi prefers direct operational summaries"

    deleted = client.delete("/v1/default/banks/pixelated/documents/doc-1", headers=headers)
    assert deleted.status_code == 204


def test_hindsight_recall_treats_empty_tag_list_as_no_extra_filter(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    headers = {
        "Authorization": "Bearer actor-token",
        "X-Memory-Actor-Id": "api-server",
        "X-Memory-User-Id": "vivi",
    }

    retain = client.post(
        "/v1/default/banks/pixelated/memories",
        json={
            "items": [
                {
                    "content": "Vivi likes concise runbooks",
                    "document_id": "doc-empty-tags",
                    "context": '{"user_id":"vivi","metadata":{"project_id":"pixelated"}}',
                    "tags": ["project_id:pixelated"],
                }
            ]
        },
        headers=headers,
    )
    assert retain.status_code == 200

    recall = client.post(
        "/v1/default/banks/pixelated/memories/recall",
        json={"query": "concise runbooks", "tags": []},
        headers=headers,
    )
    assert recall.status_code == 200
    assert recall.json()["results"][0]["document_id"] == "doc-empty-tags"


def test_hindsight_document_routes_enforce_user_scope(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    auth = {"Authorization": "Bearer actor-token"}
    actor_headers = {**auth, "X-Memory-Actor-Id": "api-server"}

    retain = client.post(
        "/v1/default/banks/pixelated/memories",
        json={
            "items": [
                {
                    "content": "Vivi private memory",
                    "document_id": "doc-locked",
                    "context": '{"user_id":"mallory","metadata":{"project_id":"other"}}',
                    "tags": ["user:mallory", "project_id:other", "custom:tag"],
                }
            ]
        },
        headers={**actor_headers, "X-Memory-User-Id": "vivi"},
    )
    assert retain.status_code == 200

    other_user_get = client.get(
        "/v1/default/banks/pixelated/documents/doc-locked",
        headers={**actor_headers, "X-Memory-User-Id": "mallory"},
    )
    assert other_user_get.status_code == 404

    overwrite_attempt = client.post(
        "/v1/default/banks/pixelated/memories",
        json={
            "items": [
                {
                    "content": "Mallory overwrite attempt",
                    "document_id": "doc-locked",
                    "context": '{"user_id":"mallory","metadata":{"project_id":"other"}}',
                    "tags": ["user:mallory"],
                }
            ]
        },
        headers={**actor_headers, "X-Memory-User-Id": "mallory"},
    )
    assert overwrite_attempt.status_code == 404


def test_route_call_maps_internal_errors_to_500(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()

    class BrokenManager:
        def search_memories(self, query: str, user_id: str, limit: int = 10):
            raise RuntimeError("sqlite is on fire")

    app.state.memory_manager = BrokenManager()
    client = TestClient(app)
    response = client.post(
        "/api/memory/search",
        json={"query": "fire", "user_id": "vivi"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "vivi",
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal error while searching memory"


def test_memory_server_health_is_degraded_for_null_manager(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", "/tmp/pixelated-memory.db")
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["provider"] == "NullMemoryManager"
    assert payload["auth_model"] == "internal_service_actor_policies"
    assert payload["readiness"]["auth_configured"] is True
    assert payload["readiness"]["db_path_configured"] is True
    assert "db_path" not in payload["readiness"]
    assert payload["readiness"]["actor_policy_mode"] == "scoped"


def test_hindsight_retain_preserves_top_level_category_and_actor_metadata(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    headers = {
        "Authorization": "Bearer actor-token",
        "X-Memory-Actor-Id": "api-server",
        "X-Memory-User-Id": "vivi",
    }

    retain = client.post(
        "/v1/default/banks/pixelated/memories",
        json={
            "items": [
                {
                    "content": "Direct operational summary preference",
                    "document_id": "doc-category",
                    "context": '{"category":"preference","metadata":{"project_id":"pixelated"}}',
                    "tags": ["custom:tag"],
                }
            ]
        },
        headers=headers,
    )
    assert retain.status_code == 200

    stored = manager.get_memory("doc-category", user_id="vivi")
    assert stored is not None
    assert stored["metadata"]["category"] == "preference"
    assert stored["metadata"]["memory_actor_id"] == "api-server"


def test_mcp_routes_require_actor_identity(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.post(
        "/api/memory/search",
        json={"query": "hello", "user_id": "vivi"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-User-Id": "vivi",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Missing X-Memory-Actor-Id header"


def test_mcp_routes_require_user_header_to_match_scope(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.post(
        "/api/memory/search",
        json={"query": "hello", "user_id": "vivi"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "mallory",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "X-Memory-User-Id must match the requested user scope"


def test_memory_auth_rejects_unknown_actor(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.post(
        "/api/memory/search",
        json={"query": "hello", "user_id": "vivi"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "unknown-service",
            "X-Memory-User-Id": "vivi",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Unknown memory actor"


def test_memory_auth_rejects_actor_impersonation_outside_policy(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_POLICIES_JSON",
        '{"api-server":{"allowed_users":["vivi"]}}',
    )
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.post(
        "/api/memory/search",
        json={"query": "hello", "user_id": "mallory"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "mallory",
        },
    )

    assert response.status_code == 403
    assert "is not allowed to act for user" in response.json()["detail"]


def test_legacy_user_routes_require_user_scope_header(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()
    client = TestClient(app)

    create_response = client.post(
        "/api/memory/users",
        params={"email": "vivi", "name": "Vivi"},
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
        },
    )
    assert create_response.status_code == 400
    assert create_response.json()["detail"] == "Missing X-Memory-User-Id header"

    get_response = client.get(
        "/api/memory/users/vivi",
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "service-reader",
        },
    )
    assert get_response.status_code == 400
    assert get_response.json()["detail"] == "X-Memory-User-Id must match the requested user scope"


def test_legacy_create_user_persists_sanitized_profile_metadata(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = NullMemoryManager()
    app.state.memory_manager = manager
    client = TestClient(app)

    response = client.post(
        "/api/memory/users",
        params={
            "email": "vivi",
            "name": "Vivi",
            "role": "patient",
        },
        headers={
            "Authorization": "Bearer actor-token",
            "X-Memory-Actor-Id": "api-server",
            "X-Memory-User-Id": "vivi",
        },
    )

    assert response.status_code == 200
    stored = manager.get_all_memories("vivi", limit=10)
    assert len(stored) == 1
    metadata = stored[0]["metadata"]
    assert metadata["record_type"] == "user_registration"

    sanitized = _sanitize_user_profile_metadata({"timezone": "UTC", "is_admin": True})
    assert sanitized == {"timezone": "UTC"}
