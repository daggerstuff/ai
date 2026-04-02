import json
import time
import uuid

import pytest
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


def _configure_memory_auth_with_compat(monkeypatch) -> None:
    memory_auth.configured_actor_tokens.cache_clear()
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_TOKENS_JSON",
        '{"api-server":"actor-token","local-hindsight-cli":"compat-token"}',
    )
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_POLICIES_JSON",
        (
            '{"api-server":{"allowed_user_prefixes":["vivi","mallory","service-"]},'
            '"local-hindsight-cli":{"allowed_users":["vivi"]}}'
        ),
    )
    monkeypatch.setenv("HINDSIGHT_COMPAT_ENABLE_BEARER", "true")
    monkeypatch.setenv("HINDSIGHT_COMPAT_BEARER_ACTOR_ID", "local-hindsight-cli")
    monkeypatch.setenv("HINDSIGHT_COMPAT_DEFAULT_USER_ID", "vivi")


def _signed_request(
    client: TestClient,
    method: str,
    path: str,
    *,
    user_id: str,
    json_body=None,
    params=None,
    actor_id: str = "api-server",
    secret: str = "actor-token",
    timestamp: str | None = None,
    nonce: str | None = None,
):
    encoded_body = b""
    headers = {
        "X-Memory-Actor-Id": actor_id,
        "X-Memory-User-Id": user_id,
    }
    if json_body is not None:
        encoded_body = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if params:
        from urllib.parse import urlencode

        target = f"{path}?{urlencode(params, doseq=True)}"
    else:
        target = path
    timestamp_value = timestamp or str(int(time.time()))
    nonce_value = nonce or uuid.uuid4().hex
    headers["X-Memory-Timestamp"] = timestamp_value
    headers["X-Memory-Nonce"] = nonce_value
    headers["X-Memory-Signature"] = memory_auth.hmac.new(
        secret.encode("utf-8"),
        memory_auth._canonical_request(
            actor_id=actor_id,
            user_id=user_id,
            method=method,
            target=target,
            body=encoded_body,
            timestamp=timestamp_value,
            nonce=nonce_value,
        ).encode("utf-8"),
        memory_auth.hashlib.sha256,
    ).hexdigest()
    return client.request(
        method=method,
        url=path,
        params=params,
        content=encoded_body or None,
        headers=headers,
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
    response = _signed_request(
        client,
        "GET",
        "/api/memory/all/vivi",
        user_id="vivi",
        params={"project_id": "pixelated"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["count"] == 1
    assert payload["memories"][0]["content"] == "project alpha"


def test_memory_server_requires_explicit_actor_policy_config(monkeypatch, tmp_path) -> None:
    memory_auth.configured_actor_tokens.cache_clear()
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv("MEMORY_PROVIDER", "local_hindsight")
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", str(tmp_path / "local-hindsight.db"))
    monkeypatch.setenv("LOCAL_MEMORY_ACTOR_TOKENS_JSON", '{"api-server":"actor-token"}')
    monkeypatch.delenv("LOCAL_MEMORY_ACTOR_POLICIES_JSON", raising=False)

    app = create_memory_server()

    with pytest.raises(
        RuntimeError,
        match="LOCAL_MEMORY_ACTOR_POLICIES_JSON must be configured",
    ):
        with TestClient(app):
            pass


def test_memory_server_exposes_hindsight_compatible_routes(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    retain = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="vivi",
        json_body={
            "items": [
                {
                    "content": "Vivi prefers direct operational summaries",
                    "document_id": "doc-1",
                    "context": '{"user_id":"vivi","metadata":{"project_id":"pixelated"}}',
                    "tags": ["user:vivi", "project_id:pixelated"],
                }
            ]
        },
    )
    assert retain.status_code == 200
    assert retain.json()["results"][0]["id"] == "doc-1"

    recall = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories/recall",
        user_id="vivi",
        json_body={"query": "operational summaries", "tags": ["user:vivi"]},
    )
    assert recall.status_code == 200
    payload = recall.json()
    assert payload["results"][0]["document_id"] == "doc-1"
    assert payload["results"][0]["text"] == "Vivi prefers direct operational summaries"

    listing = _signed_request(
        client,
        "GET",
        "/v1/default/banks/pixelated/documents",
        user_id="vivi",
    )
    assert listing.status_code == 200
    assert listing.json()["items"][0]["id"] == "doc-1"

    document = _signed_request(
        client,
        "GET",
        "/v1/default/banks/pixelated/documents/doc-1",
        user_id="vivi",
    )
    assert document.status_code == 200
    assert document.json()["original_text"] == "Vivi prefers direct operational summaries"

    deleted = _signed_request(
        client,
        "DELETE",
        "/v1/default/banks/pixelated/documents/doc-1",
        user_id="vivi",
    )
    assert deleted.status_code == 204


def test_hindsight_recall_treats_empty_tag_list_as_no_extra_filter(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    retain = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="vivi",
        json_body={
            "items": [
                {
                    "content": "Vivi likes concise runbooks",
                    "document_id": "doc-empty-tags",
                    "context": '{"user_id":"vivi","metadata":{"project_id":"pixelated"}}',
                    "tags": ["project_id:pixelated"],
                }
            ]
        },
    )
    assert retain.status_code == 200

    recall = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories/recall",
        user_id="vivi",
        json_body={"query": "concise runbooks", "tags": []},
    )
    assert recall.status_code == 200
    assert recall.json()["results"][0]["document_id"] == "doc-empty-tags"


def test_hindsight_routes_accept_local_bearer_compatibility(tmp_path, monkeypatch) -> None:
    _configure_memory_auth_with_compat(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    retain = client.post(
        "/v1/default/banks/pixeldated/memories",
        json={
            "items": [
                {
                    "content": "Vivi stores local CLI memories in the shared service",
                    "document_id": "compat-doc-1",
                    "context": '{"metadata":{"project_id":"pixelated"}}',
                    "tags": ["project_id:pixelated"],
                }
            ]
        },
        headers={"Authorization": "Bearer compat-token"},
    )
    assert retain.status_code == 200
    assert retain.json()["results"][0]["id"] == "compat-doc-1"

    recall = client.post(
        "/v1/default/banks/pixeldated/memories/recall",
        json={"query": "local CLI memories"},
        headers={"Authorization": "Bearer compat-token"},
    )
    assert recall.status_code == 200
    assert recall.json()["results"][0]["document_id"] == "compat-doc-1"


def test_hindsight_document_routes_enforce_user_scope(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    retain = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="vivi",
        json_body={
            "items": [
                {
                    "content": "Vivi private memory",
                    "document_id": "doc-locked",
                    "context": '{"user_id":"vivi","metadata":{"project_id":"other"}}',
                    "tags": ["user:vivi", "project_id:other", "custom:tag"],
                }
            ]
        },
    )
    assert retain.status_code == 200

    other_user_get = _signed_request(
        client,
        "GET",
        "/v1/default/banks/pixelated/documents/doc-locked",
        user_id="mallory",
    )
    assert other_user_get.status_code == 404

    overwrite_attempt = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="mallory",
        json_body={
            "items": [
                {
                    "content": "Mallory overwrite attempt",
                    "document_id": "doc-locked",
                    "context": '{"user_id":"mallory","metadata":{"project_id":"other"}}',
                    "tags": ["user:mallory"],
                }
            ]
        },
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
    response = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="vivi",
        json_body={"query": "fire", "user_id": "vivi"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal error while searching memory"


def test_memory_server_health_is_healthy_for_null_manager(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", "/tmp/pixelated-memory.db")
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["provider"] == "NullMemoryManager"
    assert payload["auth_model"] == "internal_service_hmac_actor_policies"
    assert payload["readiness"]["auth_configured"] is True
    assert payload["readiness"]["db_path_configured"] is True
    assert "db_path" not in payload["readiness"]
    assert payload["readiness"]["actor_policy_mode"] == "scoped"
    assert payload["readiness"]["signature_required"] is True


def test_hindsight_retain_preserves_top_level_category_and_actor_metadata(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager

    client = TestClient(app)
    retain = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="vivi",
        json_body={
            "items": [
                {
                    "content": "Direct operational summary preference",
                    "document_id": "doc-category",
                    "context": '{"category":"preference","metadata":{"project_id":"pixelated"}}',
                    "tags": ["custom:tag"],
                }
            ]
        },
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
    response = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="mallory",
        json_body={"query": "hello", "user_id": "vivi"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "X-Memory-User-Id must match the requested user scope"


def test_memory_auth_rejects_unknown_actor(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()

    client = TestClient(app)
    response = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="vivi",
        actor_id="unknown-service",
        json_body={"query": "hello", "user_id": "vivi"},
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
    response = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="mallory",
        json_body={"query": "hello", "user_id": "mallory"},
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
            "X-Memory-Actor-Id": "api-server",
        },
    )
    assert create_response.status_code == 400
    assert create_response.json()["detail"] == "Missing X-Memory-User-Id header"

    get_response = _signed_request(
        client,
        "GET",
        "/api/memory/users/vivi",
        user_id="service-reader",
    )
    assert get_response.status_code == 400
    assert get_response.json()["detail"] == "X-Memory-User-Id must match the requested user scope"


def test_legacy_create_user_persists_sanitized_profile_metadata(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = NullMemoryManager()
    app.state.memory_manager = manager
    client = TestClient(app)

    response = _signed_request(
        client,
        "POST",
        "/api/memory/users",
        user_id="vivi",
        params={
            "email": "vivi",
            "name": "Vivi",
            "role": "patient",
        },
    )

    assert response.status_code == 200
    stored = manager.get_all_memories("vivi", limit=10)
    assert len(stored) == 1
    metadata = stored[0]["metadata"]
    assert metadata["record_type"] == "user_registration"

    sanitized = _sanitize_user_profile_metadata({"timezone": "UTC", "is_admin": True})
    assert sanitized == {"timezone": "UTC"}


def test_memory_auth_rejects_replayed_signed_request(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()
    client = TestClient(app)

    first = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="vivi",
        json_body={"query": "hello", "user_id": "vivi"},
        nonce="replay-nonce",
        timestamp=str(int(time.time())),
    )
    second = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="vivi",
        json_body={"query": "hello", "user_id": "vivi"},
        nonce="replay-nonce",
        timestamp=str(int(time.time())),
    )

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "Replay detected for signed memory request"


def test_memory_auth_rejects_stale_signed_request(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    app.state.memory_manager = NullMemoryManager()
    client = TestClient(app)

    response = _signed_request(
        client,
        "POST",
        "/api/memory/search",
        user_id="vivi",
        json_body={"query": "hello", "user_id": "vivi"},
        timestamp="1",
    )

    assert response.status_code == 401
    assert "outside the allowed window" in response.json()["detail"]


def test_hindsight_retain_rejects_conflicting_identity_payload(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    app = create_memory_server()
    manager = LocalHindsightMemoryManager(db_path=str(tmp_path / "local-hindsight.db"))
    app.state.memory_manager = manager
    client = TestClient(app)

    response = _signed_request(
        client,
        "POST",
        "/v1/default/banks/pixelated/memories",
        user_id="vivi",
        json_body={
            "items": [
                {
                    "content": "Conflicting scope should fail",
                    "document_id": "doc-conflict",
                    "context": '{"user_id":"mallory","metadata":{"project_id":"pixelated"}}',
                    "tags": ["user:mallory"],
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "does not match X-Memory-User-Id" in response.json()["detail"]


def test_memory_server_health_reports_local_db_probe(tmp_path, monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    db_path = tmp_path / "local-hindsight.db"
    monkeypatch.setenv("HINDSIGHT_LOCAL_DB_PATH", str(db_path))
    app = create_memory_server()
    app.state.memory_manager = LocalHindsightMemoryManager(db_path=str(db_path))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["readiness"]["db_ready"] is True
    assert payload["readiness"]["db_writable"] is True
    assert payload["readiness"]["db_quick_check"] == "ok"
