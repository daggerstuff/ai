import hashlib
import hmac
import json
import time
import uuid

import pytest
from starlette.testclient import TestClient

from ai.api import index as api_index
from ai.api.mcp_server import memory_auth


def _configure_memory_auth(monkeypatch) -> None:
    memory_auth.configured_actor_tokens.cache_clear()
    memory_auth.configured_actor_policies.cache_clear()
    memory_auth.readiness_details.cache_clear()
    monkeypatch.setenv("LOCAL_MEMORY_ACTOR_TOKENS_JSON", '{"api-server":"actor-token"}')
    monkeypatch.setenv(
        "LOCAL_MEMORY_ACTOR_POLICIES_JSON",
        '{"api-server":{"allowed_users":["vivi"]}}',
    )


def _signed_reflect_request(
    client: TestClient,
    *,
    user_id: str,
    conversation_text: str,
    header_user_id: str | None = None,
    actor_id: str = "api-server",
    secret: str = "actor-token",
):
    payload = {"conversation_text": conversation_text, "user_id": user_id}
    encoded_body = json.dumps(payload).encode("utf-8")
    timestamp = str(int(time.time()))
    nonce = uuid.uuid4().hex
    scoped_user = header_user_id or user_id
    signature = hmac.new(
        secret.encode("utf-8"),
        memory_auth._canonical_request(
            actor_id=actor_id,
            user_id=scoped_user,
            method="POST",
            target="/reflect",
            body=encoded_body,
            timestamp=timestamp,
            nonce=nonce,
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return client.post(
        "/reflect",
        content=encoded_body,
        headers={
            "Content-Type": "application/json",
            "X-Memory-Actor-Id": actor_id,
            "X-Memory-User-Id": scoped_user,
            "X-Memory-Timestamp": timestamp,
            "X-Memory-Nonce": nonce,
            "X-Memory-Signature": signature,
        },
    )


class _DummyReflectionBootstrap:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def reflect_now(self, *, conversation_text: str, user_id: str):
        self.calls.append((conversation_text, user_id))

        class _Result:
            crisis_detected = False
            requires_manual_review = False
            memories_preserved = ["m1"]
            memories_consolidated = []

        return _Result()


def test_reflect_requires_signed_memory_actor_headers(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    api_index._reflection_bootstrap = _DummyReflectionBootstrap()
    client = TestClient(api_index.app)

    response = client.post(
        "/reflect",
        json={"conversation_text": "hello", "user_id": "vivi"},
        headers={"X-Memory-User-Id": "vivi"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Missing X-Memory-Actor-Id header"


def test_reflection_startup_failure_propagates(monkeypatch) -> None:
    async def fail_startup():
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(api_index, "create_and_start", fail_startup)

    with pytest.raises(RuntimeError, match="bootstrap failed"), TestClient(api_index.app):
        pass


def test_reflect_requires_header_user_scope_to_match_body(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    api_index._reflection_bootstrap = _DummyReflectionBootstrap()
    client = TestClient(api_index.app)

    response = _signed_reflect_request(
        client,
        user_id="vivi",
        header_user_id="mallory",
        conversation_text="reflect this",
    )

    assert response.status_code == 400
    assert response.json()["error"] == "X-Memory-User-Id must match the requested user scope"


def test_reflect_rejects_null_user_id(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    api_index._reflection_bootstrap = _DummyReflectionBootstrap()
    client = TestClient(api_index.app)

    response = client.post(
        "/reflect",
        json={"conversation_text": "reflect this", "user_id": None},
        headers={"X-Memory-User-Id": "vivi"},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "user_id required"


def test_reflect_uses_signed_user_scope(monkeypatch) -> None:
    _configure_memory_auth(monkeypatch)
    bootstrap = _DummyReflectionBootstrap()
    api_index._reflection_bootstrap = bootstrap
    client = TestClient(api_index.app)

    response = _signed_reflect_request(
        client,
        user_id="vivi",
        conversation_text="retain this reflection",
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert bootstrap.calls == [("retain this reflection", "vivi")]
