"""Integration tests for ai/research/api.py — PIX-510 Task 4."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai.research.api import (
    MemoryStore,
    app,
    get_classifier,
    get_scorer,
    get_store,
)
from ai.research.emotion_classifier import EmotionClassifier
from ai.research.importance_scorer import ImportanceScorer

# Shared store across all tests so each test can see data created by others
_shared_store = MemoryStore()
_shared_scorer = ImportanceScorer.from_env()
_shared_classifier = EmotionClassifier(mode="lexicon")

app.dependency_overrides[get_store] = lambda: _shared_store
app.dependency_overrides[get_scorer] = lambda: _shared_scorer
app.dependency_overrides[get_classifier] = lambda: _shared_classifier


@pytest.fixture(autouse=True)
def clear_store():
    """Reset the shared store before each test."""
    _shared_store._store.clear()
    _shared_store._index.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ─── Health ──────────────────────────────────────────────────────────────────


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "memory_count" in data
    assert "scorer_latency_ms" in data
    assert "classifier_latency_ms" in data


# ─── Create memory ─────────────────────────────────────────────────────────────


def test_create_memory_returns_201(client: TestClient) -> None:
    response = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Client expressed anxiety about test results"},
    )
    assert response.status_code == 201
    block = response.json()
    assert block["id"].startswith("mem_")
    assert block["tenantId"] == "t1"
    assert block["sessionId"] == "s1"
    assert block["content"] == "Client expressed anxiety about test results"
    assert "importance" in block
    assert "emotions" in block
    assert "gating" in block


def test_create_memory_auto_classifies_fear(client: TestClient) -> None:
    response = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "I feel really scared and anxious about my results"},
    )
    assert response.status_code == 201
    block = response.json()
    assert "fear" in block["emotions"]["categories"]


def test_create_memory_crisis_flag(client: TestClient) -> None:
    response = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "I want to hurt myself"},
    )
    assert response.status_code == 201
    block = response.json()
    assert block["gating"]["crisisFlag"] is True


def test_create_memory_with_custom_emotions(client: TestClient) -> None:
    response = client.post(
        "/memories",
        json={
            "tenantId": "t1",
            "sessionId": "s1",
            "content": "Test content",
            "emotions": {"valence": 0.8, "arousal": 0.6, "categories": ["joy"]},
        },
    )
    assert response.status_code == 201
    block = response.json()
    assert block["emotions"]["valence"] == 0.8
    assert block["emotions"]["categories"] == ["joy"]


# ─── Get memory ────────────────────────────────────────────────────────────────


def test_get_memory_returns_block(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Get this block"},
    )
    mem_id = create_resp.json()["id"]

    response = client.get(f"/memories/{mem_id}?tenant_id=t1")
    assert response.status_code == 200
    assert response.json()["id"] == mem_id


def test_get_memory_not_found(client: TestClient) -> None:
    response = client.get("/memories/nonexistent?tenant_id=t1")
    assert response.status_code == 404


def test_get_memory_wrong_tenant(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Private memory"},
    )
    mem_id = create_resp.json()["id"]

    response = client.get(f"/memories/{mem_id}?tenant_id=t2")
    assert response.status_code == 404  # tenant isolation


# ─── Search memories ──────────────────────────────────────────────────────────


def test_search_returns_memories(client: TestClient) -> None:
    for i in range(3):
        client.post(
            "/memories",
            json={"tenantId": "t1", "sessionId": f"s{i}", "content": f"Memory {i}"},
        )

    response = client.get("/memories?tenant_id=t1&limit=10")
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3


def test_search_by_session_id(client: TestClient) -> None:
    client.post("/memories", json={"tenantId": "t1", "sessionId": "s1", "content": "Session 1"})
    client.post("/memories", json={"tenantId": "t1", "sessionId": "s2", "content": "Session 2"})

    response = client.get("/memories?tenant_id=t1&session_id=s1")
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["sessionId"] == "s1"


def test_search_by_min_importance(client: TestClient) -> None:
    # Create one high-importance and one low-importance block
    client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Crisis: I want to hurt myself"},
    )
    client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Neutral note about weather"},
    )

    response = client.get("/memories?tenant_id=t1&crisis_only=true")
    assert response.status_code == 200
    results = response.json()
    assert all(r["gating"]["crisisFlag"] for r in results)


def test_search_pagination(client: TestClient) -> None:
    for i in range(5):
        client.post("/memories", json={"tenantId": "t1", "sessionId": "s1", "content": f"Mem {i}"})

    first_page = client.get("/memories?tenant_id=t1&limit=2&offset=0")
    second_page = client.get("/memories?tenant_id=t1&limit=2&offset=2")

    assert len(first_page.json()) == 2
    assert len(second_page.json()) == 2


def test_search_empty_for_unknown_tenant(client: TestClient) -> None:
    response = client.get("/memories?tenant_id=unknown_tenant")
    assert response.status_code == 200
    assert response.json() == []


# ─── Update memory ─────────────────────────────────────────────────────────────


def test_update_content_rescores(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Neutral content"},
    )
    mem_id = create_resp.json()["id"]
    old_score = create_resp.json()["importance"]["raw"]

    update_resp = client.patch(
        f"/memories/{mem_id}?tenant_id=t1",
        json={"content": "I am terrified and in crisis about my health"},
    )
    assert update_resp.status_code == 200
    new_score = update_resp.json()["importance"]["raw"]
    assert new_score > old_score  # crisis content should score higher


def test_update_importance(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Test"},
    )
    mem_id = create_resp.json()["id"]

    update_resp = client.patch(f"/memories/{mem_id}?tenant_id=t1", json={"importance": 0.99})
    assert update_resp.status_code == 200
    assert update_resp.json()["importance"]["raw"] == 0.99


# ─── Delete memory ─────────────────────────────────────────────────────────────


def test_delete_memory_returns_204(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Delete me"},
    )
    mem_id = create_resp.json()["id"]

    response = client.delete(f"/memories/{mem_id}?tenant_id=t1")
    assert response.status_code == 204

    # Verify gone
    get_resp = client.get(f"/memories/{mem_id}?tenant_id=t1")
    assert get_resp.status_code == 404


def test_delete_memory_not_found(client: TestClient) -> None:
    response = client.delete("/memories/nonexistent?tenant_id=t1")
    assert response.status_code == 404


# ─── Score endpoint ──────────────────────────────────────────────────────────


def test_score_memory(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Normal content"},
    )
    mem_id = create_resp.json()["id"]

    response = client.post("/memories/score?tenant_id=t1", json={"memory_id": mem_id})
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == mem_id
    assert "importance" in data
    assert "components" in data
    assert "recency" in data["components"]
    assert "relevance" in data["components"]


def test_score_with_context(client: TestClient) -> None:
    create_resp = client.post(
        "/memories",
        json={"tenantId": "t1", "sessionId": "s1", "content": "Discussion about anxiety management techniques"},
    )
    mem_id = create_resp.json()["id"]

    no_ctx = client.post("/memories/score?tenant_id=t1&context=", json={"memory_id": mem_id})
    with_ctx = client.post(
        "/memories/score?tenant_id=t1&context=anxiety management therapy coping",
        json={"memory_id": mem_id},
    )

    assert no_ctx.status_code == 200
    assert with_ctx.status_code == 200
    # With matching context, relevance component should be higher
    assert with_ctx.json()["components"]["relevance"] >= no_ctx.json()["components"]["relevance"]


# ─── Trajectory endpoint ──────────────────────────────────────────────────────


def test_trajectory(client: TestClient) -> None:
    # Create a session
    for content in [
        "I am feeling okay today",
        "I am getting a bit worried",
        "I am feeling anxious and scared",
        "I feel panic rising",
    ]:
        client.post(
            "/memories",
            json={"tenantId": "t1", "sessionId": "s_trajectory", "content": content},
        )

    response = client.get("/memories/trajectory/s_trajectory?tenant_id=t1")
    assert response.status_code == 200
    data = response.json()
    assert data["sessionId"] == "s_trajectory"
    assert data["memoryCount"] == 4
    assert data["trend"] in ("escalating", "volatile", "stable")
    assert isinstance(data["trajectory"], list)
    assert len(data["trajectory"]) == 4


def test_trajectory_empty_session(client: TestClient) -> None:
    response = client.get("/memories/trajectory/unknown_session?tenant_id=t1")
    assert response.status_code == 200
    data = response.json()
    assert data["trend"] == "stable"
    assert data["memoryCount"] == 0
