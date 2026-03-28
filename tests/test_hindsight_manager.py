from typing import Any, Dict, Tuple

from ai.memory.hindsight_manager import HindsightMemoryManager


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = b"{}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[Tuple[str, str, Dict[str, Any]]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.retain_params: Dict[str, str] = {}
        self.bank_id = "pixeldated"

    def request(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        timeout: float,
        **kwargs: Any,
    ) -> FakeResponse:
        self.calls.append((method, url, kwargs))
        path = url.split(f"/v1/default/banks/{self.bank_id}", 1)[1]

        if method == "POST" and path == "/memories":
            item = kwargs["json"]["items"][0]
            doc_id = item["document_id"]
            self.documents[doc_id] = {
                "id": doc_id,
                "bank_id": self.bank_id,
                "original_text": item["content"],
                "created_at": "2026-03-27T02:00:00+00:00",
                "updated_at": "2026-03-27T02:00:00+00:00",
                "tags": item.get("tags", []),
            }
            self.retain_params[doc_id] = item.get("context", "")
            return FakeResponse({"success": True})

        if method == "POST" and path == "/memories/recall":
            return FakeResponse(
                {
                    "results": [
                        {
                            "id": "unit-1",
                            "document_id": "doc-1",
                            "text": "remembered project preference",
                            "context": self.retain_params.get(
                                "doc-1",
                                '{"user_id":"vivi","metadata":{"visibility":"private","project_id":"pixelated"}}',
                            ),
                            "tags": [],
                            "mentioned_at": "2026-03-27T02:00:00+00:00",
                            "chunk_id": "chunk-1",
                        }
                    ]
                }
            )

        if method == "GET" and path == "/documents":
            items = []
            for doc_id, document in self.documents.items():
                items.append(
                    {
                        "id": doc_id,
                        "bank_id": self.bank_id,
                        "created_at": document["created_at"],
                        "updated_at": document["updated_at"],
                        "tags": document["tags"],
                        "retain_params": '{"context":%s}' % self._json_string(self.retain_params[doc_id]),
                    }
                )
            return FakeResponse({"items": items})

        if method == "GET" and path.startswith("/documents/"):
            doc_id = path.rsplit("/", 1)[1]
            document = self.documents.get(doc_id)
            if document is None:
                return FakeResponse({}, status_code=404)
            return FakeResponse(document)

        if method == "DELETE" and path.startswith("/documents/"):
            doc_id = path.rsplit("/", 1)[1]
            self.documents.pop(doc_id, None)
            self.retain_params.pop(doc_id, None)
            return FakeResponse({"success": True})

        raise AssertionError(f"Unhandled request: {method} {path}")

    @staticmethod
    def _json_string(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'


def test_hindsight_manager_add_get_search_update_delete() -> None:
    session = FakeSession()
    manager = HindsightMemoryManager(api_key="test-key", session=session)

    memory_id = manager.add_memory(
        "Alice prefers terse commit messages",
        "vivi",
        metadata={"visibility": "private", "project_id": "pixelated"},
        category="preference",
    )

    stored = manager.get_memory(memory_id)
    assert stored is not None
    assert stored["content"] == "Alice prefers terse commit messages"
    assert stored["metadata"]["project_id"] == "pixelated"
    assert stored["metadata"]["visibility"] == "private"

    results = manager.search_memories("commit messages", "vivi", limit=5)
    assert results[0]["id"] == "doc-1"
    assert results[0]["metadata"]["project_id"] == "pixelated"

    assert manager.update_memory(
        memory_id,
        "Alice prefers concise commit messages",
        metadata={"visibility": "private", "project_id": "pixelated"},
    )
    updated = manager.get_memory(memory_id)
    assert updated is not None
    assert updated["content"] == "Alice prefers concise commit messages"

    assert manager.delete_memory(memory_id)
    assert manager.get_memory(memory_id) is None


def test_hindsight_manager_get_all_filters_by_user_tag() -> None:
    session = FakeSession()
    manager = HindsightMemoryManager(api_key="test-key", session=session)

    manager.add_memory("memory one", "vivi", metadata={"visibility": "private"})
    manager.add_memory("memory two", "other", metadata={"visibility": "private"})

    results = manager.get_all_memories("vivi")

    assert len(results) == 1
    assert results[0]["content"] == "memory one"
