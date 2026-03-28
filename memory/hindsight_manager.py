import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import requests

from ai.api.memory.base import BaseMemoryManager

logger = logging.getLogger(__name__)


class HindsightMemoryManager(BaseMemoryManager):
    """Hindsight-backed memory manager for shared operational memory."""

    DEFAULT_TYPES = ["world", "experience", "observation"]
    METADATA_TAG_KEYS = (
        "visibility",
        "org_id",
        "project_id",
        "agent_id",
        "run_id",
        "session_id",
        "category",
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        bank_id: Optional[str] = None,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("HINDSIGHT_API_KEY")
        self.api_url = (
            api_url
            or os.environ.get("HINDSIGHT_API_URL")
            or "https://api.hindsight.vectorize.io"
        ).rstrip("/")
        self.bank_id = bank_id or os.environ.get("HINDSIGHT_BANK_ID") or "pixeldated"
        self.timeout = timeout
        self.session = session or requests.Session()

        if not self.api_key:
            raise ValueError("HINDSIGHT_API_KEY is required for HindsightMemoryManager")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _url(self, suffix: str) -> str:
        return f"{self.api_url}/v1/default/banks/{self.bank_id}{suffix}"

    def _request(self, method: str, suffix: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method=method,
            url=self._url(suffix),
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()

    def _make_document_id(self, user_id: str) -> str:
        return f"hindsight-{user_id}-{uuid.uuid4().hex[:12]}"

    def _serialize_context(
        self,
        *,
        user_id: str,
        metadata: Optional[Dict[str, Any]],
        category: Optional[str],
    ) -> str:
        payload = {
            "user_id": user_id,
            "metadata": metadata or {},
            "category": category or (metadata or {}).get("category"),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _parse_context(self, context: Optional[str]) -> Dict[str, Any]:
        if not context:
            return {}
        try:
            parsed = json.loads(context)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _metadata_to_tags(
        self,
        *,
        user_id: str,
        metadata: Optional[Dict[str, Any]],
        category: Optional[str],
    ) -> List[str]:
        merged = dict(metadata or {})
        if category:
            merged["category"] = category
        tags = [f"user:{user_id}"]
        for key in self.METADATA_TAG_KEYS:
            value = merged.get(key)
            if value is None or value == "":
                continue
            tags.append(f"{key}:{value}")
        return tags

    def _metadata_from_tags(self, tags: Optional[List[str]]) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        for tag in tags or []:
            if ":" not in tag:
                continue
            key, value = tag.split(":", 1)
            if key == "user":
                continue
            metadata[key] = value
        return metadata

    def _record_from_document_summary(
        self,
        item: Dict[str, Any],
        *,
        original_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        tags = item.get("tags") or []
        metadata = self._metadata_from_tags(tags)
        context_payload = {}
        retain_params = item.get("retain_params")
        if isinstance(retain_params, str):
            try:
                parsed = json.loads(retain_params)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                context_payload = self._parse_context(parsed.get("context"))
        user_id = context_payload.get("user_id")
        if not user_id:
            for tag in tags:
                if tag.startswith("user:"):
                    user_id = tag.split(":", 1)[1]
                    break
        content = original_text if original_text is not None else ""
        return {
            "id": item["id"],
            "memory": content,
            "content": content,
            "user_id": user_id,
            "metadata": context_payload.get("metadata") or metadata,
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
        }

    def _record_from_recall(self, result: Dict[str, Any]) -> Dict[str, Any]:
        context_payload = self._parse_context(result.get("context"))
        metadata = context_payload.get("metadata") or self._metadata_from_tags(
            result.get("tags")
        )
        return {
            "id": result.get("document_id") or result.get("id"),
            "memory": result.get("text", ""),
            "content": result.get("text", ""),
            "user_id": context_payload.get("user_id"),
            "metadata": metadata,
            "score": result.get("score"),
            "source_id": result.get("id"),
            "document_id": result.get("document_id"),
            "chunk_id": result.get("chunk_id"),
            "created_at": result.get("mentioned_at"),
        }

    def add(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        category = kwargs.get("category")
        document_id = kwargs.get("document_id") or self._make_document_id(user_id)
        payload = {
            "items": [
                {
                    "content": content,
                    "document_id": document_id,
                    "context": self._serialize_context(
                        user_id=user_id,
                        metadata=metadata,
                        category=category,
                    ),
                    "tags": self._metadata_to_tags(
                        user_id=user_id,
                        metadata=metadata,
                        category=category,
                    ),
                }
            ]
        }
        self._request("POST", "/memories", json=payload)
        return {"results": [{"id": document_id}]}

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "budget": kwargs.get("budget", "mid"),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "types": kwargs.get("types", self.DEFAULT_TYPES),
            "tags": [f"user:{user_id}"],
            "tags_match": kwargs.get("tags_match", "any"),
        }
        response = self._request("POST", "/memories/recall", json=payload) or {}
        results = response.get("results", [])
        return {
            "results": [self._record_from_recall(item) for item in results[:limit]]
        }

    def get_all(self, user_id: str, **kwargs: Any) -> Dict[str, Any]:
        limit = kwargs.get("limit", 100)
        offset = kwargs.get("offset", 0)
        response = self._request(
            "GET",
            "/documents",
            params={"limit": limit, "offset": offset},
        ) or {}
        items = response.get("items", [])
        records = [
            self.get(item["id"])
            for item in items
            if f"user:{user_id}" in (item.get("tags") or [])
        ]
        return {"results": [record for record in records if record]}

    def get(self, memory_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        try:
            document = self._request("GET", f"/documents/{memory_id}")
        except Exception:
            return None
        if not document:
            return None
        return self._record_from_document_summary(
            document,
            original_text=document.get("original_text", ""),
        )

    def update(self, memory_id: str, new_content: str, **kwargs: Any) -> bool:
        existing = self.get(memory_id)
        if not existing:
            return False
        metadata = kwargs.get("metadata")
        if metadata is None:
            metadata = existing.get("metadata")
        user_id = existing.get("user_id")
        if not user_id:
            return False
        self._request("DELETE", f"/documents/{memory_id}")
        self.add(
            new_content,
            user_id=user_id,
            metadata=metadata,
            document_id=memory_id,
            category=(metadata or {}).get("category"),
        )
        return True

    def delete(self, memory_id: str, **kwargs: Any) -> bool:
        self._request("DELETE", f"/documents/{memory_id}")
        return True

    def delete_all(self, user_id: str, **kwargs: Any) -> bool:
        memories = self.get_all(user_id).get("results", [])
        for memory in memories:
            self.delete(memory["id"])
        return True

    def add_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
    ) -> str:
        result = self.add(content, user_id, metadata=metadata, category=category)
        return result["results"][0]["id"]

    def search_memories(
        self, query: str, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self.search(query, user_id, limit=limit)["results"]

    def get_all_memories(
        self, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.get_all(user_id, limit=limit)["results"]

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        return self.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.update(memory_id, new_content, metadata=metadata)

    def delete_memory(self, memory_id: str) -> bool:
        return self.delete(memory_id)

    def clear_memory(self, user_id: str) -> bool:
        return self.delete_all(user_id)
