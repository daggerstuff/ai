"""Minimal production-minded Youtube processor wrapper."""

from __future__ import annotations

from typing import Any

from .youtube_rag_system import YouTubeRAGSystem


class YoutubeProcessor:
    """Process YouTube-derived payloads through a small RAG + formatting flow."""

    def __init__(self, rag_system: YouTubeRAGSystem | None = None) -> None:
        self.rag_system = rag_system or YouTubeRAGSystem()

    def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")

        query = payload.get("query") or payload.get("text") or ""
        response = self.rag_system.answer(str(query))
        return {
            "status": "ok",
            "query": str(query),
            "response": response,
        }


__all__ = ["YoutubeProcessor"]
