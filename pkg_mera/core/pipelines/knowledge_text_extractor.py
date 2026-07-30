"""Knowledge extraction helpers for prompt and corpus text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractedKnowledge:
    source_id: str
    snippets: list[str]
    tags: list[str]


class KnowledgeTextExtractor:
    """Extract high-signal text spans for downstream indexing."""

    def extract(self, payload: dict[str, Any] | str, *, max_snippets: int = 5) -> list[str]:
        text = self._extract_text(payload)
        if not text:
            return []

        sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
        return sentences[:max_snippets]

    def extract_with_metadata(self, payload: dict[str, Any] | str, *, source_id: str = "default") -> ExtractedKnowledge:
        snippets = self.extract(payload)
        tags = self._infer_tags(payload)
        return ExtractedKnowledge(source_id=source_id, snippets=snippets, tags=tags)

    def summarize_snippets(self, snippets: list[str]) -> str:
        joined = " ".join(snippets).strip()
        return joined[:280] if len(joined) > 280 else joined

    def _extract_text(self, payload: dict[str, Any] | str) -> str:
        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            return ""
        for key in ("text", "content", "prompt", "document"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return " "

    def _infer_tags(self, payload: dict[str, Any] | str) -> list[str]:
        text = self._extract_text(payload).lower()
        tags: list[str] = []
        if "crisis" in text:
            tags.append("safety")
        if any(k in text for k in ("emotion", "feeling", "anxious", "sad")):
            tags.append("emotion")
        if "therapy" in text:
            tags.append("therapy")
        return tags


__all__ = ["ExtractedKnowledge", "KnowledgeTextExtractor"]
