"""Simple retrieval-augmented generation scaffold for YouTube corpus lookups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RAGMatch:
    video_id: str
    title: str
    chunk: str
    score: float


class YouTubeRAGSystem:
    """Very small deterministic RAG system over in-memory transcripts."""

    def __init__(self, corpus: list[dict[str, Any]] | None = None) -> None:
        self.corpus = corpus or []

    def add_document(self, video_id: str, title: str, transcript: str) -> None:
        self.corpus.append({"video_id": video_id, "title": title, "transcript": transcript})

    def search(self, query: str, *, top_k: int = 3) -> list[RAGMatch]:
        q = query.lower()
        scored = []
        for doc in self.corpus:
            text = str(doc.get("transcript", "")).lower()
            score = float(text.count(q))
            if score > 0:
                scored.append(
                    RAGMatch(
                        video_id=str(doc.get("video_id", "")),
                        title=str(doc.get("title", "")),
                        chunk=doc.get("transcript", "")[:160],
                        score=score,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def generate_context(self, query: str, *, top_k: int = 3) -> str:
        matches = self.search(query, top_k=top_k)
        return "\n---\n".join(match.chunk for match in matches)

    def answer(self, query: str, *, top_k: int = 3) -> dict[str, Any]:
        matches = self.search(query, top_k=top_k)
        return {
            "query": query,
            "answers": [match.chunk for match in matches],
            "scores": [match.score for match in matches],
            "count": len(matches),
        }


__all__ = ["RAGMatch", "YouTubeRAGSystem"]
