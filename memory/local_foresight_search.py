from __future__ import annotations

from typing import Any


def build_recall_results(documents: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    results = []
    for document in documents:
        rank = float(document.get("rank", 999.0))
        score = 1.0 / (1.0 + max(rank, 0.0))
        results.append(
            {
                "id": document["id"],
                "document_id": document["id"],
                "chunk_id": f"{document['id']}:0",
                "text": document["content"],
                "context": document.get("context") or "",
                "tags": list(document.get("tags") or []),
                "score": score,
                "mentioned_at": document["updated_at"],
            }
        )
        if len(results) >= limit:
            break
    return results
