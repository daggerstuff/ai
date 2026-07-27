"""Literature Matcher Sub-Agent — retrieves relevant case reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..types import (
    KnowledgeMatch,
    PatientCase,
)

logger = logging.getLogger(__name__)


class LiteratureMatcherAgent:
    """Hybrid search (semantic + keyword) against a rare-disease
    knowledge base to retrieve matching case reports."""

    def __init__(self, knowledge_base: Any) -> None:
        self._kb = knowledge_base

    def search(self, case: PatientCase, top_k: int = 5) -> dict[str, Any]:
        """Run hybrid retrieval and return ranked matches."""
        query = self._build_query(case)
        results = self._kb.hybrid_search(query, top_k=top_k)

        matches = [
            {
                "disease_id": disease.disease_id,
                "disease_name": disease.name,
                "score": score,
                "match_type": match_type,
            }
            for disease, score, match_type in results
        ]

        transparency = self._compute_transparency(matches, top_k)

        return {
            "matches": matches[:top_k],
            "semantic_hits": sum(1 for m in matches if m["match_type"] == "semantic"),
            "keyword_hits": sum(1 for m in matches if m["match_type"] == "keyword"),
            "transparency_score": transparency,
        }

    # ------------------------------------------------------------------ #

    def _build_query(self, case: PatientCase) -> str:
        parts = [s.name for s in case.symptoms]
        parts.extend(case.family_history)
        parts.append(case.demographics.get("age_group", "adult"))
        return " ".join(parts)

    @staticmethod
    def _compute_transparency(
        matches: list[dict[str, Any]], top_k: int
    ) -> float:
        if not matches:
            return 0.0
        seen = matches[:top_k]
        scores = [m["score"] for m in seen]
        top_score = max(scores) if scores else 0.0
        return top_score
