"""Hybrid knowledge-base retrieval for rare-disease evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .state import RareDiseaseState  # noqa: F401 — re-exported by __init__
from .types import (
    DiseaseRarity,
    KnowledgeMatch,
    OrganSystem,
    RareDisease,
    Symptom,
)


class RareDiseaseKnowledgeBase:
    """Abstract interface for rare-disease knowledge retrieval."""

    def search_by_symptom(self, symptom: Symptom, top_k: int = 10) -> list[KnowledgeMatch]:
        raise NotImplementedError

    def search_by_gene(self, gene: str, top_k: int = 10) -> list[KnowledgeMatch]:
        raise NotImplementedError

    def search_by_organ_system(
        self, system: OrganSystem, top_k: int = 10
    ) -> list[KnowledgeMatch]:
        raise NotImplementedError

    def get_disease(self, disease_id: str) -> RareDisease | None:
        raise NotImplementedError

    def hybrid_search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[RareDisease, float, str]]:
        """Combine semantic + keyword + structured retrieval."""
        keyword_hits = self._keyword_search(query, top_k)
        structured_hits = self._structured_search(query, top_k)
        return self._merge(keyword_hits, structured_hits, top_k)

    def _keyword_search(
        self, query: str, top_k: int
    ) -> list[tuple[RareDisease, float, str]]:
        raise NotImplementedError

    def _structured_search(
        self, query: str, top_k: int
    ) -> list[tuple[RareDisease, float, str]]:
        raise NotImplementedError

    def _merge(
        self,
        keyword: list[tuple[RareDisease, float, str]],
        structured: list[tuple[RareDisease, float, str]],
        top_k: int,
    ) -> list[tuple[RareDisease, float, str]]:
        """Combine keyword + structured results, deduplicate, re-rank."""
        scored: dict[str, tuple[RareDisease, float]] = {}
        for disease, score, _src in keyword + structured:
            existing = scored.get(disease.disease_id)
            if existing is None or score > existing[1]:
                scored[disease.disease_id] = (disease, score)
        ranked = sorted(scored.values(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class InMemoryRareDiseaseKnowledgeBase(RareDiseaseKnowledgeBase):
    """CPU-resolvable in-memory knowledge base for tests and fallback."""

    def __init__(self) -> None:
        self._diseases: dict[str, RareDisease] = {}
        self._index: dict[str, set[str]] = {}  # token → disease_id

    def add_disease(self, disease: RareDisease) -> None:
        self._diseases[disease.disease_id] = disease
        tokens = self._tokenize(
            " ".join(
                [
                    disease.name,
                    *[s.value for s in disease.organ_systems],
                    *disease.hpo_terms,
                    disease.rarity.value,
                ]
            )
        )
        for token in tokens:
            self._index.setdefault(token, set()).add(disease.disease_id)

    # ------------------------------------------------------------------ #
    #  RareDiseaseKnowledgeBase interface                                  #
    # ------------------------------------------------------------------ #

    def search_by_symptom(
        self, symptom: Symptom, top_k: int = 10
    ) -> list[KnowledgeMatch]:
        matches: list[KnowledgeMatch] = []
        for disease in self._diseases.values():
            score = self._symptom_overlap(symptom, disease)
            if score > 0:
                matches.append(
                    KnowledgeMatch(
                        disease_id=disease.disease_id,
                        score=score,
                        match_type="structured",
                        source="symptom_overlap",
                    )
                )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def search_by_gene(self, gene: str, top_k: int = 10) -> list[KnowledgeMatch]:
        token = gene.lower()
        ids = self._index.get(token, set())
        results = []
        for did in ids:
            d = self._diseases[did]
            results.append(
                KnowledgeMatch(
                    disease_id=did,
                    score=0.5,
                    match_type="keyword",
                    source="gene_index",
                )
            )
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    def search_by_organ_system(
        self, system: OrganSystem, top_k: int = 10
    ) -> list[KnowledgeMatch]:
        results = []
        for disease in self._diseases.values():
            if system in disease.organ_systems:
                results.append(
                    KnowledgeMatch(
                        disease_id=disease.disease_id,
                        score=1.0,
                        match_type="structured",
                        source="organ_system",
                    )
                )
        return results[:top_k]

    def get_disease(self, disease_id: str) -> RareDisease | None:
        return self._diseases.get(disease_id)

    def _keyword_search(
        self, query: str, top_k: int
    ) -> list[tuple[RareDisease, float, str]]:
        tokens = self._tokenize(query)
        scores: dict[str, float] = {}
        for token in tokens:
            for did in self._index.get(token, set()):
                scores[did] = scores.get(did, 0.0) + 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (self._diseases[did], score / len(tokens), "keyword")
            for did, score in ranked[:top_k]
            if did in self._diseases
        ]

    def _structured_search(
        self, query: str, top_k: int
    ) -> list[tuple[RareDisease, float, str]]:
        q = query.lower()
        results: list[tuple[RareDisease, float, str]] = []
        for disease in self._diseases.values():
            score = 0.0
            if q in disease.name.lower():
                score += 2.0
            for term in disease.hpo_terms:
                if q in term.lower():
                    score += 1.0
            if score > 0:
                results.append((disease, min(score, 3.0), "structured"))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------ #
    #  Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z]{3,}", text.lower())}

    @staticmethod
    def _symptom_overlap(symptom: Symptom, disease: RareDisease) -> float:
        score = 0.0
        if symptom.name.lower() in disease.name.lower():
            score += 0.3
        for hpo in disease.hpo_terms:
            if symptom.name.lower() in hpo.lower():
                score += 0.5
        if symptom.severity == SymptomSeverity.SEVERE:
            score *= 1.2
        return min(score, 1.0)
