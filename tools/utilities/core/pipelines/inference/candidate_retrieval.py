"""
PIX-3912: Memorize Stage — Candidate Retrieval

Given a patient presentation, retrieve candidate diagnoses from:
- Knowledge base (rare disease profiles)
- Memory (past similar cases)
- Hierarchy (parent/child/sibling conditions)

Hybrid retrieval: semantic similarity + hierarchical closeness + keyword overlap.
Returns top-K candidates with similarity scores and retrieval evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ai.research.therapeutic_concept_hierarchy import TherapeuticConceptHierarchy


@dataclass
class RetrievalEvidence:
    """Evidence for why a candidate was retrieved."""

    source: str  # "knowledge_base", "memory", "hierarchy"
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateDiagnosis:
    """A retrieved candidate diagnosis with scores and evidence."""

    condition_id: str
    condition_name: str
    retrieval_score: float
    semantic_score: float
    hierarchy_score: float
    keyword_score: float
    evidence: list[RetrievalEvidence] = field(default_factory=list)
    # Hierarchical path for interpretability
    hierarchy_path: list[str] = field(default_factory=list)


class CandidateRetrievalEngine:
    """
    Hybrid retrieval engine for the Memorize stage.

    Combines three signals:
    1. Semantic similarity (sentence-transformer embeddings of patient text vs condition descriptions)
    2. Hierarchical closeness (how close in the therapeutic concept tree)
    3. Keyword overlap (explicit symptom mention matching)
    """

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        knowledge_base: dict[str, str] | None = None,
        memory_cases: list[dict[str, Any]] | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.5,
        hierarchy_weight: float = 0.3,
        keyword_weight: float = 0.2,
    ):
        self.hierarchy = hierarchy
        self.knowledge_base = knowledge_base or {}
        self.memory_cases = memory_cases or []
        self.semantic_weight = semantic_weight
        self.hierarchy_weight = hierarchy_weight
        self.keyword_weight = keyword_weight

        # Lazy-load embedding model
        self._embedding_model: SentenceTransformer | None = None
        self._embedding_model_name = embedding_model
        self._condition_embeddings: dict[str, np.ndarray] = {}

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _embed_text(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def _get_condition_embedding(self, condition_id: str) -> np.ndarray:
        """Get or compute embedding for a condition description."""
        if condition_id in self._condition_embeddings:
            return self._condition_embeddings[condition_id]

        # Build description from hierarchy
        node = self.hierarchy.get_node(condition_id)
        if node is None:
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())

        desc_parts = [node.name]
        # Add ancestor names for context
        for ancestor in self.hierarchy.get_ancestors(condition_id):
            desc_parts.append(ancestor.name)
        # Add knowledge base text if available
        if condition_id in self.knowledge_base:
            desc_parts.append(self.knowledge_base[condition_id])

        description = " | ".join(desc_parts)
        emb = self._embed_text(description)
        self._condition_embeddings[condition_id] = emb
        return emb

    def _keyword_overlap_score(self, patient_text: str, condition_id: str) -> float:
        """Compute keyword overlap between patient text and condition symptoms."""
        patient_tokens = set(re.findall(r"\b\w+\b", patient_text.lower()))

        # Gather symptom tokens from hierarchy leaves under this condition
        leaves = self.hierarchy.get_leaves(condition_id)
        if not leaves:
            # Use the condition name itself
            condition_tokens = set(re.findall(r"\b\w+\b", self.hierarchy.get_node(condition_id).name.lower()))
        else:
            condition_tokens = set()
            for leaf in leaves:
                condition_tokens.update(re.findall(r"\b\w+\b", leaf.name.lower()))

        if not condition_tokens:
            return 0.0

        overlap = len(patient_tokens & condition_tokens)
        return overlap / max(len(condition_tokens), 1)

    def _hierarchy_proximity_score(
        self, query_condition_id: str | None, candidate_id: str
    ) -> float:
        """
        If we have an initial query condition, score hierarchical closeness.
        Otherwise return a neutral score.
        """
        if query_condition_id is None or query_condition_id not in self.hierarchy:
            return 0.5
        sim = self.hierarchy.similarity(query_condition_id, candidate_id)
        return sim

    def retrieve(
        self,
        patient_presentation: str,
        top_k: int = 10,
        initial_guess: str | None = None,
    ) -> list[CandidateDiagnosis]:
        """
        Retrieve top-K candidate diagnoses for a patient presentation.

        Parameters
        ----------
        patient_presentation : free-text clinical description
        top_k : number of candidates to return
        initial_guess : optional condition_id from a prior screening step
        """
        patient_emb = self._embed_text(patient_presentation)
        patient_tokens = set(re.findall(r"\b\w+\b", patient_presentation.lower()))

        # Candidate pool: all level-1 conditions
        condition_ids = [n.id for n in self.hierarchy.get_nodes_at_level(1)]

        candidates: list[CandidateDiagnosis] = []

        for cid in condition_ids:
            cond_emb = self._get_condition_embedding(cid)
            node = self.hierarchy.get_node(cid)
            if node is None:
                continue

            # Semantic similarity (cosine)
            semantic_sim = float(
                np.dot(patient_emb, cond_emb)
                / (np.linalg.norm(patient_emb) * np.linalg.norm(cond_emb) + 1e-8)
            )

            # Hierarchy proximity
            hierarchy_sim = self._hierarchy_proximity_score(initial_guess, cid)

            # Keyword overlap
            keyword_sim = self._keyword_overlap_score(patient_presentation, cid)

            # Combined retrieval score
            retrieval_score = (
                self.semantic_weight * semantic_sim
                + self.hierarchy_weight * hierarchy_sim
                + self.keyword_weight * keyword_sim
            )

            # Build evidence
            evidence: list[RetrievalEvidence] = []
            evidence.append(
                RetrievalEvidence(
                    source="semantic",
                    score=semantic_sim,
                    details={"embedding_model": self._embedding_model_name},
                )
            )
            evidence.append(
                RetrievalEvidence(
                    source="hierarchy",
                    score=hierarchy_sim,
                    details={"lca_level": self.hierarchy.lowest_common_ancestor_level(initial_guess, cid) if initial_guess else None},
                )
            )
            evidence.append(
                RetrievalEvidence(
                    source="keyword",
                    score=keyword_sim,
                    details={"overlap_tokens": list(patient_tokens & self._get_condition_tokens(cid))},
                )
            )

            # Memory evidence
            memory_score = self._memory_similarity(patient_presentation, cid)
            if memory_score > 0.3:
                evidence.append(
                    RetrievalEvidence(
                        source="memory",
                        score=memory_score,
                        details={"num_similar_cases": len(self.memory_cases)},
                    )
                )
                retrieval_score += 0.1 * memory_score  # small boost from memory

            hierarchy_path = [n.name for n in self.hierarchy.get_ancestors(cid)]
            hierarchy_path.reverse()
            hierarchy_path.append(node.name)

            candidates.append(
                CandidateDiagnosis(
                    condition_id=cid,
                    condition_name=node.name,
                    retrieval_score=retrieval_score,
                    semantic_score=semantic_sim,
                    hierarchy_score=hierarchy_sim,
                    keyword_score=keyword_sim,
                    evidence=evidence,
                    hierarchy_path=hierarchy_path,
                )
            )

        # Sort by retrieval score descending
        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:top_k]

    def _get_condition_tokens(self, condition_id: str) -> set[str]:
        """Extract tokens from a condition's leaf symptoms."""
        leaves = self.hierarchy.get_leaves(condition_id)
        tokens: set[str] = set()
        for leaf in leaves:
            tokens.update(re.findall(r"\b\w+\b", leaf.name.lower()))
        return tokens

    def _memory_similarity(self, patient_presentation: str, condition_id: str) -> float:
        """Compute similarity to past cases in memory."""
        if not self.memory_cases:
            return 0.0

        patient_emb = self._embed_text(patient_presentation)
        scores: list[float] = []
        for case in self.memory_cases:
            case_text = case.get("presentation", "")
            case_condition = case.get("condition_id", "")
            if case_condition != condition_id:
                continue
            case_emb = self._embed_text(case_text)
            sim = float(
                np.dot(patient_emb, case_emb)
                / (np.linalg.norm(patient_emb) * np.linalg.norm(case_emb) + 1e-8)
            )
            scores.append(sim)

        return max(scores) if scores else 0.0

    def add_memory_case(self, presentation: str, condition_id: str, outcome: dict[str, Any] | None = None) -> None:
        """Add a past case to the memory store."""
        self.memory_cases.append({
            "presentation": presentation,
            "condition_id": condition_id,
            "outcome": outcome or {},
        })

    def add_knowledge_base_entry(self, condition_id: str, description: str) -> None:
        """Add or update a knowledge base entry."""
        self.knowledge_base[condition_id] = description
        # Invalidate cached embedding
        self._condition_embeddings.pop(condition_id, None)
