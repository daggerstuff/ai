"""Memorize stage (Mera Task 3) — hybrid retrieval against the concept hierarchy.

Given a :class:`PatientPresentation`, the memorize stage produces a set of
retrieval candidates from the knowledge base using the fitted
:class:`HierarchicalEmbedder`.  It blends semantic similarity (embedding
cosine), hierarchical closeness (Wu-Palmer), and keyword overlap so that
candidates matching parent-level concepts are still surfaced when a
specific subtype has few examples.
"""

from __future__ import annotations

from typing import Any

from .contrastive import HierarchicalEmbedder
from .hierarchy import TherapeuticConceptHierarchy
from .types import (
    Candidate,
    MemorizeRankConfig,
    MeraResult,
    PatientPresentation,
    RetrievalEvidence,
)


class MemorizeStage:
    """Hybrid retrieval stage using the fitted hierarchical embedder."""

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        embedder: HierarchicalEmbedder,
        config: MemorizeRankConfig | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.embedder = embedder
        self.config = config or MemorizeRankConfig()

    def retrieve(
        self, presentation: PatientPresentation, top_k: int = 20
    ) -> list[Candidate]:
        """Retrieve top candidates for a presentation."""
        top_k = max(1, top_k)
        # Encode presentation findings to a single vector.
        query_vec = self.embedder.encode_presentation(
            presentation, self.hierarchy
        )

        candidates: list[Candidate] = []
        all_cond_ids = [
            n.node_id
            for n in self.hierarchy.nodes.values()
            if n.level.value == 1  # CONDITION level
        ]

        for cid in all_cond_ids:
            cond_node = self.hierarchy.get(cid)
            if cond_node is None:
                continue
            # Semantic similarity (cosine in embedding space).
            cond_vec = self.embedder.encode_condition(
                self.hierarchy, cid
            )
            sim_sem = self.embedder.cosine(query_vec, cond_vec)

            # Hierarchical closeness via Wu-Palmer similarity to presentation.
            # We approximate by mapping presentation descriptors to closest node.
            closest_node = self._closest_node_for_findings(
                presentation
            )
            sim_hier = (
                self.hierarchy.similarity(closest_node, cid)
                if closest_node else 0.0
            )

            # Keyword overlap: descriptor/token overlap.
            sim_kw = self._keyword_overlap(presentation, cond_node)

            # Blended retrieval score per MemorizeRankConfig weights.
            alpha = self.config.retrieval_alpha
            beta = self.config.retrieval_beta
            gamma = self.config.retrieval_gamma
            score = (
                alpha * sim_sem
                + beta * sim_hier
                + gamma * sim_kw
            )

            evidence = RetrievalEvidence(
                source="semantic" if sim_sem > sim_hier else "hierarchical",
                score=score,
                detail=f"sim_sem={sim_sem:.3f} sim_hier={sim_hier:.3f} kw={sim_kw:.3f}",
            )

            # Build hierarchy path from root to condition.
            path = self.hierarchy.path_to_root(cid)
            path = [p for p in path if p != self.hierarchy.root_id]

            candidates.append(
                Candidate(
                    condition_id=cond_node.condition_id or cid,
                    condition_name=cond_node.name,
                    hierarchy_node_id=cid,
                    retrieval_score=score,
                    retrieval_evidence=[evidence],
                    hierarchy_path=path,
                    confidence=sim_sem,
                )
            )

        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        return candidates[:top_k]

    def _closest_node_for_findings(
        self, presentation: PatientPresentation
    ) -> str | None:
        """Find the closest hierarchy node to any finding's descriptors."""
        best_node: str | None = None
        best_score = -1.0
        for finding in presentation.findings:
            text_lower = finding.text.lower()
            for token in text_lower.split():
                if len(token) <= 2:
                    continue
                for nid, node in self.hierarchy.nodes.items():
                    if nid == self.hierarchy.root_id:
                        continue
                    descriptors = [d.lower() for d in node.descriptors]
                    score = 1.0 if token in descriptors else 0.0
                    if score > best_score:
                        best_score = score
                        best_node = nid
        return best_node

    def _keyword_overlap(
        self, presentation: PatientPresentation, node: Any
    ) -> float:
        """Simple Jaccard overlap between finding tokens and node descriptors."""
        tokens: set[str] = set()
        for finding in presentation.findings:
            for w in finding.text.lower().split():
                if len(w) > 2:
                    tokens.add(w)
        descriptors = {d.lower() for d in node.descriptors}
        if not descriptors:
            return 0.0
        overlap = tokens & descriptors
        return len(overlap) / len(tokens | descriptors)

    def memorize(self, presentation: PatientPresentation) -> MeraResult:
        """Run full Memorize stage: retrieve and assemble MeraResult."""
        candidates = self.retrieve(
            presentation, top_k=self.config.top_k_retrieval
        )
        # Prune below floor.
        floor = self.config.prune_floor
        candidates = [
            c for c in candidates if c.retrieval_score >= floor
        ]
        return MeraResult(
            case_id=presentation.case_id,
            ranked_candidates=candidates,
            total_latency_ms=0.0,
            hierarchy_used=True,
            stages={"memorize": "completed", "retrieval_count": len(candidates)},
        )
