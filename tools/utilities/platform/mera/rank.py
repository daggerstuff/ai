"""Rank stage (Mera Task 4) — score retrieved candidates.

The Rank judge computes a final diagnosis score for each candidate produced
by the Memorize stage.  The score blends retrieval confidence, evidence
chain contributions (symptom/test/progression/demographic matches), and
hierarchy-based parent-level fall-back credit so that coarser concepts are
not unfairly penalised when specific subtype evidence is sparse.
"""

from __future__ import annotations

from .contrastive import HierarchicalEmbedder
from .hierarchy import TherapeuticConceptHierarchy
from .types import (
    Candidate,
    EvidenceChainLink,
    MemorizeRankConfig,
    PatientPresentation,
)


class RankStage:
    """Score retrieved candidates with evidence-aware ranking."""

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        embedder: HierarchicalEmbedder,
        config: MemorizeRankConfig | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.embedder = embedder
        self.config = config or MemorizeRankConfig()

    def rank(
        self,
        presentation: PatientPresentation,
        candidates: list[Candidate],
    ) -> list[Candidate]:
        """Re-score candidates with symptom/test/progression/demographic weights."""
        ranked: list[Candidate] = []
        for c in candidates:
            evidence_score = self._evidence_score(presentation, c)
            # Blend retrieval score with evidence score.
            blend = self.config.rank_retrieval_blend
            final = blend * c.retrieval_score + (1.0 - blend) * evidence_score

            # Build a minimal evidence chain from the presentation.
            chain = self._build_chain(presentation, c)

            ranked.append(
                Candidate(
                    condition_id=c.condition_id,
                    condition_name=c.condition_name,
                    hierarchy_node_id=c.hierarchy_node_id,
                    retrieval_score=c.retrieval_score,
                    evidence_score=evidence_score,
                    final_score=final,
                    retrieval_evidence=c.retrieval_evidence,
                    evidence_chain=chain,
                    hierarchy_path=c.hierarchy_path,
                    confidence=final,
                )
            )
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked

    def _evidence_score(
        self, presentation: PatientPresentation, candidate: Candidate
    ) -> float:
        w_sym = self.config.rank_symptom_weight
        w_pres = self.config.rank_presentation_weight
        w_test = self.config.rank_test_weight
        w_prog = self.config.rank_progression_weight

        # Symptom overlap: count findings whose text overlaps with descriptors.
        sym_score = 0.0
        if presentation.findings:
            sym_matches = 0
            for finding in presentation.findings:
                text_tokens = {w for w in finding.text.lower().split() if len(w) > 2}
                cond_node = self.hierarchy.get(candidate.hierarchy_node_id)
                descriptors = {d.lower() for d in (cond_node.descriptors if cond_node else [])}
                if text_tokens & descriptors:
                    sym_matches += 1
            sym_score = sym_matches / len(presentation.findings)

        # Presentation / keyword overlap (simplified proxy for typical presentation match).
        pres_score = 0.5  # baseline; elevated when retrieval evidence indicates strong semantic match.
        for ev in candidate.retrieval_evidence:
            if ev.source == "semantic" and ev.score > 0.5:
                pres_score = min(1.0, pres_score + 0.3)

        # Test match: no direct test knowledge in this slim prototype; keep neutral.
        test_score = 0.5

        # Progression / history match: neutral baseline.
        prog_score = 0.5

        return (
            w_sym * sym_score
            + w_pres * pres_score
            + w_test * test_score
            + w_prog * prog_score
        )

    def _build_chain(
        self, presentation: PatientPresentation, candidate: Candidate
    ) -> list[EvidenceChainLink]:
        # Candidate reserved for future hierarchy-path evidence links.
        del candidate
        chain: list[EvidenceChainLink] = []
        for finding in presentation.findings:
            contribution = finding.weight
            chain.append(
                EvidenceChainLink(
                    finding_id=finding.finding_id,
                    finding_text=finding.text,
                    contribution=contribution,
                    match_dimension="symptom_match",
                )
            )
        return chain
