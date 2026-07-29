"""Pipeline wiring (Mera) — connects Memorize + Rank stages."""

from __future__ import annotations

from .contrastive import HierarchicalEmbedder
from .hierarchy import TherapeuticConceptHierarchy, build_default_hierarchy
from .memorize import MemorizeStage
from .rank import RankStage
from .types import MemorizeRankConfig, MeraResult, PatientPresentation


class MeraPipeline:
    """Full Memorize & Rank pipeline."""

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy | None = None,
        embedder: HierarchicalEmbedder | None = None,
        config: MemorizeRankConfig | None = None,
    ) -> None:
        self.hierarchy = hierarchy or build_default_hierarchy()
        self.embedder = embedder or HierarchicalEmbedder()
        self.config = config or MemorizeRankConfig()
        self.memorize = MemorizeStage(self.hierarchy, self.embedder, self.config)
        self.rank = RankStage(self.hierarchy, self.embedder, self.config)

    def run(self, presentation: PatientPresentation) -> MeraResult:
        candidates = self.memorize.retrieve(
            presentation, top_k=self.config.top_k_retrieval
        )
        ranked = self.rank.rank(presentation, candidates)
        return MeraResult(
            case_id=presentation.case_id,
            ranked_candidates=ranked,
            total_latency_ms=0.0,
            hierarchy_used=True,
            stages={"memorize_candidates": len(candidates), "ranked_candidates": len(ranked)},
        )
