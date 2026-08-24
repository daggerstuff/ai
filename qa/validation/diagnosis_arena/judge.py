"""
DiagnosisArena judges.

A ``Judge`` scores a ``(case, response)`` pair and returns a ``Judgment``.
The interface is deliberately minimal so production deployments can plug in
GPT-4o-as-judge, another LLM, or a domain-tuned classifier.

Two reference implementations are provided:

- ``HeuristicJudge``: deterministic, rule-based, no network — useful for
  unit tests and offline regression checks.
- ``LLMJudge``: protocol stub for systems that wrap an LLM. Concrete
  implementations belong to the integration layer, not this package.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from .types import (
    DIAGNOSTIC_DIMENSIONS,
    ClinicalCase,
    DimensionScore,
    GeneratedDiagnosis,
    Judgment,
    ModelResponse,
    ResponseFormat,
    TierScore,
)


class Judge(ABC):
    """Abstract judge."""

    @abstractmethod
    def judge(self, case: ClinicalCase, response: ModelResponse) -> Judgment:
        """Score one (case, response) pair."""


class HeuristicJudge(Judge):
    """
    Deterministic, rule-based judge.

    Scoring rules (per the paper):

    - Tier::
        Identical: response.final_diagnosis normalized-equals case.final_diagnosis
        Relevant: response.final_diagnosis token-overlaps case.final_diagnosis
        Irrelevant: otherwise

    - Hypothesis generation::
        |response.hypothesis_list ∩ case.differential_diagnoses| / |case.differential_diagnoses|
        (clipped to [0, 1])

    - Evidence interpretation::
        |response.evidence_cited ∩ case.supporting_evidence| / |case.supporting_evidence|

    - Differential diagnosis::
        |response.differential_list ∩ case.differential_diagnoses| / |case.differential_diagnoses|

    - Final diagnosis::
        identical  -> 1.0
        relevant   -> 0.5
        irrelevant -> 0.0

    All scoring is case- and whitespace-insensitive on the final-diagnosis
    text. Hypotheses/evidence/differentials are compared as sets of
    lowercased tokens.
    """

    def __init__(self, judge_model: str = "heuristic-v1"):
        self.judge_model = judge_model

    def judge(self, case: ClinicalCase, response: ModelResponse) -> Judgment:
        tier = self._tier(case, response)
        dims = (
            DimensionScore(
                name="hypothesis_generation",
                score=self._set_overlap(response.hypothesis_list, case.differential_diagnoses),
                rationale=f"{len(set(response.hypothesis_list) & set(case.differential_diagnoses))}/{len(case.differential_diagnoses)} overlap",
            ),
            DimensionScore(
                name="evidence_interpretation",
                score=self._set_overlap(response.evidence_cited, case.supporting_evidence),
                rationale="evidence overlap with ground-truth",
            ),
            DimensionScore(
                name="differential_diagnosis",
                score=self._set_overlap(response.differential_list, case.differential_diagnoses),
                rationale="differential overlap with ground-truth",
            ),
            DimensionScore(
                name="final_diagnosis",
                score=tier.numeric,
                rationale=f"tier={tier.value}",
            ),
        )
        return Judgment(
            response_id=response.response_id,
            case_id=case.case_id,
            tier=tier,
            dimensions=dims,
            judge_model=self.judge_model,
            notes="",
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _tokens(values) -> set[str]:
        return {v.strip().lower() for v in values if v and v.strip()}

    def _tier(self, case: ClinicalCase, response: ModelResponse) -> TierScore:
        if response.format is ResponseFormat.MCQ:
            if response.mcq_selected and case.mcq_options:
                try:
                    if case.mcq_options.index(response.mcq_selected) == 0:
                        return TierScore.IDENTICAL
                except ValueError:
                    pass
            return TierScore.IRRELEVANT

        gt = self._normalize(case.final_diagnosis)
        pred = self._normalize(response.final_diagnosis)
        if not gt and not pred:
            return TierScore.RELEVANT
        if not pred:
            return TierScore.IRRELEVANT
        if gt == pred:
            return TierScore.IDENTICAL
        if self._token_overlap(gt, pred):
            return TierScore.RELEVANT
        return TierScore.IRRELEVANT

    @staticmethod
    def _token_overlap(a: str, b: str) -> bool:
        if not a or not b:
            return False
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        return bool(tokens_a & tokens_b)

    def _set_overlap(self, response_items, ground_truth) -> float:
        gt = self._tokens(ground_truth)
        pred = self._tokens(response_items)
        if not gt:
            return 0.0
        return len(gt & pred) / len(gt)


class LLMJudge(Judge):
    """
    LLM-backed judge stub.

    Accepts an arbitrary scoring function with the signature
    ``(case, response) -> Judgment``. Use this with a wrapper that calls
    GPT-4o, Claude, or a domain-tuned model. The wrapper lives in the
    integration layer (e.g. ``ai.qa.validation.llm_judge_openai``).
    """

    def __init__(
        self,
        scorer: Callable[[ClinicalCase, ModelResponse], Judgment],
        judge_model: str = "llm",
    ):
        if scorer is None:
            msg = "scorer callable required for LLMJudge"
            raise ValueError(msg)
        self._scorer = scorer
        self.judge_model = judge_model

    def judge(self, case: ClinicalCase, response: ModelResponse) -> Judgment:
        judgment = self._scorer(case, response)
        for dim in judgment.dimensions:
            if dim.name not in DIAGNOSTIC_DIMENSIONS:
                msg = f"unknown dimension: {dim.name}"
                raise ValueError(msg)
        return judgment


class ClinicalDiagnosisJudge(Judge):
    """Backward-compatible judge used by historical callers in `conversation_eval_metrics`."""

    def __init__(self, *, judge_model: str = "clinical-judge-v1", llm_judge: Judge | None = None) -> None:
        self.judge_model = judge_model
        self.llm_judge = llm_judge
        self.rng = __import__("random").Random(0)

    def _llm_judge_call(self, case: ClinicalCase, response: ModelResponse) -> dict:
        if self.llm_judge is None:
            return {"tier": TierScore.IRRELEVANT, "scores": {}}
        judgment = self.llm_judge.judge(case, response)
        return {
            "tier": judgment.tier,
            "scores": {d.name: d.score for d in judgment.dimensions},
        }

    def _deterministic_judge(self, case: ClinicalCase, response: GeneratedDiagnosis) -> Judgment:
        candidate_overlap = self._token_overlap_ratio(case.key_differentiators, response.differential_list)
        evidence_overlap = self._token_overlap_ratio(case.ground_truth_evidence, response.evidence_cited)

        return Judgment(
            response_id=response.response_id,
            case_id=case.case_id,
            tier=TierScore.RELEVANT,
            dimensions=(
                DimensionScore(name="hypothesis_generation", score=candidate_overlap, rationale=""),
                DimensionScore(name="evidence_interpretation", score=evidence_overlap, rationale=""),
                DimensionScore(name="differential_diagnosis", score=candidate_overlap, rationale=""),
                DimensionScore(name="final_diagnosis", score=0.5, rationale="deterministic fallback"),
            ),
            judge_model=self.judge_model,
            notes="deterministic fallback",
        )

    def judge(self, case: ClinicalCase, response: ModelResponse) -> Judgment:
        if isinstance(response, GeneratedDiagnosis):
            return self._deterministic_judge(case, response)
        return self._deterministic_judge(case, GeneratedDiagnosis(
            response_id=response.response_id,
            case_id=response.case_id,
            format=response.format,
            final_diagnosis=response.final_diagnosis,
            differential_list=response.differential_list,
            evidence_cited=response.evidence_cited,
            reasoning=getattr(response, "reasoning", ""),
        ))

    @staticmethod
    def _token_overlap_ratio(a: tuple[str, ...], b: tuple[str, ...]) -> float:
        gt = {x.strip().lower() for x in a if x and x.strip()}
        pred = {x.strip().lower() for x in b if x and x.strip()}
        if not gt:
            return 0.0
        return len(gt & pred) / len(gt)


__all__ = ["Judge", "HeuristicJudge", "LLMJudge", "ClinicalDiagnosisJudge"]
