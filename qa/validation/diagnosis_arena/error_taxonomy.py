"""
Diagnostic error taxonomy.

Classifies diagnostic mistakes per the bias categories flagged by the
DiagnosisArena paper and clinical-reasoning literature:

- ``premature_closure``: ends differential early; final_diagnosis selected
  without exhausting the differential list.
- ``anchoring_bias``: latches onto a single hypothesis; differential is a
  near-singleton and does not include the ground truth.
- ``availability_bias``: proposes only common, prevalent diagnoses; omits
  rare/atypical ones appropriate to the case difficulty.
- ``confirmation_bias``: cites only evidence that supports the chosen
  diagnosis; ground-truth supporting evidence absent from response.
- ``overconfidence``: asserts a single diagnosis with high confidence
  (final_diagnosis set, reasoning short, no differentials).
- ``irrelevant``: response entirely misses the clinical context.
"""

from __future__ import annotations

import re

from .types import ClinicalCase, Difficulty, ModelResponse, TierScore

ERROR_TAXONOMY: tuple[str, ...] = (
    "premature_closure",
    "anchoring_bias",
    "availability_bias",
    "confirmation_bias",
    "overconfidence",
    "irrelevant",
)


def classify_errors(
    case: ClinicalCase,
    response: ModelResponse,
    tier: TierScore,
) -> tuple[str, ...]:
    """
    Return the set of error categories that apply to a (case, response) pair.

    Errors are independent flags; multiple can co-occur on the same case.
    """
    errors: list[str] = []

    final = (response.final_diagnosis or "").strip()
    diffs = [d for d in response.differential_list if d]
    hypo = [h for h in response.hypothesis_list if h]
    cited = [e for e in response.evidence_cited if e]

    if tier is TierScore.IRRELEVANT and not final and not diffs and not hypo:
        errors.append("irrelevant")
        return tuple(errors)

    gt_set = {d.strip().lower() for d in case.differential_diagnoses if d}
    final_in_gt = final.strip().lower() in gt_set if final else False

    if final and len(diffs) <= 1 and final_in_gt is False:
        errors.append("premature_closure")

    if len(diffs) <= 1 and not final_in_gt:
        errors.append("anchoring_bias")

    if case.difficulty is Difficulty.COMPLEX and all(_is_common_diagnosis(d) for d in diffs + hypo):
        errors.append("availability_bias")

    gt_evidence = {e.strip().lower() for e in case.supporting_evidence if e}
    cited_lower = {e.strip().lower() for e in cited}
    if final and gt_evidence and not (gt_evidence & cited_lower):
        errors.append("confirmation_bias")

    if final and len(diffs) <= 1 and _is_short_reasoning(response.reasoning):
        errors.append("overconfidence")

    return tuple(errors)


def _is_common_diagnosis(text: str) -> bool:
    """A trivially-likely-common diagnosis heuristic (placeholder)."""
    text = (text or "").strip().lower()
    if not text:
        return True
    return text in _COMMON_DIAGNOSES


_COMMON_DIAGNOSES: frozenset[str] = frozenset(
    {
        "common cold",
        "viral infection",
        "stress",
        "anxiety",
        "depression",
        "hypertension",
    },
)


def _is_short_reasoning(reasoning: str) -> bool:
    reasoning = (reasoning or "").strip()
    if not reasoning:
        return True
    return len(reasoning.split()) < 30 or len(re.findall(r"\w+", reasoning)) < 30


class ErrorTaxonomy(tuple[str, ...]):
    """Backward-compatible alias for the error taxonomy tuple."""
