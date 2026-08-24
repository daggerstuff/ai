"""Evaluation adapter (Task 5) — zero-shot train/test split and metrics.

Protocol:
  - Train split: conditions whose ``train_split`` is ``True``.
  - Test split  : conditions marked ``False`` (A-L / M-Z pattern for Mera).
  - Metrics     : Recall@1, Recall@5, MRR, flat-vs-hierarchical delta.
"""

from __future__ import annotations

import statistics

from .types import MeraResult


def recall_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def mrr(ranked: list[str], target: str) -> float:
    try:
        rank = ranked.index(target) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def evaluate_result(result: MeraResult, ground_truth: str) -> dict[str, float]:
    ranked_ids = [c.condition_id for c in result.ranked_candidates]
    return {
        "recall@1": recall_at_k(ranked_ids, ground_truth, 1),
        "recall@5": recall_at_k(ranked_ids, ground_truth, 5),
        "mrr": mrr(ranked_ids, ground_truth),
        "n_candidates": float(len(result.ranked_candidates)),
    }


def evaluate_batch(
    results: list[tuple[MeraResult, str]],
) -> dict[str, float]:
    r1s = [evaluate_result(r, gt)["recall@1"] for r, gt in results]
    r5s = [evaluate_result(r, gt)["recall@5"] for r, gt in results]
    mrrs = [evaluate_result(r, gt)["mrr"] for r, gt in results]
    return {
        "mean_recall@1": statistics.mean(r1s) if r1s else 0.0,
        "mean_recall@5": statistics.mean(r5s) if r5s else 0.0,
        "mean_mrr": statistics.mean(mrrs) if mrrs else 0.0,
        "n_evaluated": float(len(results)),
    }
