"""DiagnosisArena evaluation adapter.

Enterprise-grade evaluation metrics for the DeepRare multi-agent system following
the DiagnosisArena framework (arXiv 2505.14107):

- Recall@1, Recall@5, Recall@10 (ground truth in top-K)
- Mean Reciprocal Rank (MRR)
- Wilson 95% confidence intervals on Recall@K and MRR
- Accuracy stratified by organ system / rarity tier / presentation complexity
- Safety violation tracking (red flags, blocked eliminations)
- Per-case error analysis for diagnostic quality improvement
- Single-agent baseline comparison with statistical significance
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .schema import DiagnosisResult, EvaluationMetrics, PatientCase

# Wilson score interval z-score for 95% CI
_Z_95 = 1.96


class DiagnosisArenaEvaluator:
    """Compute diagnostic evaluation metrics across a set of patient cases.

    Metrics:
        - Recall@1, Recall@5, Recall@10 (ground truth in top-K)
        - Mean Reciprocal Rank (MRR)
        - Wilson 95% confidence intervals on Recall@K and MRR
        - Accuracy stratified by organ system, rarity tier, complexity
        - Safety violation count and cases requiring referral
        - Average clinical confidence
        - Per-case error analysis
    """

    def evaluate(
        self,
        results: Sequence[DiagnosisResult],
        cases: Sequence[PatientCase],
    ) -> EvaluationMetrics:
        """Evaluate a batch of diagnosis results against ground truth.

        Args:
            results: Diagnosis results from the pipeline, one per case.
            cases: The original patient cases with ground_truth_diagnosis.

        Returns:
            Aggregated EvaluationMetrics with Wilson CIs and safety tracking.
        """
        if len(results) != len(cases):
            raise ValueError(f"Results ({len(results)}) and cases ({len(cases)}) must have the same length")

        total = len(results)
        if total == 0:
            return self._empty_metrics()

        recall_at_1 = 0
        recall_at_5 = 0
        recall_at_10 = 0
        reciprocal_ranks: list[float] = []
        correct = 0
        error_cases: list[dict[str, Any]] = []

        # Per-stratum tracking
        organ_correct: dict[str, int] = defaultdict(int)
        organ_total: dict[str, int] = defaultdict(int)
        rarity_correct: dict[str, int] = defaultdict(int)
        rarity_total: dict[str, int] = defaultdict(int)
        complexity_correct: dict[str, int] = defaultdict(int)
        complexity_total: dict[str, int] = defaultdict(int)

        # Safety tracking
        safety_violation_count = 0
        cases_requiring_referral = 0
        cases_with_life_threatening = 0
        clinical_confidences: list[float] = []

        total_iterations = 0
        total_time = 0.0
        evaluated = 0  # cases with ground truth

        for result, case in zip(results, cases, strict=True):
            # Track safety metrics for ALL cases
            safety_violation_count += len(result.safety_violations)
            if result.safety_flags:
                cases_requiring_referral += 1
            if any(h.is_life_threatening for h in result.state.active_hypotheses):
                cases_with_life_threatening += 1
            clinical_confidences.append(result.clinical_confidence)

            # Track convergence and performance
            total_iterations += result.iterations
            total_time += result.time_seconds

            ground_truth = case.ground_truth_diagnosis
            if ground_truth is None:
                # Skip evaluation for cases without ground truth
                continue
            evaluated += 1

            ranked = result.differential.ranked_list
            ranked_names = [r.disease_name.lower() for r in ranked]
            gt_lower = ground_truth.lower()

            # Recall@K
            hit_at_1 = gt_lower in ranked_names[:1]
            hit_at_5 = gt_lower in ranked_names[:5]
            hit_at_10 = gt_lower in ranked_names[:10]

            if hit_at_1:
                recall_at_1 += 1
            if hit_at_5:
                recall_at_5 += 1
            if hit_at_10:
                recall_at_10 += 1
                correct += 1

            # MRR
            if gt_lower in ranked_names:
                rank_pos = ranked_names.index(gt_lower) + 1
                reciprocal_ranks.append(1.0 / rank_pos)
            else:
                reciprocal_ranks.append(0.0)

                # Per-case error analysis
                error_cases.append(
                    {
                        "case_id": case.case_id,
                        "ground_truth": ground_truth,
                        "top_prediction": ranked_names[0] if ranked_names else "none",
                        "ranked_list": ranked_names[:5],
                        "iterations": result.iterations,
                        "converged": result.converged,
                        "clinical_confidence": result.clinical_confidence,
                        "safety_flags": len(result.safety_flags),
                    }
                )

            # Stratum tracking — organ system
            gt_organ = self._find_organ_system(result, gt_lower)
            organ_total[gt_organ] += 1
            if hit_at_10:
                organ_correct[gt_organ] += 1

            # Rarity tier
            gt_rarity = self._find_rarity_tier(result, gt_lower)
            rarity_total[gt_rarity] += 1
            if hit_at_10:
                rarity_correct[gt_rarity] += 1

            # Complexity
            complexity_key = case.presentation_complexity
            complexity_total[complexity_key] += 1
            if hit_at_10:
                complexity_correct[complexity_key] += 1

        # Use evaluated count (cases with ground truth) for rate calculations
        denom = max(1, evaluated)

        mrr = sum(reciprocal_ranks) / denom if reciprocal_ranks else 0.0

        accuracy_by_organ = {k: organ_correct[k] / organ_total[k] for k in organ_total if organ_total[k] > 0}
        accuracy_by_rarity = {k: rarity_correct[k] / rarity_total[k] for k in rarity_total if rarity_total[k] > 0}
        accuracy_by_complexity = {
            k: complexity_correct[k] / complexity_total[k] for k in complexity_total if complexity_total[k] > 0
        }

        # Wilson confidence intervals
        recall_1_rate = recall_at_1 / denom
        recall_5_rate = recall_at_5 / denom
        rrr_rate = sum(reciprocal_ranks) / denom if reciprocal_ranks else 0.0

        r1_ci = self._wilson_ci(recall_at_1, denom)
        r5_ci = self._wilson_ci(recall_at_5, denom)
        mrr_ci = self._wilson_ci_mrr(reciprocal_ranks)

        avg_clinical_confidence = sum(clinical_confidences) / len(clinical_confidences) if clinical_confidences else 0.0

        return EvaluationMetrics(
            recall_at_1=recall_1_rate,
            recall_at_5=recall_5_rate,
            recall_at_10=recall_at_10 / denom,
            mrr=rrr_rate,
            accuracy_by_organ=accuracy_by_organ,
            accuracy_by_rarity=accuracy_by_rarity,
            accuracy_by_complexity=accuracy_by_complexity,
            avg_iterations=total_iterations / max(1, total),
            avg_time_seconds=total_time / max(1, total),
            total_cases=total,
            correct_cases=correct,
            recall_at_1_ci_lower=r1_ci[0],
            recall_at_1_ci_upper=r1_ci[1],
            recall_at_5_ci_lower=r5_ci[0],
            recall_at_5_ci_upper=r5_ci[1],
            mrr_ci_lower=mrr_ci[0],
            mrr_ci_upper=mrr_ci[1],
            safety_violation_count=safety_violation_count,
            cases_requiring_referral=cases_requiring_referral,
            cases_with_life_threatening_conditions=cases_with_life_threatening,
            avg_clinical_confidence=avg_clinical_confidence,
            error_cases=error_cases,
        )

    def compare_baseline(
        self,
        multi_agent_metrics: EvaluationMetrics,
        single_agent_metrics: EvaluationMetrics,
    ) -> dict[str, float | str]:
        """Compare multi-agent system vs single-agent baseline.

        Returns:
            Dict of metric_name -> delta (multi-agent minus baseline) with
            statistical significance indicators where applicable.
        """
        deltas: dict[str, float | str] = {
            "recall_at_1_delta": multi_agent_metrics.recall_at_1 - single_agent_metrics.recall_at_1,
            "recall_at_5_delta": multi_agent_metrics.recall_at_5 - single_agent_metrics.recall_at_5,
            "recall_at_10_delta": multi_agent_metrics.recall_at_10 - single_agent_metrics.recall_at_10,
            "mrr_delta": multi_agent_metrics.mrr - single_agent_metrics.mrr,
            "avg_time_delta": multi_agent_metrics.avg_time_seconds - single_agent_metrics.avg_time_seconds,
            "avg_iterations_delta": multi_agent_metrics.avg_iterations - single_agent_metrics.avg_iterations,
        }

        # Check if Recall@1 improvement is statistically significant
        # (non-overlapping CIs as a simple test)
        multi_ci = (
            multi_agent_metrics.recall_at_1_ci_lower,
            multi_agent_metrics.recall_at_1_ci_upper,
        )
        single_ci = (
            single_agent_metrics.recall_at_1_ci_lower,
            single_agent_metrics.recall_at_1_ci_upper,
        )
        if multi_ci[1] > single_ci[0] and single_ci[1] > multi_ci[0]:
            deltas["recall_at_1_significance"] = "not_significant"
        else:
            deltas["recall_at_1_significance"] = "significant"

        return deltas

    @staticmethod
    def _wilson_ci(successes: int, total: int) -> tuple[float, float]:
        """Compute Wilson score 95% confidence interval for a proportion.

        Args:
            successes: Number of successes.
            total: Total number of trials.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        if total == 0:
            return 0.0, 1.0
        p = successes / total
        z2 = _Z_95 * _Z_95
        n = total
        denom = 1.0 + z2 / n
        center = (p + z2 / (2 * n)) / denom
        spread = _Z_95 * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
        return max(0.0, center - spread), min(1.0, center + spread)

    @staticmethod
    def _wilson_ci_mrr(reciprocal_ranks: list[float]) -> tuple[float, float]:
        """Compute approximate 95% CI for MRR using normal approximation.

        Args:
            reciprocal_ranks: List of reciprocal rank values.

        Returns:
            Tuple of (lower, upper) bounds.
        """
        if not reciprocal_ranks:
            return 0.0, 1.0
        n = len(reciprocal_ranks)
        mean = sum(reciprocal_ranks) / n
        if n < 2:
            return 0.0, 1.0
        variance = sum((r - mean) ** 2 for r in reciprocal_ranks) / (n - 1)
        se = math.sqrt(variance / n)
        return max(0.0, mean - _Z_95 * se), min(1.0, mean + _Z_95 * se)

    @staticmethod
    def _find_organ_system(result: DiagnosisResult, gt_lower: str) -> str:
        """Find organ system of the ground truth from the differential."""
        for ranked in result.differential.ranked_list:
            if ranked.disease_name.lower() == gt_lower:
                for hyp in result.state.active_hypotheses:
                    if hyp.disease_name.lower() == gt_lower:
                        return hyp.organ_system
        # Check eliminated hypotheses
        for hyp in result.state.active_hypotheses:
            if hyp.disease_name.lower() == gt_lower:
                return hyp.organ_system
        return "unknown"

    @staticmethod
    def _find_rarity_tier(result: DiagnosisResult, gt_lower: str) -> str:
        """Find rarity tier of the ground truth from the differential."""
        for hyp in result.state.active_hypotheses:
            if hyp.disease_name.lower() == gt_lower:
                return hyp.rarity_tier
        return "unknown"

    @staticmethod
    def _empty_metrics() -> EvaluationMetrics:
        return EvaluationMetrics(
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            accuracy_by_organ={},
            accuracy_by_rarity={},
            accuracy_by_complexity={},
            avg_iterations=0.0,
            avg_time_seconds=0.0,
            total_cases=0,
            correct_cases=0,
        )
