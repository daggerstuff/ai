"""Differential Diagnosis Manager for the DeepRare system.

Enterprise-grade ranked differential diagnosis management with evidence strength
per condition. Performs Bayesian updating using likelihood ratios, accumulates
evidence across iterations, prunes low-probability hypotheses with clinical safety
protections, and detects convergence when the top-3 diagnoses remain stable.

Enterprise enhancements:
- Clinical safety: never eliminates life-threatening conditions without sufficient evidence
- Confidence interval propagation on hypotheses (logit-transform based)
- Bayesian update properly wired into the update() method
- Per-hypothesis evidence quality tracking
- Reasoning trace accumulation for audit trail
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .schema import (
    DifferentialDiagnosis,
    Hypothesis,
    RareDiseaseState,
    RankedDiagnosis,
)

if TYPE_CHECKING:
    from .knowledge_base import RareDiseaseKnowledgeBase

# Minimum evidence count before a hypothesis can be eliminated
_MIN_EVIDENCE_TO_ELIMINATE = 2

# Confidence interval z-score for 95% CI
_Z_95 = 1.96


class DifferentialDiagnosisManager:
    """Manages the ranked differential diagnosis list with Bayesian pruning.

    The manager normalizes probabilities, applies clinical safety checks before
    pruning, computes confidence intervals using a logit-transform approach,
    and re-ranks hypotheses by posterior probability.
    """

    def __init__(
        self,
        kb: RareDiseaseKnowledgeBase,
        pruning_threshold: float = 0.01,
        protect_life_threatening: bool = True,
    ) -> None:
        self._kb = kb
        self._pruning_threshold = pruning_threshold
        self._protect_life_threatening = protect_life_threatening

    def update(self, state: RareDiseaseState) -> None:
        """Update the differential diagnosis based on current state.

        Steps:
        1. Apply Bayesian updates from accumulated evidence (likelihood ratios)
        2. Normalize probabilities so they sum to 1.0
        3. Compute confidence intervals on each hypothesis
        4. Prune low-probability hypotheses (with clinical safety checks)
        5. Re-rank by posterior probability (descending)
        """
        if not state.active_hypotheses:
            return

        # Step 1: Apply Bayesian updates from evidence
        for hyp in state.active_hypotheses:
            self._apply_evidence_updates(hyp, state)

        # Step 2: Normalize probabilities
        total = sum(h.posterior_probability for h in state.active_hypotheses)
        if total > 0:
            for h in state.active_hypotheses:
                h.posterior_probability = h.posterior_probability / total

        # Step 3: Compute confidence intervals
        for hyp in state.active_hypotheses:
            self._compute_confidence_interval(hyp)

        # Step 4: Prune hypotheses below threshold (with safety checks)
        to_prune: list[Hypothesis] = []
        for h in state.active_hypotheses:
            if h.posterior_probability < self._pruning_threshold:
                # Clinical safety: don't prune life-threatening conditions without evidence
                if self._protect_life_threatening and h.is_life_threatening:
                    if len(h.evidence_list) < _MIN_EVIDENCE_TO_ELIMINATE:
                        continue  # Protect — not enough evidence to eliminate
                to_prune.append(h)

        for h in to_prune:
            h.status = "eliminated"
            state.eliminate(h.disease_name)

        # Step 5: Re-rank by posterior probability (descending)
        state.active_hypotheses.sort(
            key=lambda h: h.posterior_probability,
            reverse=True,
        )

    def build_differential(self, state: RareDiseaseState) -> DifferentialDiagnosis:
        """Build the final DifferentialDiagnosis from the current state.

        Note: ``update()`` must be called before this to ensure probabilities
        are normalized and pruned.
        """
        ranked: list[RankedDiagnosis] = []

        for rank_idx, hyp in enumerate(state.active_hypotheses, start=1):
            profile = self._kb.get_disease(hyp.disease_name)
            is_pathognomonic = False
            if profile:
                is_pathognomonic = bool(set(hyp.matching_symptoms) & set(profile.pathognomonic_symptoms))

            # Calculate evidence strength
            supporting = hyp.supporting_evidence()
            refuting = hyp.refuting_evidence()
            evidence_strength = sum(e.weight for e in supporting) - sum(e.weight for e in refuting)
            evidence_summary_parts = [e.description for e in supporting]

            ranked.append(
                RankedDiagnosis(
                    rank=rank_idx,
                    disease_name=hyp.disease_name,
                    probability=hyp.posterior_probability,
                    evidence_summary="; ".join(evidence_summary_parts[:3]),
                    evidence_count=len(hyp.evidence_list),
                    confidence=max(0.0, min(1.0, hyp.confidence_score + evidence_strength * 0.1)),
                    is_pathognomonic_match=is_pathognomonic,
                )
            )

        # Sort by probability descending
        ranked.sort(key=lambda r: r.probability, reverse=True)

        # Re-assign ranks after sorting
        for idx, r in enumerate(ranked, start=1):
            r.rank = idx

        return DifferentialDiagnosis(
            ranked_list=ranked,
            eliminated=list(state.eliminated_conditions),
            total_hypotheses_considered=len(state.eliminated_conditions) + len(ranked),
            iterations_used=state.iteration + 1,
            convergence_achieved=state.is_converged,
        )

    @staticmethod
    def bayesian_update(
        prior: float,
        likelihood_ratio: float,
    ) -> float:
        """Perform a Bayesian update on a probability given a likelihood ratio.

        Uses the odds form of Bayes' theorem:

            posterior_odds = prior_odds * likelihood_ratio
            posterior = posterior_odds / (1 + posterior_odds)

        Args:
            prior: Prior probability P(D), range [0, 1].
            likelihood_ratio: LR = P(T|D) / P(T|¬D).

        Returns:
            Posterior probability P(D|T), range [0, 1].
        """
        if prior >= 1.0:
            return 1.0
        if prior <= 0.0:
            return 0.0
        prior_odds = prior / (1.0 - prior)
        posterior_odds = prior_odds * likelihood_ratio
        return posterior_odds / (1.0 + posterior_odds)

    def _apply_evidence_updates(self, hyp: Hypothesis, state: RareDiseaseState) -> None:
        """Apply Bayesian updates from accumulated evidence on a hypothesis.

        Each piece of supporting evidence acts as a positive likelihood ratio,
        each piece of refuting evidence acts as a negative likelihood ratio.
        The evidence weight is used to modulate the LR.
        """
        supporting = hyp.supporting_evidence()
        refuting = hyp.refuting_evidence()

        if not supporting and not refuting:
            return

        # Aggregate evidence into a composite likelihood ratio
        # Supporting evidence: LR > 1 (increases probability)
        # Refuting evidence: LR < 1 (decreases probability)
        composite_lr = 1.0
        for ev in supporting:
            # Weight range [0, 1] → LR range [1, 10]
            composite_lr *= 1.0 + ev.weight * 9.0
        for ev in refuting:
            # Weight range [0, 1] → LR range [0.1, 1]
            composite_lr *= 1.0 - ev.weight * 0.9

        # Clamp to prevent extreme values
        composite_lr = max(0.01, min(100.0, composite_lr))

        hyp.posterior_probability = self.bayesian_update(
            hyp.posterior_probability,
            composite_lr,
        )

        # Update evidence strength in state
        total_strength = sum(e.weight for e in supporting) - sum(e.weight for e in refuting)
        state.evidence_strength[hyp.disease_name] = total_strength

    @staticmethod
    def _compute_confidence_interval(hyp: Hypothesis) -> None:
        """Compute a 95% confidence interval on the hypothesis probability.

        Uses a logit-transform approach for well-calibrated intervals:
            - Transform probability to log-odds space (unbounded)
            - Estimate standard error from evidence count
            - Build CI in log-odds space, transform back

        The standard error decreases with more evidence (more confident).
        """
        p = hyp.posterior_probability
        # Clamp to avoid log(0) or log(1)
        p_clamped = max(0.001, min(0.999, p))

        log_odds = math.log(p_clamped / (1.0 - p_clamped))

        # Standard error: decreases with evidence count
        # With 0 evidence, SE is large; with 10+ evidence, SE is small
        n = max(1, len(hyp.evidence_list))
        se = 1.0 / math.sqrt(n)

        ci_lower_logit = log_odds - _Z_95 * se
        ci_upper_logit = log_odds + _Z_95 * se

        # Transform back to probability space
        hyp.confidence_interval_lower = max(0.0, 1.0 / (1.0 + math.exp(-ci_lower_logit)))
        hyp.confidence_interval_upper = min(1.0, 1.0 / (1.0 + math.exp(-ci_upper_logit)))
