"""ControllerOrchestrator — central controller for the 3-tier agent system."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .differential import DifferentialDiagnosisManager
from .state import ConvergenceStatus, RareDiseaseState
from .types import (
    DifferentialEntry,
    DiseaseRarity,
    OrchestrationResult,
    PatientCase,
)

logger = logging.getLogger(__name__)


class ControllerOrchestrator:
    """Central controller that decomposes a patient case into sub-agent
    tasks, sequences invocations, and detects convergence."""

    def __init__(
        self,
        symptom_analyzer: Any,
        test_interpreter: Any,
        literature_matcher: Any,
        differential: DifferentialDiagnosisManager,
        max_iterations: int = 10,
    ) -> None:
        self._symptom_analyzer = symptom_analyzer
        self._test_interpreter = test_interpreter
        self._literature_matcher = literature_matcher
        self._differential = differential
        self._max_iterations = max_iterations

    def run(self, case: PatientCase) -> OrchestrationResult:
        """Execute the full 3-tier diagnostic pipeline."""
        start = time.monotonic()

        # Tier 1 — Symptom Analysis
        logger.info("Tier 1: Symptom analysis for case %s", case.case_id)
        symptom_results = self._symptom_analyzer.analyze(case)

        # Tier 2 — Test Interpretation
        logger.info("Tier 2: Test interpretation for case %s", case.case_id)
        test_results = self._test_interpreter.interpret(case, [])

        # Tier 3 — Literature Matching
        logger.info("Tier 3: Literature matching for case %s", case.case_id)
        lit_results = self._literature_matcher.search(case)

        # Differential diagnosis with Bayesian updating
        differential = self._build_differential(case, symptom_results, test_results)
        state = self._differential.initialise(case, differential)

        iterations = 0
        for iteration in range(self._max_iterations):
            state.iteration_count = iteration
            evidence: list[Any] = []
            evidence.extend(test_results.get("evidence", []))
            differential = self._differential.update(state, evidence, differential)
            pruned = self._differential.prune(state)
            if pruned:
                logger.debug("Pruned %d low-probability hypotheses", len(pruned))

            if self._differential.has_converged(state):
                state.convergence_status = ConvergenceStatus.CONVERGED
                logger.info(
                    "Converged after %d iterations for case %s",
                    iteration + 1,
                    case.case_id,
                )
                break
        else:
            state.convergence_status = ConvergenceStatus.MAX_ITERATIONS_REACHED

        latency_ms = (time.monotonic() - start) * 1000

        return OrchestrationResult(
            case_id=case.case_id,
            differential=differential,
            convergence_status=state.convergence_status,
            iterations=state.iteration_count + 1,
            total_latency_ms=latency_ms,
            sub_agent_results={
                "symptom_analysis": symptom_results,
                "test_interpretation": test_results,
                "literature_matching": lit_results,
            },
        )

    # ------------------------------------------------------------------ #

    def _build_differential(
        self,
        case: PatientCase,
        symptom_results: dict[str, Any],
        test_results: dict[str, Any],
    ) -> list[DifferentialEntry]:
        """Build initial differential from symptom analysis scores."""
        matrix = symptom_results.get("probability_matrix", {})
        differential: list[DifferentialEntry] = []
        for disease_id, score in sorted(matrix.items(), key=lambda x: x[1], reverse=True)[:20]:
            differential.append(
                DifferentialEntry(
                    disease_id=disease_id,
                    disease_name=disease_id,
                    organ_systems=[],
                    rarity=DiseaseRarity.RARE if score < 0.5 else DiseaseRarity.UNCOMMON,
                    posterior_probability=score,
                )
            )
        return differential if differential else [
            DifferentialEntry(
                disease_id="unknown",
                disease_name="Undifferentiated rare disease",
                organ_systems=[],
                rarity=DiseaseRarity.ULTRA_RARE,
                posterior_probability=1.0,
            )
        ]
