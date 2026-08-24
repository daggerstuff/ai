from __future__ import annotations

import logging
from typing import Any

from ..types import Evidence, OrganSystem, PatientCase

"""Test Interpreter Sub-Agent — interprets lab/imaging/genetic results."""

logger = logging.getLogger(__name__)

THRESHOLD_BROAD_DIFFERENTIAL = 5


class TestInterpreterAgent:
    """Maps test findings to disease-specific diagnostic criteria,
    applies Bayesian updating, and requests additional tests when needed."""

    def __init__(self, knowledge_base: Any) -> None:
        self._kb = knowledge_base

    def interpret(self, case: PatientCase, prior_differential: list[Any]) -> dict[str, Any]:
        """Interpret test results and update the differential."""
        evidence: list[Evidence] = []
        broad_differential = len(prior_differential) > THRESHOLD_BROAD_DIFFERENTIAL

        for result in case.test_results:
            for finding in result.findings:
                matches = (
                    self._kb.search_by_organ_system(
                        finding.organ_systems[0] if finding.organ_systems else OrganSystem.NERVOUS,
                        top_k=5,
                    )
                    if finding.organ_systems
                    else []
                )
                for match in matches:
                    weight = 0.7 if finding.is_abnormal else 0.2
                    evidence.append(
                        Evidence(
                            source="lab",
                            content=(
                                f"Finding: {finding.test_name} = {finding.value} (abnormal={finding.is_abnormal})"
                            ),
                            weight=weight,
                            disease_id=match.disease_id,
                        )
                    )

        requested_tests: list[str] = []
        if broad_differential:
            requested_tests.append("genetic_panel")
            requested_tests.append("advanced_imaging")

        # Bayesian update placeholder
        updated_scores: dict[str, float] = {}
        for ev in evidence:
            if ev.disease_id:
                updated_scores[ev.disease_id] = updated_scores.get(ev.disease_id, 0.0) + ev.weight

        return {
            "evidence": evidence,
            "requested_additional_tests": requested_tests,
            "updated_scores": updated_scores,
            "differential_was_broad": broad_differential,
        }
