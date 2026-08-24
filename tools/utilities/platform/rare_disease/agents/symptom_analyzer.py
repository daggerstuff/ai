from __future__ import annotations

import logging
from typing import Any

from ..types import PatientCase, Symptom

"""Symptom Analyzer Sub-Agent — maps symptoms to rare-disease phenotype patterns."""

logger = logging.getLogger(__name__)

THRESHOLD_HIGH_MATCH = 0.7
THRESHOLD_MULTISYSTEM = 2


class SymptomAnalyzerAgent:
    """Maps symptoms to phenotype patterns and produces a structured
    symptom-disease probability matrix."""

    def __init__(self, knowledge_base: Any) -> None:
        self._kb = knowledge_base

    def analyze(self, case: PatientCase) -> dict[str, Any]:
        """Run the symptom-analysis pipeline and return structured results."""
        matrix: dict[str, float] = {}
        pathognomonic: list[str] = []
        clusters: list[str] = []

        for symptom in case.symptoms:
            matches = self._kb.search_by_symptom(symptom, top_k=10)
            for match in matches:
                matrix[match.disease_id] = matrix.get(match.disease_id, 0.0) + match.score
                if match.score > THRESHOLD_HIGH_MATCH:
                    disease = self._kb.get_disease(match.disease_id)
                    if disease:
                        pathognomonic.append(disease.name)

        clusters = self._detect_clusters(case.symptoms)

        # Normalize scores to [0, 1]
        max_score = max(matrix.values()) if matrix else 1.0
        if max_score > 0:
            matrix = {k: v / max_score for k, v in matrix.items()}

        return {
            "probability_matrix": matrix,
            "pathognomonic_symptoms": list(set(pathognomonic)),
            "symptom_clusters": clusters,
            "profile": self._structured_profile(case.symptoms),
        }

    # ------------------------------------------------------------------ #

    def _detect_clusters(self, symptoms: list[Symptom]) -> list[str]:
        clusters: list[str] = []
        systems = set()
        for s in symptoms:
            systems.update(s.organ_systems)
        if len(systems) > THRESHOLD_MULTISYSTEM:
            clusters.append("multisystem_overlap")
        return clusters

    @staticmethod
    def _structured_profile(symptoms: list[Symptom]) -> dict[str, Any]:
        onset_mixed = any(s.onset for s in symptoms)
        return {
            "symptom_count": len(symptoms),
            "severity_distribution": {
                "mild": sum(1 for s in symptoms if s.severity.name == "MILD"),
                "moderate": sum(1 for s in symptoms if s.severity.name == "MODERATE"),
                "severe": sum(1 for s in symptoms if s.severity.name == "SEVERE"),
            },
            "onset_pattern": "mixed" if onset_mixed else "uniform",
        }
