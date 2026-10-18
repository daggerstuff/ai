"""Top-level RareDiseasePipeline that wires all sub-agents together."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .agents.literature_matcher import LiteratureMatcherAgent
from .agents.symptom_analyzer import SymptomAnalyzerAgent
from .agents.test_interpreter import TestInterpreterAgent
from .differential import DifferentialDiagnosisManager
from .knowledge_base import InMemoryRareDiseaseKnowledgeBase
from .orchestrator import ControllerOrchestrator
from .types import (
    PatientCase,
)

logger = logging.getLogger(__name__)


@dataclass
class RareDiseasePipeline:
    """Full diagnostic pipeline: symptom analysis → test interpretation
    → literature matching → differential management → convergence."""

    orchestrator: ControllerOrchestrator
    max_iterations: int = 10

    def run(self, case: PatientCase) -> dict[str, Any]:
        """Execute the end-to-end diagnostic pipeline."""
        result = self.orchestrator.run(case)
        return {
            "case_id": result.case_id,
            "differential": [
                {
                    "disease_id": e.disease_id,
                    "disease_name": e.disease_name,
                    "posterior_probability": e.posterior_probability,
                    "organ_systems": [s.value for s in e.organ_systems],
                    "rarity": e.rarity.value,
                }
                for e in result.differential
            ],
            "convergence_status": result.convergence_status.value,
            "iterations": result.iterations,
            "total_latency_ms": result.total_latency_ms,
            "sub_agent_results": result.sub_agent_results,
        }


def build_default_pipeline() -> RareDiseasePipeline:
    """Construct a ready-to-use pipeline with in-memory components."""

    kb = InMemoryRareDiseaseKnowledgeBase()
    symptom_agent = SymptomAnalyzerAgent(kb)
    test_agent = TestInterpreterAgent(kb)
    literature_agent = LiteratureMatcherAgent(kb)
    differential = DifferentialDiagnosisManager()
    orchestrator = ControllerOrchestrator(
        symptom_agent,
        test_agent,
        literature_agent,
        differential,
    )
    return RareDiseasePipeline(orchestrator=orchestrator)
