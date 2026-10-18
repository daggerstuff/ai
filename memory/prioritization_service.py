# Prioritization Service
# Implements a thin wrapper that integrates the ReprioritizationEngine
# with the acquisition scoring pipeline. This is a lightweight integration
# intended for Workstream E (PIX-536).

from __future__ import annotations

from .acquisition_rubric import AcquisitionScore
from .reprioritization_engine import EvidenceItem, ReprioritizationEngine


def adjust_acquisition_priorities(
    base_score: AcquisitionScore,
    evidence: list[EvidenceItem],
) -> list[str]:
    """Return a reordered list of priority tier identifiers based on evidence.

    The function extracts the current priority tier from ``base_score.priority_tier``
    and treats it as the initial ordering. It then applies the ``ReprioritizationEngine``
    to compute a new ordering that reflects evaluation evidence.
    """
    # Map the priority tier to a simple list of identifiers; for now we use the tier name
    base_priority = [base_score.priority_tier.value]
    engine = ReprioritizationEngine(base_priority=base_priority)
    return engine.compute_new_order(evidence)


# Example usage (will be exercised in tests)
if __name__ == "__main__":
    sample_score = AcquisitionScore(
        therapeutic_relevance=8,
        data_structure_quality=7,
        training_integration=6,
        ethical_accessibility=9,
        overall_score=7.5,
    )
    sample_evidence = [
        EvidenceItem(source_id="eval1", evidence_type="gap", score=2.0, details={"task_id": "high"}),
    ]
