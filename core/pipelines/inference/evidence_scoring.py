"""
PIX-3912: Rank Stage — Evidence-Based Scoring

Score each candidate diagnosis against patient evidence.
- Scoring factors: symptom match (0.4), typical presentation (0.25), test results (0.2), progression pattern (0.15)
- Evidence chain: which specific findings support each diagnosis?
- Final ranking: weighted combination of retrieval score + evidence score
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ai.memory.therapeutic_concept_hierarchy import TherapeuticConceptHierarchy

from .candidate_retrieval import CandidateDiagnosis, RetrievalEvidence


@dataclass
class EvidenceFinding:
    """A single finding that supports or contradicts a diagnosis."""

    finding_type: str  # "symptom", "test", "progression", "typical_presentation"
    description: str
    weight: float  # contribution to score
    direction: str = "support"  # "support" | "contradict"
    confidence: float = 1.0


@dataclass
class ScoredDiagnosis:
    """A diagnosis candidate with full evidence-based scoring."""

    condition_id: str
    condition_name: str
    retrieval_score: float
    evidence_score: float
    final_score: float
    symptom_score: float
    typical_presentation_score: float
    test_result_score: float
    progression_score: float
    findings: list[EvidenceFinding] = field(default_factory=list)
    hierarchy_path: list[str] = field(default_factory=list)
    confidence: float = 0.0


class EvidenceScoringEngine:
    """
    Evidence-based scoring engine for the Rank stage.

    Weights
    -------
    symptom_match          : 0.40
    typical_presentation   : 0.25
    test_results           : 0.20
    progression_pattern    : 0.15
    """

    WEIGHTS = {
        "symptom_match": 0.40,
        "typical_presentation": 0.25,
        "test_results": 0.20,
        "progression_pattern": 0.15,
    }

    def __init__(
        self,
        hierarchy: TherapeuticConceptHierarchy,
        typical_presentations: dict[str, list[str]] | None = None,
        test_profiles: dict[str, list[dict[str, Any]]] | None = None,
        progression_profiles: dict[str, list[str]] | None = None,
    ):
        self.hierarchy = hierarchy
        self.typical_presentations = typical_presentations or {}
        self.test_profiles = test_profiles or {}
        self.progression_profiles = progression_profiles or {}

    def score(
        self,
        candidates: list[CandidateDiagnosis],
        patient_presentation: str,
        test_results: list[dict[str, Any]] | None = None,
        progression_notes: str | None = None,
    ) -> list[ScoredDiagnosis]:
        """
        Score each candidate diagnosis against patient evidence.

        Parameters
        ----------
        candidates : retrieved candidates from Memorize stage
        patient_presentation : free-text clinical description
        test_results : list of dicts with keys: name, value, normal_range, flag
        progression_notes : free-text notes on symptom progression
        """
        scored: list[ScoredDiagnosis] = []

        for candidate in candidates:
            findings: list[EvidenceFinding] = []

            # 1. Symptom match score
            symptom_score, symptom_findings = self._score_symptom_match(
                candidate.condition_id, patient_presentation
            )
            findings.extend(symptom_findings)

            # 2. Typical presentation score
            typical_score, typical_findings = self._score_typical_presentation(
                candidate.condition_id, patient_presentation
            )
            findings.extend(typical_findings)

            # 3. Test result score
            test_score, test_findings = self._score_test_results(
                candidate.condition_id, test_results or []
            )
            findings.extend(test_findings)

            # 4. Progression pattern score
            progression_score, progression_findings = self._score_progression(
                candidate.condition_id, progression_notes or ""
            )
            findings.extend(progression_findings)

            # Combined evidence score
            evidence_score = (
                self.WEIGHTS["symptom_match"] * symptom_score
                + self.WEIGHTS["typical_presentation"] * typical_score
                + self.WEIGHTS["test_results"] * test_score
                + self.WEIGHTS["progression_pattern"] * progression_score
            )

            # Final score blends retrieval + evidence
            final_score = 0.5 * candidate.retrieval_score + 0.5 * evidence_score

            # Confidence based on number and strength of supporting findings
            support_findings = [f for f in findings if f.direction == "support"]
            confidence = (
                np.mean([f.confidence for f in support_findings])
                if support_findings
                else 0.0
            )

            scored.append(
                ScoredDiagnosis(
                    condition_id=candidate.condition_id,
                    condition_name=candidate.condition_name,
                    retrieval_score=candidate.retrieval_score,
                    evidence_score=evidence_score,
                    final_score=final_score,
                    symptom_score=symptom_score,
                    typical_presentation_score=typical_score,
                    test_result_score=test_score,
                    progression_score=progression_score,
                    findings=findings,
                    hierarchy_path=candidate.hierarchy_path,
                    confidence=confidence,
                )
            )

        # Sort by final score descending
        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored

    def _score_symptom_match(
        self, condition_id: str, patient_text: str
    ) -> tuple[float, list[EvidenceFinding]]:
        """Score how many of the condition's symptoms appear in the patient text."""
        findings: list[EvidenceFinding] = []
        leaves = self.hierarchy.get_leaves(condition_id)
        if not leaves:
            return 0.25, findings  # penalize if no symptom definitions available

        patient_lower = patient_text.lower()
        matched = 0
        total_weight = 0

        for leaf in leaves:
            # Simple keyword matching (can be enhanced with NLP)
            symptom_name = leaf.name.lower()
            keywords = re.findall(r"\b\w+\b", symptom_name)
            match_score = sum(1 for kw in keywords if kw in patient_lower) / max(len(keywords), 1)
            weight = leaf.specificity_weight
            total_weight += weight

            if match_score > 0.5:
                matched += weight
                findings.append(
                    EvidenceFinding(
                        finding_type="symptom",
                        description=f"Patient reports '{leaf.name}'",
                        weight=weight * match_score,
                        direction="support",
                        confidence=match_score,
                    )
                )
            elif match_score > 0.0:
                findings.append(
                    EvidenceFinding(
                        finding_type="symptom",
                        description=f"Partial match for '{leaf.name}'",
                        weight=weight * match_score * 0.5,
                        direction="support",
                        confidence=match_score * 0.5,
                    )
                )

        # Also check for contradictions (symptoms explicitly denied)
        for leaf in leaves:
            symptom_name = leaf.name.lower()
            negation_patterns = [
                rf"\bno\s+{re.escape(symptom_name)}\b",
                rf"\bdenies\s+{re.escape(symptom_name)}\b",
                rf"\bwithout\s+{re.escape(symptom_name)}\b",
            ]
            for pattern in negation_patterns:
                if re.search(pattern, patient_lower):
                    findings.append(
                        EvidenceFinding(
                            finding_type="symptom",
                            description=f"Patient denies '{leaf.name}'",
                            weight=leaf.specificity_weight * 0.3,
                            direction="contradict",
                            confidence=0.7,
                        )
                    )

        score = matched / max(total_weight, 1e-6)
        return min(score, 1.0), findings

    def _score_typical_presentation(
        self, condition_id: str, patient_text: str
    ) -> tuple[float, list[EvidenceFinding]]:
        """Score alignment with typical presentation patterns."""
        findings: list[EvidenceFinding] = []
        typical = self.typical_presentations.get(condition_id, [])
        if not typical:
            return 0.5, findings

        patient_lower = patient_text.lower()
        matched = 0
        for pattern in typical:
            pattern_lower = pattern.lower()
            keywords = re.findall(r"\b\w+\b", pattern_lower)
            match_ratio = sum(1 for kw in keywords if kw in patient_lower) / max(len(keywords), 1)
            if match_ratio > 0.5:
                matched += 1
                findings.append(
                    EvidenceFinding(
                        finding_type="typical_presentation",
                        description=f"Matches typical pattern: '{pattern}'",
                        weight=match_ratio,
                        direction="support",
                        confidence=match_ratio,
                    )
                )

        score = matched / len(typical)
        return score, findings

    def _score_test_results(
        self, condition_id: str, test_results: list[dict[str, Any]]
    ) -> tuple[float, list[EvidenceFinding]]:
        """Score consistency with expected test result profiles."""
        findings: list[EvidenceFinding] = []
        profile = self.test_profiles.get(condition_id, [])
        if not test_results or not profile:
            return 0.5, findings

        matched = 0
        for test in test_results:
            test_name = test.get("name", "").lower()
            test_flag = test.get("flag", "normal").lower()
            for expected in profile:
                expected_name = expected.get("name", "").lower()
                expected_flag = expected.get("expected_flag", "abnormal").lower()
                if test_name == expected_name:
                    if test_flag == expected_flag:
                        matched += 1
                        findings.append(
                            EvidenceFinding(
                                finding_type="test",
                                description=f"{test_name}: {test_flag} (expected)",
                                weight=1.0,
                                direction="support",
                                confidence=0.9,
                            )
                        )
                    else:
                        findings.append(
                            EvidenceFinding(
                                finding_type="test",
                                description=f"{test_name}: {test_flag} (expected {expected_flag})",
                                weight=0.5,
                                direction="contradict",
                                confidence=0.7,
                            )
                        )

        score = matched / max(len(profile), 1)
        return score, findings

    def _score_progression(
        self, condition_id: str, progression_notes: str
    ) -> tuple[float, list[EvidenceFinding]]:
        """Score consistency with expected progression patterns."""
        findings: list[EvidenceFinding] = []
        patterns = self.progression_profiles.get(condition_id, [])
        if not patterns or not progression_notes:
            return 0.5, findings

        notes_lower = progression_notes.lower()
        matched = 0
        for pattern in patterns:
            pattern_lower = pattern.lower()
            keywords = re.findall(r"\b\w+\b", pattern_lower)
            match_ratio = sum(1 for kw in keywords if kw in notes_lower) / max(len(keywords), 1)
            if match_ratio > 0.5:
                matched += 1
                findings.append(
                    EvidenceFinding(
                        finding_type="progression",
                        description=f"Matches progression: '{pattern}'",
                        weight=match_ratio,
                        direction="support",
                        confidence=match_ratio,
                    )
                )

        score = matched / len(patterns)
        return score, findings

    def add_typical_presentation(self, condition_id: str, patterns: list[str]) -> None:
        """Add typical presentation patterns for a condition."""
        self.typical_presentations.setdefault(condition_id, []).extend(patterns)

    def add_test_profile(self, condition_id: str, tests: list[dict[str, Any]]) -> None:
        """Add expected test result profile for a condition."""
        self.test_profiles.setdefault(condition_id, []).extend(tests)

    def add_progression_profile(self, condition_id: str, patterns: list[str]) -> None:
        """Add expected progression patterns for a condition."""
        self.progression_profiles.setdefault(condition_id, []).extend(patterns)
