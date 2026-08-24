"""Symptom Analyzer sub-agent for rare disease diagnosis.

Maps patient symptoms to rare disease phenotype patterns, identifies
pathognomonic symptoms, clusters co-occurring symptoms, and analyzes
temporal progression. Output is a symptom-disease probability matrix
that seeds the differential diagnosis.

Enterprise upgrades:
- HPO ontology term resolution and synonym expansion
- Severity-weighted probability scoring
- Age-of-onset matching against disease profiles
- Sex-based adjustments for X-linked conditions
- Fuzzy symptom name matching (SequenceMatcher)
- Confidence interval estimation for hypotheses

Based on the DeepRare architecture (arXiv 2506.20430).
"""

from __future__ import annotations

import math
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from ..schema import (
    DiseaseProfile,
    Hypothesis,
    PatientCase,
    RarityTier,
    SymptomAnalysisResult,
    SymptomProfile,
)

if TYPE_CHECKING:
    from ..knowledge_base import RareDiseaseKnowledgeBase

# X-linked diseases — higher prior probability for male patients
_XLINKED_DISEASES: frozenset[str] = frozenset(
    {
        "Duchenne Muscular Dystrophy",
        "Hemophilia A",
        "Fabry Disease",
        "Fragile X Syndrome",
        "Alport Syndrome",
        "Adrenoleukodystrophy",
    }
)

# Severity weights — high-severity symptoms carry more diagnostic signal
_SEVERITY_WEIGHTS: dict[str, float] = {
    "mild": 0.8,
    "moderate": 1.0,
    "severe": 1.3,
    "life_threatening": 1.6,
}

# Fuzzy match threshold for symptom name similarity
_FUZZY_THRESHOLD: float = 0.75


class SymptomAnalyzer:
    """Sub-agent that analyzes symptoms and maps them to rare disease phenotypes.

    Responsibilities:
    - Identify pathognomonic symptoms unique to specific diseases
    - Cluster co-occurring symptoms
    - Analyze temporal progression patterns
    - Generate initial disease hypotheses with probability estimates
    - Recommend additional patient history inquiries
    - Resolve HPO ontology terms for symptom standardization
    - Weight symptoms by clinical severity
    - Adjust priors for patient age and sex
    """

    def __init__(self, kb: RareDiseaseKnowledgeBase) -> None:
        self._kb = kb

    def analyze(self, case: PatientCase) -> SymptomAnalysisResult:
        """Analyze patient symptoms and return disease probability matrix."""
        symptoms = case.presenting_symptoms
        if not symptoms:
            return SymptomAnalysisResult(reasoning="No presenting symptoms provided for analysis.")

        # Expand symptom names with HPO synonyms and fuzzy matching
        expanded_symptoms = self._expand_symptoms(symptoms)
        symptom_names = [s.name for s in symptoms]

        match_fractions = self._kb.search_by_symptoms(expanded_symptoms)

        pathognomonic_found: list[str] = []
        symptom_clusters = self._cluster_symptoms(symptoms)
        temporal_analysis = self._analyze_temporal_progression(symptoms)

        probability_matrix: dict[str, float] = {}
        new_hypotheses: list[Hypothesis] = []
        recommended_inquiries: list[str] = []

        for disease_name, match_fraction in match_fractions.items():
            profile = self._kb.get_disease(disease_name)
            if not profile:
                continue

            # Check pathognomonic symptoms with fuzzy matching
            patho_boost = 0.0
            for symptom in symptoms:
                if self._fuzzy_match_any(symptom.name, profile.pathognomonic_symptoms):
                    if symptom.name not in pathognomonic_found:
                        pathognomonic_found.append(symptom.name)
                    patho_boost += 0.15

            # Severity-weighted match score
            severity_weight = self._compute_severity_weight(symptoms, profile)
            weighted_match = match_fraction * severity_weight

            # Age-of-onset adjustment
            age_factor = self._age_adjustment(case.patient_age, profile.typical_onset)

            # Sex adjustment for X-linked diseases
            sex_factor = self._sex_adjustment(case.patient_sex, disease_name)

            # Base prior from prevalence
            prevalence_prior = self._prevalence_to_prior(profile.rarity_tier)

            # Combined probability with all adjustments
            probability = min(
                0.95,
                weighted_match * 0.5 + patho_boost + prevalence_prior * 0.1 * age_factor * sex_factor,
            )

            probability_matrix[disease_name] = probability

            # Create hypothesis for diseases with meaningful probability
            if probability > 0.05:
                matching = self._find_matching_symptoms(symptom_names, profile)
                missing = self._find_missing_symptoms(symptom_names, profile)
                ci_lower, ci_upper = self._estimate_ci(probability, len(matching))

                hypothesis = Hypothesis(
                    disease_name=disease_name,
                    disease_id=profile.orpha_id,
                    prior_probability=prevalence_prior,
                    posterior_probability=probability,
                    organ_system=profile.organ_system,
                    rarity_tier=profile.rarity_tier,
                    matching_symptoms=matching,
                    missing_symptoms=missing[:5],
                    confidence_score=min(1.0, probability + patho_boost),
                    confidence_interval_lower=ci_lower,
                    confidence_interval_upper=ci_upper,
                    is_life_threatening=self._is_life_threatening(profile),
                    iteration_created=0,
                )
                new_hypotheses.append(hypothesis)

        # Recommend inquiries for missing high-value symptoms
        for hyp in new_hypotheses[:5]:
            for missing_symptom in hyp.missing_symptoms[:3]:
                inquiry = f"Check for presence of '{missing_symptom}' (relevant to {hyp.disease_name})"
                if inquiry not in recommended_inquiries:
                    recommended_inquiries.append(inquiry)

        # Sort hypotheses by posterior probability
        new_hypotheses.sort(key=lambda h: h.posterior_probability, reverse=True)

        reasoning = self._build_reasoning(
            symptom_names,
            match_fractions,
            pathognomonic_found,
            symptom_clusters,
            temporal_analysis,
            case,
        )

        return SymptomAnalysisResult(
            symptom_disease_matrix=probability_matrix,
            identified_pathognomonic=pathognomonic_found,
            symptom_clusters=symptom_clusters,
            temporal_progression_analysis=temporal_analysis,
            new_hypotheses=new_hypotheses,
            recommended_inquiries=recommended_inquiries[:10],
            reasoning=reasoning,
        )

    def _expand_symptoms(self, symptoms: list[SymptomProfile]) -> list[str]:
        """Expand symptom names using HPO synonyms and fuzzy matching.

        Returns the original symptom names plus any HPO synonyms found
        in the knowledge base ontology, improving match coverage.
        """
        expanded: list[str] = []
        for s in symptoms:
            expanded.append(s.name)
            # Try HPO term resolution
            hpo_term = self._kb.resolve_hpo_term(s.name)
            if hpo_term:
                synonyms = self._kb.get_hpo_synonyms(hpo_term)
                for syn in synonyms:
                    if syn not in expanded:
                        expanded.append(syn)
        return expanded

    def _fuzzy_match_any(self, symptom: str, candidates: list[str]) -> bool:
        """Check if symptom fuzzy-matches any candidate string."""
        symptom_lower = symptom.lower()
        for c in candidates:
            if symptom_lower == c.lower():
                return True
            ratio = SequenceMatcher(None, symptom_lower, c.lower()).ratio()
            if ratio >= _FUZZY_THRESHOLD:
                return True
        return False

    def _find_matching_symptoms(self, symptom_names: list[str], profile: DiseaseProfile) -> list[str]:
        """Find symptoms that match the disease profile (fuzzy)."""
        all_profile_symptoms = profile.common_symptoms + profile.pathognomonic_symptoms
        matching: list[str] = []
        for s in symptom_names:
            if self._fuzzy_match_any(s, all_profile_symptoms):
                matching.append(s)
        return matching

    def _find_missing_symptoms(self, symptom_names: list[str], profile: DiseaseProfile) -> list[str]:
        """Find profile symptoms not present in patient (fuzzy)."""
        all_profile_symptoms = profile.common_symptoms + profile.pathognomonic_symptoms
        missing: list[str] = []
        for ps in all_profile_symptoms:
            if not self._fuzzy_match_any(ps, symptom_names):
                missing.append(ps)
        return missing

    def _compute_severity_weight(self, symptoms: list[SymptomProfile], profile: DiseaseProfile) -> float:
        """Compute severity-weighted match multiplier.

        High-severity symptoms matching a disease profile increase
        the diagnostic signal. We compute the average severity weight
        of the patient's symptoms that match the profile.
        """
        all_profile_symptoms = profile.common_symptoms + profile.pathognomonic_symptoms
        matched_weights: list[float] = []
        for s in symptoms:
            if self._fuzzy_match_any(s.name, all_profile_symptoms):
                w = _SEVERITY_WEIGHTS.get(s.severity, 1.0)
                # Boost for pathognomonic matches
                if self._fuzzy_match_any(s.name, profile.pathognomonic_symptoms):
                    w *= 1.5
                matched_weights.append(w)
        if not matched_weights:
            return 1.0
        return sum(matched_weights) / len(matched_weights)

    def _age_adjustment(self, patient_age: int | None, typical_onset: str) -> float:
        """Adjust probability based on patient age vs typical onset.

        Returns a multiplier:
        - 1.0 if age aligns with typical onset
        - 0.3-0.7 if age is significantly outside onset range
        """
        if not typical_onset or patient_age is None:
            return 1.0

        onset_lower = typical_onset.lower()

        # Map onset descriptions to approximate age ranges (years)
        onset_ranges: dict[str, tuple[int, int]] = {
            "neonatal": (0, 1),
            "infancy": (0, 2),
            "early_childhood": (0, 6),
            "childhood": (0, 12),
            "adult": (18, 80),
            "variable": (0, 80),
            "adolescence": (10, 19),
            "young_adult": (18, 35),
            "middle_age": (35, 65),
        }

        for key, (low, high) in onset_ranges.items():
            if key in onset_lower:
                if low <= patient_age <= high:
                    return 1.0
                # Penalize for age mismatch, but don't eliminate
                distance = min(abs(patient_age - low), abs(patient_age - high))
                return max(0.3, 1.0 - distance * 0.03)

        return 1.0

    def _sex_adjustment(self, patient_sex: str, disease_name: str) -> float:
        """Adjust probability for X-linked diseases based on patient sex.

        X-linked recessive diseases are much more common in males.
        """
        if disease_name not in _XLINKED_DISEASES:
            return 1.0

        sex_lower = patient_sex.lower() if patient_sex else ""
        if sex_lower == "male":
            return 1.0
        elif sex_lower == "female":
            return 0.1  # X-linked recessive: females are carriers, rarely affected
        return 0.5  # Unknown sex — neutral

    def _estimate_ci(self, probability: float, evidence_count: int) -> tuple[float, float]:
        """Estimate a rough confidence interval for the probability.

        Uses a beta-distribution-inspired approach: more evidence
        narrows the CI. With 0 evidence the CI is wide; with 10+
        pieces of evidence it's narrow.
        """
        if evidence_count == 0:
            return (max(0.0, probability * 0.5), min(1.0, probability + (1 - probability) * 0.5))
        # Pseudo alpha/beta for a Beta prior
        alpha = 1.0 + evidence_count * probability
        beta = 1.0 + evidence_count * (1 - probability)
        # Approximate 95% CI using mean ± 2*std
        mean = alpha / (alpha + beta)
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        std = math.sqrt(variance)
        lower = max(0.0, mean - 2 * std)
        upper = min(1.0, mean + 2 * std)
        return (lower, upper)

    def _is_life_threatening(self, profile: DiseaseProfile) -> bool:
        """Determine if a disease is life-threatening based on rarity and organ system."""
        life_threatening_organs = {"cardiovascular", "neurological", "metabolic"}
        if profile.organ_system in life_threatening_organs:
            return True
        if profile.rarity_tier == "ultra_rare":
            return True
        return False

    def _cluster_symptoms(self, symptoms: list[SymptomProfile]) -> list[list[str]]:
        """Group symptoms that co-occur by category and temporal pattern."""
        clusters: dict[str, list[str]] = {}
        for s in symptoms:
            key = s.category
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(s.name)

        # Also cluster by temporal pattern
        for s in symptoms:
            temporal_key = f"onset:{s.onset}"
            if temporal_key not in clusters:
                clusters[temporal_key] = []
            if s.name not in clusters[temporal_key]:
                clusters[temporal_key].append(s.name)

        return [list(group) for group in clusters.values() if len(group) > 1]

    def _analyze_temporal_progression(self, symptoms: list[SymptomProfile]) -> str:
        """Analyze temporal patterns in symptom presentation."""
        if not symptoms:
            return "No symptoms to analyze temporally."

        onset_types = {s.onset for s in symptoms}
        progression_types = {s.progression for s in symptoms}

        parts: list[str] = []

        if "acute" in onset_types and "chronic" in onset_types:
            parts.append(
                "Mixed acute and chronic onset suggests possible acute-on-chronic or multi-phase disease presentation."
            )
        elif "acute" in onset_types:
            parts.append("Acute onset pattern may suggest infectious, metabolic, or acute neurological event.")
        elif "congenital" in onset_types:
            parts.append("Congenital onset suggests genetic or developmental disorder.")
        elif "chronic" in onset_types:
            parts.append("Chronic onset suggests neurodegenerative, metabolic, or progressive disease.")
        elif "infancy" in onset_types or "early_childhood" in onset_types:
            parts.append("Early-onset symptoms suggest congenital or developmental metabolic disorder.")

        if "stepwise" in progression_types:
            parts.append("Stepwise progression is characteristic of metabolic or neurodegenerative disorders.")
        elif "worsening" in progression_types:
            parts.append("Progressive worsening suggests a progressive neurodegenerative or metabolic condition.")
        elif "fluctuating" in progression_types:
            parts.append("Fluctuating course may suggest metabolic, autoimmune, or channelopathy.")

        return " ".join(parts) if parts else "Temporal analysis inconclusive."

    def _prevalence_to_prior(self, rarity: RarityTier) -> float:
        """Convert rarity tier to a base prior probability."""
        priors = {
            "ultra_rare": 0.005,
            "rare": 0.01,
            "less_common": 0.03,
            "moderately_rare": 0.05,
        }
        return priors.get(rarity, 0.01)

    def _build_reasoning(
        self,
        symptom_names: list[str],
        match_fractions: dict[str, float],
        pathognomonic: list[str],
        clusters: list[list[str]],
        temporal: str,
        case: PatientCase,
    ) -> str:
        """Build a narrative reasoning trace for transparency."""
        parts: list[str] = []
        parts.append(f"Analyzed {len(symptom_names)} presenting symptoms for {case.patient_age}yo {case.patient_sex}.")

        if pathognomonic:
            parts.append(f"Identified {len(pathognomonic)} pathognomonic symptom(s): {', '.join(pathognomonic)}.")

        top_matches = sorted(match_fractions.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_matches:
            match_str = ", ".join(f"{n} ({f:.2f})" for n, f in top_matches)
            parts.append(f"Top disease matches: {match_str}.")

        if clusters:
            parts.append(f"Identified {len(clusters)} symptom cluster(s).")

        parts.append(f"Temporal analysis: {temporal}")

        return " ".join(parts)
