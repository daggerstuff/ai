#!/usr/bin/env python3
"""
DSM-5 Therapeutic Accuracy Validation System
Validates clinical accuracy and therapeutic interventions against DSM-5 standards.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DSMDisorder(Enum):
    """Major DSM-5 disorder categories."""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    BIPOLAR = "bipolar"
    PTSD = "ptsd"
    OCD = "ocd"
    ADHD = "adhd"
    EATING_DISORDERS = "eating_disorders"
    SUBSTANCE_USE = "substance_use"
    PERSONALITY_DISORDERS = "personality_disorders"
    SCHIZOPHRENIA = "schizophrenia"


class TherapeuticApproach(Enum):
    """Evidence-based therapeutic approaches."""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    ACT = "acceptance_commitment_therapy"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    EMDR = "emdr"
    MINDFULNESS = "mindfulness"
    SOLUTION_FOCUSED = "solution_focused"


class ValidationCategory(Enum):
    """DSM-5 validation categories."""
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    SYMPTOM_RECOGNITION = "symptom_recognition"
    THERAPEUTIC_INTERVENTION = "therapeutic_intervention"
    PROFESSIONAL_BOUNDARIES = "professional_boundaries"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    EVIDENCE_BASED_PRACTICE = "evidence_based_practice"
    CRISIS_MANAGEMENT = "crisis_management"
    CULTURAL_COMPETENCY = "cultural_competency"


@dataclass
class DSMCriteria:
    """DSM-5 diagnostic criteria for a disorder."""
    disorder: DSMDisorder
    primary_symptoms: list[str]
    duration_requirements: str
    severity_specifiers: list[str]
    exclusion_criteria: list[str]
    associated_features: list[str]


@dataclass
class TherapeuticIntervention:
    """Therapeutic intervention assessment."""
    approach: TherapeuticApproach
    techniques: list[str]
    appropriateness_score: float
    evidence_level: str
    contraindications: list[str]


@dataclass
class DSMValidationResult:
    """DSM-5 validation result."""
    conversation_id: str
    overall_accuracy: float
    category_scores: dict[ValidationCategory, float]
    identified_disorders: list[DSMDisorder]
    therapeutic_interventions: list[TherapeuticIntervention]
    compliance_issues: list[str]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class DSM5AccuracyValidator:
    """
    DSM-5 therapeutic accuracy validation system.
    """

    def __init__(self):
        """Initialize the DSM-5 validator."""
        self.dsm_criteria = self._load_dsm_criteria()
        self.therapeutic_guidelines = self._load_therapeutic_guidelines()
        self.validation_history: list[DSMValidationResult] = []

    def _load_dsm_criteria(self) -> dict[DSMDisorder, DSMCriteria]:
        """Load DSM-5 diagnostic criteria."""
        return {
            DSMDisorder.DEPRESSION: DSMCriteria(
                disorder=DSMDisorder.DEPRESSION,
                primary_symptoms=[
                    "depressed mood", "anhedonia", "weight changes", "sleep disturbance",
                    "fatigue", "worthlessness", "concentration problems", "suicidal ideation"
                ],
                duration_requirements="2 weeks minimum",
                severity_specifiers=["mild", "moderate", "severe"],
                exclusion_criteria=["manic episodes", "substance use", "medical condition"],
                associated_features=["anxiety", "irritability", "cognitive impairment"]
            ),
            DSMDisorder.ANXIETY: DSMCriteria(
                disorder=DSMDisorder.ANXIETY,
                primary_symptoms=[
                    "excessive worry", "restlessness", "fatigue", "concentration difficulty",
                    "irritability", "muscle tension", "sleep disturbance"
                ],
                duration_requirements="6 months minimum",
                severity_specifiers=["mild", "moderate", "severe"],
                exclusion_criteria=["substance use", "medical condition", "other mental disorder"],
                associated_features=["avoidance", "physical symptoms", "panic attacks"]
            ),
            DSMDisorder.PTSD: DSMCriteria(
                disorder=DSMDisorder.PTSD,
                primary_symptoms=[
                    "intrusive memories", "avoidance", "negative cognitions", "hyperarousal",
                    "flashbacks", "nightmares", "emotional numbing"
                ],
                duration_requirements="1 month minimum",
                severity_specifiers=["mild", "moderate", "severe"],
                exclusion_criteria=["substance use", "medical condition"],
                associated_features=["dissociation", "depression", "anxiety"]
            ),
            DSMDisorder.BIPOLAR: DSMCriteria(
                disorder=DSMDisorder.BIPOLAR,
                primary_symptoms=[
                    "manic episodes", "depressive episodes", "elevated mood", "grandiosity",
                    "decreased sleep", "racing thoughts", "distractibility"
                ],
                duration_requirements="4 days (hypomania) or 7 days (mania)",
                severity_specifiers=["mild", "moderate", "severe"],
                exclusion_criteria=["substance use", "medical condition"],
                associated_features=["psychosis", "mixed features", "rapid cycling"]
            )
        }

    def _load_therapeutic_guidelines(self) -> dict[TherapeuticApproach, dict[str, Any]]:
        """Load evidence-based therapeutic guidelines."""
        return {
            TherapeuticApproach.CBT: {
                "primary_techniques": [
                    "cognitive restructuring", "behavioral activation", "exposure therapy",
                    "thought records", "behavioral experiments"
                ],
                "effective_for": [DSMDisorder.DEPRESSION, DSMDisorder.ANXIETY, DSMDisorder.PTSD],
                "evidence_level": "strong",
                "session_structure": "structured, goal-oriented",
                "contraindications": ["severe psychosis", "active substance use"]
            },
            TherapeuticApproach.DBT: {
                "primary_techniques": [
                    "mindfulness", "distress tolerance", "emotion regulation",
                    "interpersonal effectiveness"
                ],
                "effective_for": [DSMDisorder.PERSONALITY_DISORDERS],
                "evidence_level": "strong",
                "session_structure": "skills-based, group and individual",
                "contraindications": ["unwillingness to engage", "severe cognitive impairment"]
            },
            TherapeuticApproach.EMDR: {
                "primary_techniques": [
                    "bilateral stimulation", "resource installation", "trauma processing"
                ],
                "effective_for": [DSMDisorder.PTSD],
                "evidence_level": "strong",
                "session_structure": "8-phase protocol",
                "contraindications": ["dissociative disorders", "severe instability"]
            }
        }

    def validate_conversation(self, conversation: dict[str, Any]) -> DSMValidationResult:
        """
        Validate conversation against DSM-5 standards.

        Args:
            conversation: Conversation data to validate

        Returns:
            DSMValidationResult with detailed assessment
        """
        conversation_id = conversation.get("id", "unknown")
        logger.info(f"Validating DSM-5 accuracy for conversation {conversation_id}")

        # Analyze conversation content
        content = str(conversation.get("content", ""))
        turns = conversation.get("turns", [])

        # Identify potential disorders
        identified_disorders = self._identify_disorders(content, turns)

        # Assess therapeutic interventions
        interventions = self._assess_interventions(content, turns, identified_disorders)

        # Calculate category scores
        category_scores = self._calculate_category_scores(
            content, turns, identified_disorders, interventions
        )

        # Calculate overall accuracy
        overall_accuracy = sum(category_scores.values()) / len(category_scores)

        # Identify compliance issues
        compliance_issues = self._identify_compliance_issues(
            content, turns, identified_disorders, interventions
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            identified_disorders, interventions, compliance_issues, category_scores
        )

        result = DSMValidationResult(
            conversation_id=conversation_id,
            overall_accuracy=overall_accuracy,
            category_scores=category_scores,
            identified_disorders=identified_disorders,
            therapeutic_interventions=interventions,
            compliance_issues=compliance_issues,
            recommendations=recommendations
        )

        self.validation_history.append(result)
        return result

    def _identify_disorders(self, content: str, turns: list[dict]) -> list[DSMDisorder]:
        """Identify potential disorders mentioned in conversation."""
        identified = []
        content_lower = content.lower()

        # Check for disorder-specific keywords
        disorder_keywords = {
            DSMDisorder.DEPRESSION: [
                "depression", "depressed", "sad", "hopeless", "worthless",
                "suicidal", "anhedonia", "fatigue", "sleep problems"
            ],
            DSMDisorder.ANXIETY: [
                "anxiety", "anxious", "worry", "panic", "fear", "nervous",
                "restless", "tension", "phobia"
            ],
            DSMDisorder.PTSD: [
                "trauma", "ptsd", "flashback", "nightmare", "avoidance",
                "hypervigilant", "intrusive thoughts"
            ],
            DSMDisorder.BIPOLAR: [
                "bipolar", "manic", "mania", "mood swings", "elevated mood",
                "grandiose", "racing thoughts"
            ]
        }

        for disorder, keywords in disorder_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                identified.append(disorder)

        return identified

    def _assess_interventions(
        self,
        content: str,
        turns: list[dict],
        disorders: list[DSMDisorder]
    ) -> list[TherapeuticIntervention]:
        """Assess therapeutic interventions used."""
        interventions = []
        content_lower = content.lower()

        # Check for CBT techniques
        cbt_techniques = [
            "cognitive restructuring", "thought challenging", "behavioral activation",
            "exposure", "thought record", "cognitive distortion", "cognitive behavioral",
            "cbt", "negative thought patterns", "behavioral techniques"
        ]
        if any(technique in content_lower for technique in cbt_techniques):
            appropriateness = self._calculate_intervention_appropriateness(
                TherapeuticApproach.CBT, disorders
            )
            interventions.append(TherapeuticIntervention(
                approach=TherapeuticApproach.CBT,
                techniques=cbt_techniques,
                appropriateness_score=appropriateness,
                evidence_level="strong",
                contraindications=[]
            ))

        # Check for mindfulness techniques
        mindfulness_keywords = ["mindfulness", "meditation", "breathing", "grounding"]
        if any(keyword in content_lower for keyword in mindfulness_keywords):
            interventions.append(TherapeuticIntervention(
                approach=TherapeuticApproach.MINDFULNESS,
                techniques=mindfulness_keywords,
                appropriateness_score=0.8,
                evidence_level="moderate",
                contraindications=[]
            ))

        return interventions

    def _calculate_intervention_appropriateness(
        self,
        approach: TherapeuticApproach,
        disorders: list[DSMDisorder]
    ) -> float:
        """Calculate appropriateness of intervention for identified disorders."""
        if approach not in self.therapeutic_guidelines:
            return 0.5  # Neutral score for unknown approaches

        guidelines = self.therapeutic_guidelines[approach]
        effective_for = guidelines.get("effective_for", [])

        if not disorders:
            return 0.7  # Moderate score when no specific disorders identified

        # Calculate overlap between effective disorders and identified disorders
        overlap = len(set(disorders) & set(effective_for))
        total = len(disorders)

        return min(1.0, 0.5 + (overlap / total) * 0.5)

    def _calculate_category_scores(
        self,
        content: str,
        turns: list[dict],
        disorders: list[DSMDisorder],
        interventions: list[TherapeuticIntervention]
    ) -> dict[ValidationCategory, float]:
        """Calculate scores for each validation category."""
        scores = {}

        # Diagnostic accuracy
        scores[ValidationCategory.DIAGNOSTIC_ACCURACY] = self._assess_diagnostic_accuracy(
            content, disorders
        )

        # Symptom recognition
        scores[ValidationCategory.SYMPTOM_RECOGNITION] = self._assess_symptom_recognition(
            content, disorders
        )

        # Therapeutic intervention
        scores[ValidationCategory.THERAPEUTIC_INTERVENTION] = self._assess_therapeutic_quality(
            interventions
        )

        # Professional boundaries
        scores[ValidationCategory.PROFESSIONAL_BOUNDARIES] = self._assess_professional_boundaries(
            content, turns
        )

        # Ethical compliance
        scores[ValidationCategory.ETHICAL_COMPLIANCE] = self._assess_ethical_compliance(
            content, turns
        )

        # Evidence-based practice
        scores[ValidationCategory.EVIDENCE_BASED_PRACTICE] = self._assess_evidence_based_practice(
            interventions
        )

        # Crisis management
        scores[ValidationCategory.CRISIS_MANAGEMENT] = self._assess_crisis_management(
            content, turns
        )

        # Cultural competency
        scores[ValidationCategory.CULTURAL_COMPETENCY] = self._assess_cultural_competency(
            content, turns
        )

        return scores

    def _assess_diagnostic_accuracy(self, content: str, disorders: list[DSMDisorder]) -> float:
        """Assess diagnostic accuracy."""
        if not disorders:
            return 0.8  # Neutral score when no diagnosis attempted

        accuracy_score = 0.7  # Base score

        # Check for appropriate diagnostic language
        diagnostic_terms = ["symptoms", "criteria", "diagnosis", "assessment"]
        if any(term in content.lower() for term in diagnostic_terms):
            accuracy_score += 0.1

        # Check for symptom validation
        for disorder in disorders:
            if disorder in self.dsm_criteria:
                criteria = self.dsm_criteria[disorder]
                symptom_matches = sum(
                    1 for symptom in criteria.primary_symptoms
                    if symptom in content.lower()
                )
                if symptom_matches >= 2:
                    accuracy_score += 0.1

        return min(1.0, accuracy_score)

    def _assess_symptom_recognition(self, content: str, disorders: list[DSMDisorder]) -> float:
        """Assess symptom recognition accuracy."""
        recognition_score = 0.6  # Base score

        # Check for symptom exploration
        exploration_terms = ["tell me about", "describe", "when did", "how long"]
        if any(term in content.lower() for term in exploration_terms):
            recognition_score += 0.2

        # Check for symptom validation
        validation_terms = ["understand", "sounds difficult", "that must be"]
        if any(term in content.lower() for term in validation_terms):
            recognition_score += 0.2

        return min(1.0, recognition_score)

    def _assess_therapeutic_quality(self, interventions: list[TherapeuticIntervention]) -> float:
        """Assess quality of therapeutic interventions."""
        if not interventions:
            return 0.5  # Neutral score for no interventions

        # Average appropriateness scores
        avg_appropriateness = sum(i.appropriateness_score for i in interventions) / len(interventions)

        # Bonus for evidence-based approaches
        evidence_bonus = sum(
            0.1 for i in interventions if i.evidence_level == "strong"
        )

        return min(1.0, avg_appropriateness + evidence_bonus)

    def _assess_professional_boundaries(self, content: str, turns: list[dict]) -> float:
        """Assess professional boundary maintenance."""
        boundary_score = 0.9  # Start high, deduct for violations

        # Check for boundary violations
        violations = [
            "personal relationship", "friendship", "dating", "personal problems",
            "my experience", "happened to me"
        ]

        for violation in violations:
            if violation in content.lower():
                boundary_score -= 0.2

        return max(0.0, boundary_score)

    def _assess_ethical_compliance(self, content: str, turns: list[dict]) -> float:
        """Assess ethical compliance."""
        ethics_score = 0.8  # Base score

        # Check for informed consent language
        consent_terms = ["confidential", "limits", "consent", "agreement"]
        if any(term in content.lower() for term in consent_terms):
            ethics_score += 0.1

        # Check for harm assessment
        safety_terms = ["safety", "harm", "risk", "crisis"]
        if any(term in content.lower() for term in safety_terms):
            ethics_score += 0.1

        return min(1.0, ethics_score)

    def _assess_evidence_based_practice(self, interventions: list[TherapeuticIntervention]) -> float:
        """Assess evidence-based practice adherence."""
        if not interventions:
            return 0.6  # Moderate score for no specific interventions

        evidence_scores = []
        for intervention in interventions:
            if intervention.evidence_level == "strong":
                evidence_scores.append(0.9)
            elif intervention.evidence_level == "moderate":
                evidence_scores.append(0.7)
            else:
                evidence_scores.append(0.5)

        return sum(evidence_scores) / len(evidence_scores)

    def _assess_crisis_management(self, content: str, turns: list[dict]) -> float:
        """Assess crisis management capabilities."""
        crisis_score = 0.8  # Base score

        # Check for crisis indicators
        crisis_terms = ["suicide", "harm", "crisis", "emergency", "danger"]
        has_crisis_content = any(term in content.lower() for term in crisis_terms)

        if has_crisis_content:
            # Check for appropriate crisis response
            response_terms = ["safety plan", "crisis line", "emergency", "professional help"]
            if any(term in content.lower() for term in response_terms):
                crisis_score = 0.95
            else:
                crisis_score = 0.4  # Poor crisis management

        return crisis_score

    def _assess_cultural_competency(self, content: str, turns: list[dict]) -> float:
        """Assess cultural competency."""
        cultural_score = 0.7  # Base score

        # Check for cultural awareness
        cultural_terms = ["culture", "background", "beliefs", "values", "community"]
        if any(term in content.lower() for term in cultural_terms):
            cultural_score += 0.2

        # Check for inclusive language
        inclusive_terms = ["respect", "understand", "perspective", "experience"]
        if any(term in content.lower() for term in inclusive_terms):
            cultural_score += 0.1

        return min(1.0, cultural_score)

    def _identify_compliance_issues(
        self,
        content: str,
        turns: list[dict],
        disorders: list[DSMDisorder],
        interventions: list[TherapeuticIntervention]
    ) -> list[str]:
        """Identify compliance issues."""
        issues = []

        # Check for diagnostic issues
        if disorders and not ("assessment" in content.lower() or "evaluation" in content.lower()):
            issues.append("Diagnosis mentioned without proper assessment language")

        # Check for intervention appropriateness
        for intervention in interventions:
            if intervention.appropriateness_score < 0.6:
                issues.append(f"Questionable appropriateness of {intervention.approach.value}")

        # Check for crisis management
        crisis_terms = ["suicide", "harm", "kill"]
        if any(term in content.lower() for term in crisis_terms):
            if not any(response in content.lower() for response in ["safety", "crisis", "help"]):
                issues.append("Crisis content without appropriate safety response")

        return issues

    def _generate_recommendations(
        self,
        disorders: list[DSMDisorder],
        interventions: list[TherapeuticIntervention],
        compliance_issues: list[str],
        category_scores: dict[ValidationCategory, float]
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Address compliance issues
        if compliance_issues:
            recommendations.append("Address identified compliance issues immediately")

        # Low category scores
        for category, score in category_scores.items():
            if score < 0.7:
                recommendations.append(f"Improve {category.value.replace('_', ' ')}")

        # Intervention recommendations
        if not interventions:
            recommendations.append("Consider incorporating evidence-based therapeutic techniques")

        # Disorder-specific recommendations
        for disorder in disorders:
            if disorder == DSMDisorder.DEPRESSION:
                recommendations.append("Consider CBT or behavioral activation for depression")
            elif disorder == DSMDisorder.ANXIETY:
                recommendations.append("Consider exposure therapy or relaxation techniques for anxiety")
            elif disorder == DSMDisorder.PTSD:
                recommendations.append("Consider trauma-focused therapy (EMDR, CPT)")

        return recommendations

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation summary statistics."""
        if not self.validation_history:
            return {"message": "No validations performed yet"}

        total_validations = len(self.validation_history)
        avg_accuracy = sum(r.overall_accuracy for r in self.validation_history) / total_validations

        # Category averages
        category_averages = {}
        for category in ValidationCategory:
            scores = [r.category_scores.get(category, 0) for r in self.validation_history]
            category_averages[category.value] = sum(scores) / len(scores)

        # Most common disorders
        all_disorders = [d for r in self.validation_history for d in r.identified_disorders]
        disorder_counts = {}
        for disorder in all_disorders:
            disorder_counts[disorder.value] = disorder_counts.get(disorder.value, 0) + 1

        return {
            "total_validations": total_validations,
            "average_accuracy": avg_accuracy,
            "category_averages": category_averages,
            "disorder_distribution": disorder_counts,
            "last_validation": self.validation_history[-1].timestamp.isoformat()
        }


def main():
    """Example usage of the DSM5AccuracyValidator."""
    validator = DSM5AccuracyValidator()

    # Example conversations
    sample_conversations = [
        {
            "id": "dsm_001",
            "content": "I understand you're experiencing symptoms of depression including persistent sadness, loss of interest, and sleep disturbances. Let's explore some cognitive behavioral techniques to help address these negative thought patterns.",
            "turns": [
                {"speaker": "user", "text": "I've been feeling depressed for weeks."},
                {"speaker": "therapist", "text": "I understand you're experiencing depression symptoms. Let's explore CBT techniques."}
            ]
        },
        {
            "id": "dsm_002",
            "content": "You mentioned having panic attacks and constant worry. These are symptoms of anxiety. I want to help you develop coping strategies.",
            "turns": [
                {"speaker": "user", "text": "I have panic attacks and worry constantly."},
                {"speaker": "therapist", "text": "These are anxiety symptoms. Let's develop coping strategies."}
            ]
        }
    ]

    # Validate conversations
    for conversation in sample_conversations:
        result = validator.validate_conversation(conversation)


        if result.compliance_issues:
            for _issue in result.compliance_issues:
                pass

        if result.recommendations:
            for _rec in result.recommendations[:3]:
                pass

    # Print summary
    validator.get_validation_summary()


if __name__ == "__main__":
    main()
