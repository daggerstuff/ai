#!/usr/bin/env python3
"""
CCD-Platform Integration for Pixelated Empathy
Connects Case Conceptualization Diagram (CCD) concepts with the existing
therapeutic simulation engine and client profile structures.
"""

import random
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from ai.tools.utilities.platform.ccd_schema import CCDConceptualization
from ai.tools.utilities.platform.pixelated_empathy_core import (
    ClientPersonality,
    DifficultClientProfile,
    DifficultyLevel,
    SessionObjective,
)

# Therapeutic quality thresholds
THRESHOLD_LOW_THERAPEUTIC = 0.4
THRESHOLD_HIGH_THERAPEUTIC = 0.8
# Trust level thresholds
THRESHOLD_LOW_TRUST = 0.3
THRESHOLD_MODERATE_TRUST = 0.6
THRESHOLD_HIGH_TRUST = 0.7
THRESHOLD_VERY_HIGH_TRUST = 0.8
# Resistance level thresholds
THRESHOLD_LOW_RESISTANCE = 0.3
THRESHOLD_MODERATE_RESISTANCE = 0.4
THRESHOLD_HIGH_RESISTANCE = 0.7
# Session limits
MAX_TURN_LENGTH = 50


class CCDIntegration:
    """Integration layer between CCD concepts and existing platform structures"""

    @staticmethod
    def ccd_profile_to_difficult_client_profile(
        ccd_template: dict[str, Any],
        client_id: str | None = None,
        name: str | None = None,
        age: int | None = None,
        gender: str | None = None,
    ) -> DifficultClientProfile:
        """
        Convert a CCD profile template to a DifficultClientProfile instance.

        Args:
            ccd_template: CCD template dictionary from CCDProfileTemplates
            client_id: Optional client ID (will generate if not provided)
            name: Optional client name (will generate if not provided)
            age: Optional client age (will generate if not provided)
            gender: Optional client gender (will generate if not provided)

        Returns:
            DifficultClientProfile instance ready for use with therapeutic simulation engine
        """
        # Extract mapping data from CCD template
        mapping = ccd_template["difficult_client_profile_mapping"]

        # Generate or use provided identifiers
        final_client_id = client_id or str(uuid.uuid4())
        final_name = name or CCDIntegration._generate_client_name()
        final_age = age or random.randint(18, 65)
        final_gender = gender or random.choice(["male", "female", "non-binary"])

        # Map CCD personality to ClientPersonality enum
        personality_mapping = {
            "resistant": ClientPersonality.RESISTANT,
            "hostile_aggressive": ClientPersonality.HOSTILE_AGGRESSIVE,
            "borderline_traits": ClientPersonality.BORDERLINE_TRAITS,
        }

        client_personality_key = ccd_template["client_personality"].value
        personality_type = personality_mapping.get(
            client_personality_key,
            ClientPersonality.RESISTANT,  # Default fallback
        )

        # Map learning objectives to SessionObjective enums
        session_objectives = []
        for obj in ccd_template["learning_objectives"]:
            if isinstance(obj, SessionObjective):
                session_objectives.append(obj)
            else:
                # Try to convert string to enum
                with suppress(ValueError):
                    session_objectives.append(SessionObjective(obj))

        # Create DifficultClientProfile instance
        return DifficultClientProfile(
            client_id=final_client_id,
            name=final_name,
            age=final_age,
            gender=final_gender,
            personality_type=personality_type,
            difficulty_level=ccd_template["suggested_difficulty"],
            presenting_problem=mapping["presenting_problem"],
            trauma_history=CCDIntegration._generate_trauma_history(personality_type),
            personality_traits=mapping["personality_traits"],
            defense_mechanisms=mapping["defense_mechanisms"],
            triggers=mapping["triggers"],
            strengths=mapping["strengths"],
            communication_style=mapping["communication_style"],
            resistance_patterns=mapping["resistance_patterns"],
            emotional_dysregulation=mapping["emotional_dysregulation"],
            interpersonal_patterns=mapping["interpersonal_patterns"],
            comorbidities=CCDIntegration._generate_comorbidities(ccd_template["suggested_difficulty"]),
            medication_issues=[],  # Will be populated based on comorbidities
            social_factors=CCDIntegration._generate_social_factors(ccd_template["suggested_difficulty"]),
            legal_issues=CCDIntegration._generate_legal_issues(personality_type),
            learning_objectives=session_objectives,
            common_therapist_mistakes=mapping["common_therapist_mistakes"],
            therapeutic_challenges=mapping["therapeutic_challenges"],
            success_indicators=mapping["success_indicators"],
            ai_instructions=CCDIntegration._create_ai_instructions(
                personality_type, ccd_template["suggested_difficulty"]
            ),
            response_patterns=mapping["response_patterns"],
            escalation_triggers=CCDIntegration._generate_escalation_triggers(personality_type),
            de_escalation_responses=CCDIntegration._generate_de_escalation_responses(personality_type),
        )

    @staticmethod
    def update_ccd_formulation_from_simulation(
        ccd_conceptualization: CCDConceptualization,
        simulation_state: dict[str, Any],
        therapist_analysis: dict[str, Any],
    ) -> CCDConceptualization:
        """
        Update CCD formulation based on simulation progress and therapist performance.

        Args:
            ccd_conceptualization: Current CCD conceptualization
            simulation_state: Current state from therapeutic simulation engine
            therapist_analysis: Analysis of therapist's intervention

        Returns:
            Updated CCD conceptualization
        """
        # Create a copy to avoid modifying original
        updated_conceptualization = CCDConceptualization(
            client_id=ccd_conceptualization.client_id,
            timestamp=datetime.now(UTC),
        )

        # Copy all existing data
        updated_conceptualization.problems = ccd_conceptualization.problems.copy()
        updated_conceptualization.factors = ccd_conceptualization.factors.copy()
        updated_conceptualization.hypotheses = ccd_conceptualization.hypotheses.copy()
        updated_conceptualization.interventions = ccd_conceptualization.interventions.copy()
        updated_conceptualization.clinician_notes = ccd_conceptualization.clinician_notes.copy()
        updated_conceptualization.revision_history = ccd_conceptualization.revision_history.copy()

        # Update formulation based on simulation progress
        if ccd_conceptualization.formulation:
            # Extract relevant information from simulation
            trust_level = simulation_state.get("trust_level", 0.5)
            resistance_level = simulation_state.get("resistance_level", 0.5)
            breakthrough_opportunity = simulation_state.get("breakthrough_opportunity", False)
            therapeutic_quality = therapist_analysis.get("therapeutic_quality", 0.5)

            # Adjust formulation based on progress
            updated_summary = CCDIntegration._adjust_formulation_summary(
                ccd_conceptualization.formulation.summary,
                trust_level,
                resistance_level,
                breakthrough_opportunity,
                therapeutic_quality,
            )

            updated_strengths = CCDIntegration._adjust_strengths(
                ccd_conceptualization.formulation.strengths, trust_level, resistance_level, therapist_analysis
            )

            updated_vulnerabilities = CCDIntegration._adjust_vulnerabilities(
                ccd_conceptualization.formulation.vulnerabilities, trust_level, resistance_level, therapist_analysis
            )

            updated_goals = CCDIntegration._adjust_treatment_goals(
                ccd_conceptualization.formulation.treatment_goals,
                trust_level,
                resistance_level,
                breakthrough_opportunity,
            )

            updated_prognosis = CCDIntegration._adjust_prognosis(
                ccd_conceptualization.formulation.prognosis,
                trust_level,
                resistance_level,
                breakthrough_opportunity,
                therapeutic_quality,
            )

            updated_confidence = CCDIntegration._adjust_confidence(
                ccd_conceptualization.formulation.confidence,
                trust_level,
                resistance_level,
                breakthrough_opportunity,
                therapeutic_quality,
            )

            # Set updated formulation
            updated_conceptualization.set_formulation(
                summary=updated_summary,
                strengths=updated_strengths,
                vulnerabilities=updated_vulnerabilities,
                treatment_goals=updated_goals,
                prognosis=updated_prognosis,
                confidence=updated_confidence,
            )

            # Add clinical note about the update
            note = (
                f"Formulation updated based on simulation: "
                f"trust={trust_level:.2f}, resistance={resistance_level:.2f}, "
                f"quality={therapeutic_quality:.2f}"
            )
            updated_conceptualization.clinician_notes.append(note)
            updated_conceptualization._log_revision("Updated formulation based on simulation progress")

        return updated_conceptualization

    @staticmethod
    def _generate_client_name() -> str:
        """Generate realistic client names"""
        first_names = ["Alex", "Jordan", "Casey", "Morgan", "Riley", "Avery", "Quinn", "Dakota"]
        last_names = ["Smith", "Johnson", "Brown", "Wilson", "Miller", "Davis", "Garcia", "Rodriguez"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"

    @staticmethod
    def _generate_trauma_history(personality_type: ClientPersonality) -> list:
        """Generate trauma history relevant to personality type"""
        trauma_histories = {
            ClientPersonality.BORDERLINE_TRAITS: ["Childhood emotional neglect", "Invalidating family environment"],
            ClientPersonality.NARCISSISTIC_TRAITS: ["Childhood emotional abuse", "Parentification"],
            ClientPersonality.TRAUMA_REACTIVE: ["Combat trauma", "Sexual assault", "Childhood abuse"],
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Domestic violence exposure", "Bullying victimization"],
            ClientPersonality.SUICIDAL_IDEATION: ["Early loss experiences", "Chronic feelings of hopelessness"],
        }

        base_trauma = trauma_histories.get(personality_type, ["Unspecified trauma history"])
        # Return 1-2 trauma history items
        return random.sample(base_trauma, min(len(base_trauma), random.randint(1, 2)))

    @staticmethod
    def _generate_comorbidities(difficulty_level: DifficultyLevel) -> list:
        """Generate comorbidities based on difficulty level"""
        comorbidities_list = [
            "Substance use disorder",
            "Eating disorder",
            "Personality disorder comorbidity",
            "Severe depression",
            "Anxiety disorders",
            "PTSD",
        ]

        # Higher difficulty = more comorbidities
        num_comorbidities = min(difficulty_level.value, len(comorbidities_list))
        return random.sample(comorbidities_list, num_comorbidities)

    @staticmethod
    def _generate_social_factors(difficulty_level: DifficultyLevel) -> list:
        """Generate social factors based on difficulty level"""
        social_factors_list = [
            "Domestic violence",
            "Financial crisis",
            "Legal problems",
            "Family conflict",
            "Work stress",
            "Housing instability",
        ]

        # Higher difficulty = more social stressors
        num_stressors = min(difficulty_level.value, len(social_factors_list))
        return random.sample(social_factors_list, num_stressors)

    @staticmethod
    def _generate_legal_issues(personality_type: ClientPersonality) -> list:
        """Generate legal complications when relevant"""
        legal_issues = {
            ClientPersonality.HOSTILE_AGGRESSIVE: ["Assault charges", "Domestic violence charges"],
            ClientPersonality.SUBSTANCE_DEPENDENT: ["DUI charges", "Drug possession"],
            ClientPersonality.RESISTANT: ["Court-mandated treatment", "Probation requirements"],
        }

        base_issues = legal_issues.get(personality_type, [])
        # Return 0-1 legal issues
        if base_issues and random.random() < THRESHOLD_LOW_TRUST:
            return random.sample(base_issues, min(len(base_issues), 1))
        return []

    @staticmethod
    def _create_ai_instructions(personality_type: ClientPersonality, difficulty_level: DifficultyLevel) -> dict:
        """Create detailed AI behavior instructions"""
        return {
            "personality_adherence": f"Consistently embody {personality_type.value} traits throughout session",
            "difficulty_calibration": (
                f"Maintain level {difficulty_level.value} difficulty - challenging but not impossible"
            ),
            "response_authenticity": "Respond as a real person with this personality would, not as an AI",
            "therapeutic_realism": "Create realistic therapeutic challenges that therapists encounter",
            "escalation_management": "Escalate resistance when therapist makes common mistakes",
            "breakthrough_opportunities": "Provide breakthrough moments when therapist demonstrates skill",
            "emotional_consistency": "Maintain emotional consistency with personality pattern",
            "boundary_testing": "Test therapist boundaries appropriately for personality type",
            "resistance_timing": "Time resistance patterns realistically within session flow",
            "vulnerability_windows": "Allow moments of vulnerability when earned therapeutically",
        }

    @staticmethod
    def _generate_escalation_triggers(personality_type: ClientPersonality) -> list:
        """Generate triggers that escalate client difficulty"""
        triggers = {
            ClientPersonality.HOSTILE_AGGRESSIVE: [
                "Therapist appears intimidated",
                "Boundaries are inconsistent",
                "Client feels judged or criticized",
            ],
            ClientPersonality.BORDERLINE_TRAITS: [
                "Therapist seems distracted or disconnected",
                "Session ending approaches",
                "Client feels misunderstood",
            ],
            ClientPersonality.RESISTANT: [
                "Direct emotional exploration",
                "Challenges to competence",
                "Perceived criticism",
            ],
        }

        base_triggers = triggers.get(personality_type, ["Poor therapeutic rapport"])
        # Return 1-2 triggers
        return random.sample(base_triggers, min(len(base_triggers), random.randint(1, 2)))

    @staticmethod
    def _generate_de_escalation_responses(personality_type: ClientPersonality) -> list:
        """Generate appropriate de-escalation responses"""
        responses = {
            ClientPersonality.HOSTILE_AGGRESSIVE: [
                "I can see you're really frustrated right now",
                "Help me understand what's making you angry",
                "Your feelings are valid, let's work with this together",
            ],
            ClientPersonality.TRAUMA_REACTIVE: [
                "You're safe here with me right now",
                "Let's focus on grounding - feel your feet on the floor",
                "Take your time, there's no pressure",
            ],
            ClientPersonality.BORDERLINE_TRAITS: [
                "I notice you're feeling overwhelmed right now",
                "Let's slow down and check in with what you need",
                "Your safety is my priority",
            ],
        }

        base_responses = responses.get(personality_type, ["I hear you", "That sounds difficult"])
        # Return 1-2 responses
        return random.sample(base_responses, min(len(base_responses), random.randint(1, 2)))

    @staticmethod
    def _adjust_formulation_summary(
        base_summary: str,
        trust_level: float,
        resistance_level: float,
        breakthrough_opportunity: bool,
        therapeutic_quality: float,
    ) -> str:
        """Adjust formulation summary based on simulation progress"""
        adjustments = []

        if trust_level > THRESHOLD_HIGH_TRUST:
            adjustments.append("increasing trust in therapeutic relationship")
        elif trust_level < THRESHOLD_LOW_TRUST:
            adjustments.append("persistent difficulties with trust")

        if resistance_level > THRESHOLD_HIGH_RESISTANCE:
            adjustments.append("significant resistance to therapeutic process")
        elif resistance_level < THRESHOLD_LOW_RESISTANCE:
            adjustments.append("decreasing resistance and increased openness")

        if breakthrough_opportunity:
            adjustments.append("evidence of breakthrough potential")

        if therapeutic_quality > THRESHOLD_HIGH_THERAPEUTIC:
            adjustments.append("high-quality therapeutic interventions observed")
        elif therapeutic_quality < THRESHOLD_LOW_THERAPEUTIC:
            adjustments.append("suboptimal therapeutic interventions noted")

        if adjustments:
            adjustment_text = "; ".join(adjustments)
            return f"{base_summary}. Current simulation shows: {adjustment_text}."
        return base_summary

    @staticmethod
    def _adjust_strengths(
        base_strengths: list, trust_level: float, resistance_level: float, therapist_analysis: dict
    ) -> list:
        """Adjust strengths list based on simulation progress"""
        adjusted = base_strengths.copy()

        # Add strengths based on good therapeutic process
        if (
            trust_level > THRESHOLD_MODERATE_TRUST
            and resistance_level < THRESHOLD_MODERATE_RESISTANCE
            and "Capacity for trust and vulnerability" not in adjusted
        ):
            adjusted.append("Capacity for trust and vulnerability")

        if (
            therapist_analysis.get("therapeutic_quality", 0) > THRESHOLD_HIGH_THERAPEUTIC
            and "Responsiveness to skilled interventions" not in adjusted
        ):
            adjusted.append("Responsiveness to skilled interventions")

        return adjusted

    @staticmethod
    def _adjust_vulnerabilities(
        base_vulnerabilities: list, trust_level: float, resistance_level: float, _therapist_analysis: dict
    ) -> list:
        """Adjust vulnerabilities list based on simulation progress"""
        adjusted = base_vulnerabilities.copy()

        # Reduce or remove vulnerabilities that are improving
        if trust_level > THRESHOLD_HIGH_TRUST:
            # Remove trust-related vulnerabilities if present
            trust_related = [v for v in adjusted if "trust" in v.lower()]
            for v in trust_related:
                if v in adjusted:
                    adjusted.remove(v)

        if resistance_level < THRESHOLD_LOW_RESISTANCE:
            # Remove resistance-related vulnerabilities if present
            resistance_related = [v for v in adjusted if "resist" in v.lower()]
            for v in resistance_related:
                if v in adjusted:
                    adjusted.remove(v)

        return adjusted

    @staticmethod
    def _adjust_treatment_goals(
        base_goals: list, trust_level: float, resistance_level: float, breakthrough_opportunity: bool
    ) -> list:
        """Adjust treatment goals based on simulation progress"""
        adjusted = base_goals.copy()

        # If making good progress, add maintenance and advanced goals
        if (
            trust_level > THRESHOLD_HIGH_TRUST
            and resistance_level < THRESHOLD_LOW_RESISTANCE
            and "Maintain therapeutic gains" not in adjusted
        ):
            adjusted.append("Maintain therapeutic gains")

        if breakthrough_opportunity and "Capitalize on breakthrough momentum" not in adjusted:
            adjusted.append("Capitalize on breakthrough momentum")

        return adjusted

    @staticmethod
    def _adjust_prognosis(
        base_prognosis: str,
        trust_level: float,
        resistance_level: float,
        breakthrough_opportunity: bool,
        therapeutic_quality: float,
    ) -> str:
        """Adjust prognosis based on simulation progress"""
        # Prognosis levels: optimistic, guarded_optimistic, guarded, poor
        prognosis_scores = {"poor": 0, "guarded": 1, "guarded_optimistic": 2, "optimistic": 3}

        current_score = prognosis_scores.get(base_prognosis, 1)

        # Adjust based on factors
        if trust_level > THRESHOLD_VERY_HIGH_TRUST:
            current_score = min(3, current_score + 1)
        elif trust_level < THRESHOLD_LOW_TRUST:
            current_score = max(0, current_score - 1)

        if resistance_level < THRESHOLD_LOW_RESISTANCE:
            current_score = min(3, current_score + 1)
        elif resistance_level > THRESHOLD_HIGH_RESISTANCE:
            current_score = max(0, current_score - 1)

        if breakthrough_opportunity:
            current_score = min(3, current_score + 1)

        if therapeutic_quality > THRESHOLD_HIGH_THERAPEUTIC:
            current_score = min(3, current_score + 1)
        elif therapeutic_quality < THRESHOLD_LOW_THERAPEUTIC:
            current_score = max(0, current_score - 1)

        # Convert back to string
        for prognosis, score in prognosis_scores.items():
            if score == current_score:
                return prognosis

        return base_prognosis  # pragma: no cover

    @staticmethod
    def _adjust_confidence(
        base_confidence: float,
        trust_level: float,
        resistance_level: float,
        breakthrough_opportunity: bool,
        therapeutic_quality: float,
    ) -> float:
        """Adjust confidence in formulation based on simulation progress"""
        adjusted = base_confidence

        # Increase confidence with good therapeutic process
        if trust_level > THRESHOLD_HIGH_TRUST and resistance_level < THRESHOLD_MODERATE_RESISTANCE:
            adjusted += 0.1
        elif trust_level < THRESHOLD_LOW_TRUST and resistance_level > THRESHOLD_HIGH_RESISTANCE:
            adjusted -= 0.1

        # Increase confidence with breakthrough opportunities
        if breakthrough_opportunity:
            adjusted += 0.15

        # Increase confidence with high-quality interventions
        if therapeutic_quality > THRESHOLD_HIGH_THERAPEUTIC:
            adjusted += 0.1
        elif therapeutic_quality < THRESHOLD_LOW_THERAPEUTIC:
            adjusted -= 0.1

        # Clamp to valid range
        return max(0.1, min(0.95, adjusted))


# Convenience functions for easy use
def create_difficult_client_from_ccd_template(
    ccd_template: dict[str, Any],
    client_id: str | None = None,
    name: str | None = None,
    age: int | None = None,
    gender: str | None = None,
) -> DifficultClientProfile:
    """
    Create a DifficultClientProfile from a CCD template.

    This is the main integration point for creating client profiles
    from CCD conceptualizations for use with the therapeutic simulation engine.
    """
    return CCDIntegration.ccd_profile_to_difficult_client_profile(ccd_template, client_id, name, age, gender)


def update_ccd_with_simulation_results(
    ccd_conceptualization: CCDConceptualization, simulation_state: dict[str, Any], therapist_analysis: dict[str, Any]
) -> CCDConceptualization:
    """
    Update a CCD conceptualization with results from a therapeutic simulation.

    This allows the CCD formulation to evolve based on actual simulation
    progress and therapist performance.
    """
    return CCDIntegration.update_ccd_formulation_from_simulation(
        ccd_conceptualization, simulation_state, therapist_analysis
    )
