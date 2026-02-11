"""
Specialized Populations Training Data System

This module provides comprehensive training data generation for specialized therapeutic
populations including trauma survivors, addiction recovery, LGBTQ+ individuals,
veterans, adolescents, elderly, and other specialized groups requiring tailored
therapeutic approaches.

Key Features:
- Trauma-informed care protocols
- Addiction treatment frameworks
- Population-specific therapeutic approaches
- Cultural competency integration
- Specialized assessment tools
- Tailored intervention strategies
- Population-specific conversation generation
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .client_scenario_generator import ClientScenario
from .therapeutic_response_generator import TherapeuticModality


class SpecializedPopulation(Enum):
    """Specialized therapeutic populations."""
    TRAUMA_SURVIVORS = "trauma_survivors"
    ADDICTION_RECOVERY = "addiction_recovery"
    LGBTQ_PLUS = "lgbtq_plus"
    VETERANS = "veterans"
    ADOLESCENTS = "adolescents"
    ELDERLY = "elderly"
    CHRONIC_ILLNESS = "chronic_illness"
    DOMESTIC_VIOLENCE = "domestic_violence"
    EATING_DISORDERS = "eating_disorders"
    GRIEF_BEREAVEMENT = "grief_bereavement"
    FIRST_RESPONDERS = "first_responders"
    HEALTHCARE_WORKERS = "healthcare_workers"


class TraumaType(Enum):
    """Types of trauma experiences."""
    CHILDHOOD_ABUSE = "childhood_abuse"
    SEXUAL_ASSAULT = "sexual_assault"
    COMBAT_TRAUMA = "combat_trauma"
    NATURAL_DISASTER = "natural_disaster"
    MEDICAL_TRAUMA = "medical_trauma"
    COMPLEX_TRAUMA = "complex_trauma"
    VICARIOUS_TRAUMA = "vicarious_trauma"
    HISTORICAL_TRAUMA = "historical_trauma"
    DEVELOPMENTAL_TRAUMA = "developmental_trauma"


class AddictionStage(Enum):
    """Stages of addiction recovery."""
    PRECONTEMPLATION = "precontemplation"
    CONTEMPLATION = "contemplation"
    PREPARATION = "preparation"
    ACTION = "action"
    MAINTENANCE = "maintenance"
    RELAPSE = "relapse"
    TERMINATION = "termination"


class CulturalFactor(Enum):
    """Cultural factors affecting treatment."""
    LANGUAGE_BARRIERS = "language_barriers"
    RELIGIOUS_BELIEFS = "religious_beliefs"
    FAMILY_DYNAMICS = "family_dynamics"
    SOCIOECONOMIC_STATUS = "socioeconomic_status"
    IMMIGRATION_STATUS = "immigration_status"
    RACIAL_IDENTITY = "racial_identity"
    GENDER_IDENTITY = "gender_identity"
    SEXUAL_ORIENTATION = "sexual_orientation"


@dataclass
class PopulationCharacteristics:
    """Characteristics of specialized population."""
    id: str
    population: SpecializedPopulation
    name: str
    description: str
    common_presentations: list[str]
    unique_challenges: list[str]
    cultural_considerations: list[CulturalFactor]
    trauma_considerations: list[TraumaType]
    preferred_modalities: list[TherapeuticModality]
    contraindicated_approaches: list[str]
    assessment_modifications: list[str]
    therapeutic_goals: list[str]
    treatment_considerations: list[str]


@dataclass
class SpecializedProtocol:
    """Specialized treatment protocol for population."""
    id: str
    population: SpecializedPopulation
    protocol_name: str
    evidence_base: str
    target_symptoms: list[str]
    intervention_phases: list[str]
    specialized_techniques: list[str]
    cultural_adaptations: list[str]
    safety_considerations: list[str]
    outcome_measures: list[str]
    session_structure: list[str]
    therapist_requirements: list[str]


@dataclass
class PopulationConversation:
    """Specialized population conversation with tailored approaches."""
    id: str
    population_characteristics: PopulationCharacteristics
    specialized_protocol: SpecializedProtocol
    client_scenario: ClientScenario
    conversation_exchanges: list[dict[str, Any]]
    cultural_adaptations_used: list[str]
    trauma_informed_elements: list[str]
    population_specific_techniques: list[str]
    therapeutic_alliance_factors: list[str]
    cultural_competency_demonstrated: list[str]
    specialized_assessments: list[str]
    treatment_modifications: list[str]
    outcome_considerations: list[str]


class SpecializedPopulationsGenerator:
    """
    Comprehensive specialized populations training data generation system.

    This class provides generation of population-specific therapeutic conversations,
    trauma-informed care protocols, and culturally competent treatment approaches.
    """

    def __init__(self):
        """Initialize the specialized populations generator."""
        self.population_characteristics = self._initialize_population_characteristics()
        self.specialized_protocols = self._initialize_specialized_protocols()
        self.cultural_frameworks = self._initialize_cultural_frameworks()
        self.trauma_informed_principles = self._initialize_trauma_informed_principles()

    def _initialize_population_characteristics(self) -> dict[SpecializedPopulation, PopulationCharacteristics]:
        """Initialize characteristics for specialized populations."""
        characteristics = {}

        # Trauma Survivors
        characteristics[SpecializedPopulation.TRAUMA_SURVIVORS] = PopulationCharacteristics(
            id="trauma_survivors",
            population=SpecializedPopulation.TRAUMA_SURVIVORS,
            name="Trauma Survivors",
            description="Individuals who have experienced traumatic events requiring trauma-informed care",
            common_presentations=[
                "PTSD symptoms",
                "Hypervigilance",
                "Dissociation",
                "Emotional dysregulation",
                "Trust issues",
                "Sleep disturbances",
                "Intrusive memories"
            ],
            unique_challenges=[
                "Trauma triggers in therapy",
                "Difficulty with trust and safety",
                "Potential for re-traumatization",
                "Complex symptom presentations",
                "Shame and self-blame",
                "Avoidance behaviors"
            ],
            cultural_considerations=[
                CulturalFactor.FAMILY_DYNAMICS,
                CulturalFactor.RELIGIOUS_BELIEFS,
                CulturalFactor.RACIAL_IDENTITY
            ],
            trauma_considerations=[
                TraumaType.CHILDHOOD_ABUSE,
                TraumaType.SEXUAL_ASSAULT,
                TraumaType.COMPLEX_TRAUMA,
                TraumaType.DEVELOPMENTAL_TRAUMA
            ],
            preferred_modalities=[
                TherapeuticModality.TRAUMA_INFORMED,
                TherapeuticModality.COGNITIVE_BEHAVIORAL,
                TherapeuticModality.MINDFULNESS_BASED
            ],
            contraindicated_approaches=[
                "Confrontational techniques",
                "Forced disclosure",
                "Intensive exposure without preparation"
            ],
            assessment_modifications=[
                "Trauma screening tools",
                "Safety assessment priority",
                "Gradual disclosure approach",
                "Trigger identification"
            ],
            therapeutic_goals=[
                "Safety and stabilization",
                "Trauma processing",
                "Integration and recovery",
                "Post-traumatic growth"
            ],
            treatment_considerations=[
                "Trauma-informed environment",
                "Paced treatment approach",
                "Safety planning",
                "Trigger management"
            ]
        )

        # Addiction Recovery
        characteristics[SpecializedPopulation.ADDICTION_RECOVERY] = PopulationCharacteristics(
            id="addiction_recovery",
            population=SpecializedPopulation.ADDICTION_RECOVERY,
            name="Addiction Recovery",
            description="Individuals in various stages of addiction recovery requiring specialized treatment approaches",
            common_presentations=[
                "Substance use disorders",
                "Cravings and urges",
                "Withdrawal symptoms",
                "Relapse concerns",
                "Shame and guilt",
                "Relationship problems",
                "Legal/financial issues"
            ],
            unique_challenges=[
                "Denial and minimization",
                "Ambivalence about change",
                "High relapse risk",
                "Comorbid mental health issues",
                "Social stigma",
                "Environmental triggers"
            ],
            cultural_considerations=[
                CulturalFactor.FAMILY_DYNAMICS,
                CulturalFactor.SOCIOECONOMIC_STATUS,
                CulturalFactor.RELIGIOUS_BELIEFS
            ],
            trauma_considerations=[
                TraumaType.CHILDHOOD_ABUSE,
                TraumaType.COMPLEX_TRAUMA
            ],
            preferred_modalities=[
                TherapeuticModality.COGNITIVE_BEHAVIORAL,
                TherapeuticModality.ACCEPTANCE_COMMITMENT,
                TherapeuticModality.SOLUTION_FOCUSED
            ],
            contraindicated_approaches=[
                "Confrontational interventions",
                "Shame-based approaches",
                "Rigid abstinence-only models without flexibility"
            ],
            assessment_modifications=[
                "Substance use history",
                "Readiness for change assessment",
                "Trigger identification",
                "Support system evaluation"
            ],
            therapeutic_goals=[
                "Motivation enhancement",
                "Relapse prevention",
                "Coping skills development",
                "Recovery maintenance"
            ],
            treatment_considerations=[
                "Stage-matched interventions",
                "Harm reduction principles",
                "Relapse as learning opportunity",
                "Holistic recovery approach"
            ]
        )

        # LGBTQ+ Population
        characteristics[SpecializedPopulation.LGBTQ_PLUS] = PopulationCharacteristics(
            id="lgbtq_plus",
            population=SpecializedPopulation.LGBTQ_PLUS,
            name="LGBTQ+ Individuals",
            description="Lesbian, gay, bisexual, transgender, queer/questioning, and other sexual/gender minorities",
            common_presentations=[
                "Identity development issues",
                "Coming out concerns",
                "Discrimination experiences",
                "Family rejection",
                "Internalized stigma",
                "Relationship challenges",
                "Gender dysphoria"
            ],
            unique_challenges=[
                "Minority stress",
                "Lack of affirming providers",
                "Legal discrimination",
                "Healthcare barriers",
                "Religious conflicts",
                "Intersectional identities"
            ],
            cultural_considerations=[
                CulturalFactor.GENDER_IDENTITY,
                CulturalFactor.SEXUAL_ORIENTATION,
                CulturalFactor.FAMILY_DYNAMICS,
                CulturalFactor.RELIGIOUS_BELIEFS
            ],
            trauma_considerations=[
                TraumaType.COMPLEX_TRAUMA,
                TraumaType.HISTORICAL_TRAUMA
            ],
            preferred_modalities=[
                TherapeuticModality.HUMANISTIC,
                TherapeuticModality.COGNITIVE_BEHAVIORAL,
                TherapeuticModality.ACCEPTANCE_COMMITMENT
            ],
            contraindicated_approaches=[
                "Conversion therapy",
                "Pathologizing approaches",
                "Assumptions about identity"
            ],
            assessment_modifications=[
                "Affirming language use",
                "Identity exploration focus",
                "Discrimination assessment",
                "Support system evaluation"
            ],
            therapeutic_goals=[
                "Identity affirmation",
                "Minority stress reduction",
                "Relationship enhancement",
                "Community connection"
            ],
            treatment_considerations=[
                "Affirming therapeutic environment",
                "Cultural competency required",
                "Intersectional awareness",
                "Community resource connection"
            ]
        )

        return characteristics

    def _initialize_specialized_protocols(self) -> dict[SpecializedPopulation, list[SpecializedProtocol]]:
        """Initialize specialized treatment protocols for populations."""
        protocols = {}

        # Trauma-Informed Care Protocols
        protocols[SpecializedPopulation.TRAUMA_SURVIVORS] = [
            SpecializedProtocol(
                id="trauma_focused_cbt",
                population=SpecializedPopulation.TRAUMA_SURVIVORS,
                protocol_name="Trauma-Focused Cognitive Behavioral Therapy",
                evidence_base="Strong empirical support for PTSD treatment",
                target_symptoms=[
                    "Intrusive memories",
                    "Avoidance behaviors",
                    "Negative cognitions",
                    "Hyperarousal symptoms"
                ],
                intervention_phases=[
                    "Stabilization and safety",
                    "Trauma processing",
                    "Integration and reconnection"
                ],
                specialized_techniques=[
                    "Cognitive restructuring for trauma",
                    "Imaginal exposure",
                    "In-vivo exposure",
                    "Relapse prevention"
                ],
                cultural_adaptations=[
                    "Culturally relevant metaphors",
                    "Family involvement when appropriate",
                    "Spiritual/religious integration",
                    "Historical trauma acknowledgment"
                ],
                safety_considerations=[
                    "Trauma trigger management",
                    "Dissociation monitoring",
                    "Safety planning",
                    "Grounding techniques"
                ],
                outcome_measures=[
                    "PTSD symptom reduction",
                    "Functional improvement",
                    "Quality of life enhancement",
                    "Trauma-related cognition changes"
                ],
                session_structure=[
                    "Safety check-in",
                    "Homework review",
                    "Skill building/processing",
                    "Grounding and closure"
                ],
                therapist_requirements=[
                    "Trauma-informed training",
                    "Cultural competency",
                    "Secondary trauma awareness",
                    "Safety protocol knowledge"
                ]
            )
        ]

        # Addiction Treatment Protocols
        protocols[SpecializedPopulation.ADDICTION_RECOVERY] = [
            SpecializedProtocol(
                id="motivational_interviewing",
                population=SpecializedPopulation.ADDICTION_RECOVERY,
                protocol_name="Motivational Interviewing for Addiction",
                evidence_base="Extensive research support for addiction treatment",
                target_symptoms=[
                    "Ambivalence about change",
                    "Low motivation",
                    "Denial and minimization",
                    "Resistance to treatment"
                ],
                intervention_phases=[
                    "Engagement and rapport",
                    "Focusing on change",
                    "Evoking motivation",
                    "Planning for change"
                ],
                specialized_techniques=[
                    "Open-ended questions",
                    "Affirmations",
                    "Reflective listening",
                    "Summarizing",
                    "Change talk elicitation"
                ],
                cultural_adaptations=[
                    "Cultural values integration",
                    "Family system consideration",
                    "Spiritual/religious resources",
                    "Community-based approaches"
                ],
                safety_considerations=[
                    "Withdrawal monitoring",
                    "Suicide risk assessment",
                    "Medical complications",
                    "Relapse prevention"
                ],
                outcome_measures=[
                    "Motivation for change",
                    "Treatment engagement",
                    "Substance use reduction",
                    "Functional improvement"
                ],
                session_structure=[
                    "Check-in and assessment",
                    "Motivation exploration",
                    "Change planning",
                    "Commitment strengthening"
                ],
                therapist_requirements=[
                    "MI training and certification",
                    "Addiction knowledge",
                    "Non-judgmental approach",
                    "Cultural sensitivity"
                ]
            )
        ]

        # LGBTQ+ Affirmative Protocols
        protocols[SpecializedPopulation.LGBTQ_PLUS] = [
            SpecializedProtocol(
                id="lgbtq_affirmative_therapy",
                population=SpecializedPopulation.LGBTQ_PLUS,
                protocol_name="LGBTQ+ Affirmative Therapy",
                evidence_base="Research-supported approach for sexual and gender minorities",
                target_symptoms=[
                    "Minority stress",
                    "Internalized stigma",
                    "Identity confusion",
                    "Discrimination trauma"
                ],
                intervention_phases=[
                    "Identity exploration",
                    "Minority stress processing",
                    "Affirmation and integration",
                    "Community connection"
                ],
                specialized_techniques=[
                    "Identity affirmation",
                    "Minority stress reduction",
                    "Coming out support",
                    "Intersectional awareness"
                ],
                cultural_adaptations=[
                    "Intersectional identity consideration",
                    "Cultural background integration",
                    "Religious/spiritual reconciliation",
                    "Family dynamics navigation"
                ],
                safety_considerations=[
                    "Disclosure safety assessment",
                    "Discrimination risk evaluation",
                    "Support system availability",
                    "Crisis intervention planning"
                ],
                outcome_measures=[
                    "Identity acceptance",
                    "Minority stress reduction",
                    "Relationship satisfaction",
                    "Community connectedness"
                ],
                session_structure=[
                    "Affirming check-in",
                    "Identity exploration",
                    "Skill building",
                    "Resource connection"
                ],
                therapist_requirements=[
                    "LGBTQ+ competency training",
                    "Bias awareness",
                    "Affirming language use",
                    "Community resource knowledge"
                ]
            )
        ]

        return protocols

    def _initialize_cultural_frameworks(self) -> dict[CulturalFactor, dict[str, Any]]:
        """Initialize cultural competency frameworks."""
        return {
            CulturalFactor.LANGUAGE_BARRIERS: {
                "considerations": [
                    "Use of interpreters",
                    "Culturally adapted materials",
                    "Non-verbal communication awareness",
                    "Concept translation challenges"
                ],
                "interventions": [
                    "Professional interpreter services",
                    "Bilingual therapy when available",
                    "Visual aids and materials",
                    "Cultural broker involvement"
                ],
                "assessment_modifications": [
                    "Language-appropriate instruments",
                    "Cultural concept clarification",
                    "Extended assessment time",
                    "Family involvement in translation"
                ]
            },
            CulturalFactor.RELIGIOUS_BELIEFS: {
                "considerations": [
                    "Spiritual coping mechanisms",
                    "Religious community support",
                    "Potential conflicts with treatment",
                    "Sacred practices integration"
                ],
                "interventions": [
                    "Spiritual assessment",
                    "Religious resource integration",
                    "Clergy collaboration",
                    "Values-based treatment planning"
                ],
                "assessment_modifications": [
                    "Spiritual history taking",
                    "Religious coping assessment",
                    "Values clarification",
                    "Community resource evaluation"
                ]
            },
            CulturalFactor.FAMILY_DYNAMICS: {
                "considerations": [
                    "Collectivist vs individualist values",
                    "Family hierarchy and roles",
                    "Decision-making processes",
                    "Intergenerational conflicts"
                ],
                "interventions": [
                    "Family therapy inclusion",
                    "Cultural mediator role",
                    "Respect for family values",
                    "Gradual change approaches"
                ],
                "assessment_modifications": [
                    "Family system assessment",
                    "Cultural value exploration",
                    "Role and responsibility mapping",
                    "Support system evaluation"
                ]
            }
        }

    def _initialize_trauma_informed_principles(self) -> dict[str, Any]:
        """Initialize trauma-informed care principles."""
        return {
            "safety": {
                "physical_safety": [
                    "Safe physical environment",
                    "Clear boundaries",
                    "Predictable routines",
                    "Emergency procedures"
                ],
                "psychological_safety": [
                    "Emotional safety assurance",
                    "Trust building",
                    "Confidentiality protection",
                    "Non-judgmental approach"
                ]
            },
            "trustworthiness": {
                "transparency": [
                    "Clear communication",
                    "Honest information sharing",
                    "Process explanation",
                    "Expectation setting"
                ],
                "reliability": [
                    "Consistent follow-through",
                    "Dependable presence",
                    "Promise keeping",
                    "Predictable responses"
                ]
            },
            "choice": {
                "empowerment": [
                    "Client autonomy respect",
                    "Decision-making involvement",
                    "Option provision",
                    "Control restoration"
                ],
                "collaboration": [
                    "Shared decision-making",
                    "Goal setting partnership",
                    "Treatment planning involvement",
                    "Feedback incorporation"
                ]
            },
            "collaboration": {
                "partnership": [
                    "Equal relationship",
                    "Mutual respect",
                    "Shared expertise",
                    "Joint problem-solving"
                ]
            },
            "cultural_humility": {
                "responsiveness": [
                    "Cultural awareness",
                    "Bias recognition",
                    "Adaptation willingness",
                    "Learning orientation"
                ]
            }
        }

    def generate_specialized_conversation(
        self,
        client_scenario: ClientScenario,
        population: SpecializedPopulation,
        num_exchanges: int = 10
    ) -> PopulationConversation:
        """
        Generate specialized population conversation with tailored approaches.

        Args:
            client_scenario: Client scenario with population-specific needs
            population: Specialized population type
            num_exchanges: Number of conversation exchanges

        Returns:
            PopulationConversation with population-specific therapeutic dialogue
        """
        # Get population characteristics
        characteristics = self.population_characteristics.get(population)
        if not characteristics:
            characteristics = self._create_generic_characteristics(population)

        # Get specialized protocol
        protocols = self.specialized_protocols.get(population, [])
        protocol = protocols[0] if protocols else self._create_generic_protocol(population)

        # Generate conversation exchanges
        exchanges = self._generate_specialized_exchanges(
            client_scenario, characteristics, protocol, num_exchanges
        )

        # Identify cultural adaptations used
        cultural_adaptations = self._identify_cultural_adaptations(exchanges, characteristics)

        # Identify trauma-informed elements
        trauma_informed_elements = self._identify_trauma_informed_elements(exchanges)

        # Identify population-specific techniques
        population_techniques = self._identify_population_techniques(exchanges, protocol)

        # Assess therapeutic alliance factors
        alliance_factors = self._assess_therapeutic_alliance_factors(exchanges, characteristics)

        # Evaluate cultural competency
        cultural_competency = self._evaluate_cultural_competency(exchanges, characteristics)

        # Identify specialized assessments
        specialized_assessments = self._identify_specialized_assessments(exchanges, characteristics)

        # Determine treatment modifications
        treatment_modifications = self._determine_treatment_modifications(exchanges, characteristics)

        # Consider outcome factors
        outcome_considerations = self._consider_outcome_factors(protocol, characteristics)

        conversation_id = f"specialized_{population.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return PopulationConversation(
            id=conversation_id,
            population_characteristics=characteristics,
            specialized_protocol=protocol,
            client_scenario=client_scenario,
            conversation_exchanges=exchanges,
            cultural_adaptations_used=cultural_adaptations,
            trauma_informed_elements=trauma_informed_elements,
            population_specific_techniques=population_techniques,
            therapeutic_alliance_factors=alliance_factors,
            cultural_competency_demonstrated=cultural_competency,
            specialized_assessments=specialized_assessments,
            treatment_modifications=treatment_modifications,
            outcome_considerations=outcome_considerations
        )

    def _create_generic_characteristics(self, population: SpecializedPopulation) -> PopulationCharacteristics:
        """Create generic characteristics for unspecified populations."""
        return PopulationCharacteristics(
            id=f"generic_{population.value}",
            population=population,
            name=f"Generic {population.value.replace('_', ' ').title()}",
            description=f"General characteristics for {population.value} population",
            common_presentations=["Varied presentations"],
            unique_challenges=["Population-specific challenges"],
            cultural_considerations=[CulturalFactor.FAMILY_DYNAMICS],
            trauma_considerations=[TraumaType.COMPLEX_TRAUMA],
            preferred_modalities=[TherapeuticModality.COGNITIVE_BEHAVIORAL],
            contraindicated_approaches=["Generic contraindications"],
            assessment_modifications=["Standard modifications"],
            therapeutic_goals=["Population-appropriate goals"],
            treatment_considerations=["Specialized considerations"]
        )

    def _create_generic_protocol(self, population: SpecializedPopulation) -> SpecializedProtocol:
        """Create generic protocol for unspecified populations."""
        return SpecializedProtocol(
            id=f"generic_{population.value}_protocol",
            population=population,
            protocol_name=f"General {population.value.replace('_', ' ').title()} Protocol",
            evidence_base="General evidence-based practices",
            target_symptoms=["Population-specific symptoms"],
            intervention_phases=["Assessment", "Intervention", "Maintenance"],
            specialized_techniques=["Population-appropriate techniques"],
            cultural_adaptations=["Cultural considerations"],
            safety_considerations=["Safety protocols"],
            outcome_measures=["Relevant outcomes"],
            session_structure=["Standard structure"],
            therapist_requirements=["Specialized training"]
        )

    def _generate_specialized_exchanges(
        self,
        client_scenario: ClientScenario,
        characteristics: PopulationCharacteristics,
        protocol: SpecializedProtocol,
        num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate conversation exchanges with population-specific considerations."""
        exchanges = []

        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_specialized_response(
                    client_scenario, characteristics, protocol, i // 2
                )
            else:  # Client turn
                exchange = self._generate_client_specialized_response(
                    client_scenario, characteristics, i // 2
                )

            exchanges.append(exchange)

        return exchanges

    def _generate_therapist_specialized_response(
        self,
        client_scenario: ClientScenario,
        characteristics: PopulationCharacteristics,
        protocol: SpecializedProtocol,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate therapist response with population-specific considerations."""

        # Select appropriate technique based on population and protocol
        if exchange_index == 0:
            content = self._generate_population_opening(characteristics)
            technique = "rapport_building"
        elif exchange_index == 1:
            content = self._generate_population_assessment(characteristics)
            technique = "specialized_assessment"
        elif exchange_index == 2:
            content = self._generate_cultural_exploration(characteristics)
            technique = "cultural_exploration"
        else:
            content = self._generate_population_intervention(characteristics, protocol)
            technique = random.choice(protocol.specialized_techniques)

        return {
            "speaker": "therapist",
            "content": content,
            "specialized_technique": technique,
            "population_focus": characteristics.population.value,
            "cultural_adaptation": self._determine_cultural_adaptation(characteristics, exchange_index),
            "trauma_informed_element": self._determine_trauma_informed_element(exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_client_specialized_response(
        self,
        client_scenario: ClientScenario,
        characteristics: PopulationCharacteristics,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate client response with population-specific considerations."""

        # Generate response based on population characteristics
        content = self._generate_population_client_content(
            client_scenario, characteristics, exchange_index
        )

        return {
            "speaker": "client",
            "content": content,
            "emotional_state": self._determine_population_emotional_state(characteristics, exchange_index),
            "cultural_expression": self._determine_cultural_expression(characteristics),
            "population_specific_concerns": self._identify_population_concerns(characteristics, content),
            "engagement_level": self._assess_population_engagement(characteristics, exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_population_opening(self, characteristics: PopulationCharacteristics) -> str:
        """Generate population-appropriate opening statement."""

        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            return "I want you to know that this is a safe space, and we'll go at your pace. You're in control of what you share and when."
        if characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            return "I appreciate you being here today. Change can be challenging, and I'm here to support you wherever you are in your journey."
        if characteristics.population == SpecializedPopulation.LGBTQ_PLUS:
            return "I want to create an affirming space where you can be your authentic self. Please let me know how I can best support you."
        if characteristics.population == SpecializedPopulation.VETERANS:
            return "Thank you for your service. I understand that transitioning to civilian life can present unique challenges, and I'm here to help."
        if characteristics.population == SpecializedPopulation.ADOLESCENTS:
            return "I know it might feel weird talking to an adult about personal stuff. This is your space, and I'm here to listen without judgment."
        return "I'm glad you're here today. I want to understand your unique experiences and how I can best support you."

    def _generate_population_assessment(self, characteristics: PopulationCharacteristics) -> str:
        """Generate population-specific assessment question."""

        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            return "Before we begin, I'd like to understand what helps you feel safe and what might be triggering for you."
        if characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            return "Can you help me understand where you are in your recovery journey and what brought you to seek support?"
        if characteristics.population == SpecializedPopulation.LGBTQ_PLUS:
            return "I'd like to understand your identity and experiences. Please share what feels comfortable for you."
        if characteristics.population == SpecializedPopulation.VETERANS:
            return "Can you tell me about your military experience and how the transition to civilian life has been for you?"
        if characteristics.population == SpecializedPopulation.ADOLESCENTS:
            return "What's been going on in your life lately? Sometimes it helps to start with what's most on your mind."
        return "Can you help me understand your unique experiences and what's brought you here today?"

    def _generate_cultural_exploration(self, characteristics: PopulationCharacteristics) -> str:
        """Generate cultural exploration question."""

        cultural_factors = characteristics.cultural_considerations

        if CulturalFactor.FAMILY_DYNAMICS in cultural_factors:
            return "Family can play an important role in healing. Can you tell me about your family and their understanding of what you're going through?"
        if CulturalFactor.RELIGIOUS_BELIEFS in cultural_factors:
            return "Many people find strength in their spiritual or religious beliefs. How do your beliefs influence your healing process?"
        if CulturalFactor.RACIAL_IDENTITY in cultural_factors:
            return "Your cultural background and identity are important parts of who you are. How do they influence your experiences?"
        return "Your cultural background and values are important to understand. Can you share what's meaningful to you?"

    def _generate_population_intervention(
        self,
        characteristics: PopulationCharacteristics,
        protocol: SpecializedProtocol
    ) -> str:
        """Generate population-specific intervention content."""

        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            return "Let's work on some grounding techniques that can help when you feel overwhelmed. Would you like to try the 5-4-3-2-1 technique?"
        if characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            return "I hear that you're feeling ambivalent about change. That's completely normal. What are some reasons you might want things to stay the same?"
        if characteristics.population == SpecializedPopulation.LGBTQ_PLUS:
            return "It sounds like you're dealing with some difficult family reactions. Your identity is valid, and their reaction doesn't change who you are."
        if characteristics.population == SpecializedPopulation.VETERANS:
            return "The skills that served you well in the military can be adapted for civilian challenges. Let's explore how your strengths can help here."
        if characteristics.population == SpecializedPopulation.ADOLESCENTS:
            return "It sounds like there's a lot of pressure from different directions. Let's break this down and figure out what you can control."
        return "Based on what you've shared, let's work on some strategies that can help with your specific situation."

    def _generate_population_client_content(
        self,
        client_scenario: ClientScenario,
        characteristics: PopulationCharacteristics,
        exchange_index: int
    ) -> str:
        """Generate population-specific client response content."""

        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            responses = [
                "I don't know if I can talk about what happened. It still feels too raw.",
                "Sometimes I feel like I'm back there, like it's happening all over again.",
                "I don't trust easily anymore. Everyone seems like a potential threat.",
                "I just want to feel normal again, but I don't remember what normal feels like."
            ]
        elif characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            responses = [
                "I know I need to change, but I'm scared. This is all I've known for so long.",
                "Everyone keeps telling me I have a problem, but I'm not sure I'm ready to admit it.",
                "I've tried to quit before, but I always end up back where I started.",
                "I want to get better, but I don't know if I'm strong enough."
            ]
        elif characteristics.population == SpecializedPopulation.LGBTQ_PLUS:
            responses = [
                "My family says they love me, but they want me to change who I am.",
                "I'm tired of hiding, but I'm scared of what will happen if I'm open about who I am.",
                "Sometimes I wonder if it would be easier to just pretend to be someone I'm not.",
                "I just want to be accepted for who I am, not who others want me to be."
            ]
        elif characteristics.population == SpecializedPopulation.VETERANS:
            responses = [
                "Civilian life is harder than I expected. Nothing makes sense anymore.",
                "I feel like I don't fit in anywhere. Civilians don't understand what I've been through.",
                "I miss the structure and purpose I had in the military. Now I feel lost.",
                "The nightmares and flashbacks make it hard to function day to day."
            ]
        elif characteristics.population == SpecializedPopulation.ADOLESCENTS:
            responses = [
                "My parents don't understand me at all. They think they know what's best, but they don't get it.",
                "School is so stressful, and everyone expects me to have my whole life figured out.",
                "I feel like I'm different from everyone else, and I don't know if that's okay.",
                "Sometimes I feel like I'm drowning, and no one notices or cares."
            ]
        else:
            responses = [
                "I'm struggling with some things that feel really overwhelming right now.",
                "I don't know if anyone can really understand what I'm going through.",
                "I want things to get better, but I don't know where to start.",
                "Sometimes I feel like I'm the only one dealing with these issues."
            ]

        return responses[exchange_index % len(responses)]

    def _determine_cultural_adaptation(
        self,
        characteristics: PopulationCharacteristics,
        exchange_index: int
    ) -> str:
        """Determine cultural adaptation being used."""

        cultural_factors = characteristics.cultural_considerations

        if CulturalFactor.FAMILY_DYNAMICS in cultural_factors:
            return "family_inclusive_approach"
        if CulturalFactor.RELIGIOUS_BELIEFS in cultural_factors:
            return "spiritual_integration"
        if CulturalFactor.LANGUAGE_BARRIERS in cultural_factors:
            return "culturally_adapted_communication"
        return "culturally_responsive_approach"

    def _determine_trauma_informed_element(self, exchange_index: int) -> str:
        """Determine trauma-informed element being demonstrated."""

        elements = ["safety", "trustworthiness", "choice", "collaboration", "cultural_humility"]
        return elements[exchange_index % len(elements)]

    def _determine_population_emotional_state(
        self,
        characteristics: PopulationCharacteristics,
        exchange_index: int
    ) -> str:
        """Determine emotional state based on population characteristics."""

        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            states = ["hypervigilant", "dissociated", "triggered", "guarded", "overwhelmed"]
        elif characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            states = ["ambivalent", "defensive", "hopeful", "ashamed", "determined"]
        elif characteristics.population == SpecializedPopulation.LGBTQ_PLUS:
            states = ["anxious", "defiant", "vulnerable", "proud", "conflicted"]
        elif characteristics.population == SpecializedPopulation.VETERANS:
            states = ["hypervigilant", "isolated", "frustrated", "stoic", "angry"]
        elif characteristics.population == SpecializedPopulation.ADOLESCENTS:
            states = ["rebellious", "confused", "emotional", "defensive", "seeking"]
        else:
            states = ["uncertain", "hopeful", "guarded", "engaged", "processing"]

        # Emotional state may improve through intervention
        if exchange_index > 2:
            states.extend(["more_open", "trusting", "hopeful"])

        return random.choice(states)

    def _determine_cultural_expression(self, characteristics: PopulationCharacteristics) -> str:
        """Determine cultural expression style."""

        cultural_factors = characteristics.cultural_considerations

        if CulturalFactor.FAMILY_DYNAMICS in cultural_factors:
            return "family_oriented"
        if CulturalFactor.RELIGIOUS_BELIEFS in cultural_factors:
            return "spiritually_informed"
        if CulturalFactor.RACIAL_IDENTITY in cultural_factors:
            return "culturally_grounded"
        return "individually_focused"

    def _identify_population_concerns(
        self,
        characteristics: PopulationCharacteristics,
        content: str
    ) -> list[str]:
        """Identify population-specific concerns from client content."""
        concerns = []
        content_lower = content.lower()

        # Check for population-specific presentations
        for presentation in characteristics.common_presentations:
            if any(word in content_lower for word in presentation.lower().split()):
                concerns.append(presentation.lower().replace(" ", "_"))

        return concerns

    def _assess_population_engagement(
        self,
        characteristics: PopulationCharacteristics,
        exchange_index: int
    ) -> str:
        """Assess client engagement level for population."""

        # Engagement may vary by population and improve over time
        if characteristics.population == SpecializedPopulation.TRAUMA_SURVIVORS:
            if exchange_index == 0:
                return "guarded"
            if exchange_index <= 2:
                return "cautious"
            return "gradually_opening"
        if characteristics.population == SpecializedPopulation.ADDICTION_RECOVERY:
            if exchange_index == 0:
                return "ambivalent"
            if exchange_index <= 2:
                return "defensive"
            return "considering"
        if exchange_index == 0:
            return "uncertain"
        if exchange_index <= 2:
            return "exploring"
        return "engaged"

    def _identify_cultural_adaptations(
        self,
        exchanges: list[dict[str, Any]],
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Identify cultural adaptations used in conversation."""
        adaptations = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "cultural_adaptation" in exchange:
                adaptations.add(exchange["cultural_adaptation"])

        return list(adaptations)

    def _identify_trauma_informed_elements(self, exchanges: list[dict[str, Any]]) -> list[str]:
        """Identify trauma-informed elements demonstrated."""
        elements = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "trauma_informed_element" in exchange:
                elements.add(exchange["trauma_informed_element"])

        return list(elements)

    def _identify_population_techniques(
        self,
        exchanges: list[dict[str, Any]],
        protocol: SpecializedProtocol
    ) -> list[str]:
        """Identify population-specific techniques used."""
        techniques = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "specialized_technique" in exchange:
                technique = exchange["specialized_technique"]
                if technique in protocol.specialized_techniques:
                    techniques.add(technique)

        return list(techniques)

    def _assess_therapeutic_alliance_factors(
        self,
        exchanges: list[dict[str, Any]],
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Assess therapeutic alliance factors demonstrated."""
        factors = []

        # Analyze therapist responses for alliance-building
        therapist_exchanges = [e for e in exchanges if e.get("speaker") == "therapist"]

        if any("safe" in e.get("content", "").lower() for e in therapist_exchanges):
            factors.append("safety_establishment")

        if any("your pace" in e.get("content", "").lower() for e in therapist_exchanges):
            factors.append("client_control")

        if any("understand" in e.get("content", "").lower() for e in therapist_exchanges):
            factors.append("empathic_understanding")

        if any("affirm" in e.get("content", "").lower() for e in therapist_exchanges):
            factors.append("validation_and_affirmation")

        return factors

    def _evaluate_cultural_competency(
        self,
        exchanges: list[dict[str, Any]],
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Evaluate cultural competency demonstrated."""
        competency = []

        # Check for cultural awareness
        therapist_content = " ".join([
            e.get("content", "") for e in exchanges
            if e.get("speaker") == "therapist"
        ]).lower()

        if "cultural" in therapist_content or "background" in therapist_content:
            competency.append("cultural_awareness")

        if "family" in therapist_content:
            competency.append("family_system_awareness")

        if "spiritual" in therapist_content or "religious" in therapist_content:
            competency.append("spiritual_sensitivity")

        if "identity" in therapist_content:
            competency.append("identity_affirmation")

        return competency

    def _identify_specialized_assessments(
        self,
        exchanges: list[dict[str, Any]],
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Identify specialized assessments conducted."""
        assessments = []

        # Check for population-specific assessment elements
        for modification in characteristics.assessment_modifications:
            if any(modification.lower().split()[0] in e.get("content", "").lower()
                   for e in exchanges if e.get("speaker") == "therapist"):
                assessments.append(modification.lower().replace(" ", "_"))

        return assessments

    def _determine_treatment_modifications(
        self,
        exchanges: list[dict[str, Any]],
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Determine treatment modifications implemented."""
        modifications = []

        # Check for population-specific treatment considerations
        for consideration in characteristics.treatment_considerations:
            if any(consideration.lower().split()[0] in e.get("content", "").lower()
                   for e in exchanges if e.get("speaker") == "therapist"):
                modifications.append(consideration.lower().replace(" ", "_"))

        return modifications

    def _consider_outcome_factors(
        self,
        protocol: SpecializedProtocol,
        characteristics: PopulationCharacteristics
    ) -> list[str]:
        """Consider outcome factors for population."""

        # Combine protocol outcomes with population goals
        outcomes = []
        outcomes.extend(protocol.outcome_measures)
        outcomes.extend(characteristics.therapeutic_goals)

        return list(set(outcomes))

    def export_specialized_conversations(
        self,
        conversations: list[PopulationConversation],
        output_file: str
    ) -> dict[str, Any]:
        """Export specialized population conversations to JSON file."""

        # Convert conversations to JSON-serializable format
        serializable_conversations = []
        for conv in conversations:
            conv_dict = asdict(conv)
            self._convert_enums_to_strings(conv_dict)
            serializable_conversations.append(conv_dict)

        export_data = {
            "metadata": {
                "total_conversations": len(conversations),
                "export_timestamp": datetime.now().isoformat(),
                "populations_covered": list({
                    conv.population_characteristics.population.value for conv in conversations
                }),
                "cultural_adaptations_used": list({
                    adaptation for conv in conversations
                    for adaptation in conv.cultural_adaptations_used
                }),
                "trauma_informed_coverage": list({
                    element for conv in conversations
                    for element in conv.trauma_informed_elements
                })
            },
            "conversations": serializable_conversations
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "populations_covered": len(export_data["metadata"]["populations_covered"]),
            "cultural_adaptations": len(export_data["metadata"]["cultural_adaptations_used"]),
            "trauma_informed_elements": len(export_data["metadata"]["trauma_informed_coverage"])
        }

    def _convert_enums_to_strings(self, obj):
        """Recursively convert enum values to strings for JSON serialization."""
        if isinstance(obj, dict):
            # Handle enum keys in dictionaries
            new_dict = {}
            for key, value in obj.items():
                # Convert enum keys to strings
                new_key = key.value if hasattr(key, "value") else key

                # Convert enum values to strings
                if hasattr(value, "value"):
                    new_dict[new_key] = value.value
                elif isinstance(value, (dict, list)):
                    new_dict[new_key] = value
                    self._convert_enums_to_strings(new_dict[new_key])
                else:
                    new_dict[new_key] = value

            # Replace original dict contents
            obj.clear()
            obj.update(new_dict)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if hasattr(item, "value"):  # Enum object
                    obj[i] = item.value
                elif isinstance(item, (dict, list)):
                    self._convert_enums_to_strings(item)
