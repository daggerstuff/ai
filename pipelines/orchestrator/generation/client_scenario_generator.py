"""
Client scenario generator for psychology knowledge integration pipeline.

This module creates realistic client scenarios by combining DSM-5 diagnostic
criteria, PDM-2 psychodynamic patterns, and Big Five personality profiles to
generate comprehensive client presentations for therapeutic conversation
training data.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.processing.big_five_processor import (
    BigFiveProcessor,
    PersonalityFactor,
)
from ai.pipelines.orchestrator.processing.dsm5_parser import (
    DSM5Parser,
    DSMCategory,
    DSMDisorder,
)
from ai.pipelines.orchestrator.processing.pdm2_parser import PDM2Parser
from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.client_scenario_generator")

# Constants for age thresholds
YOUNG_ADULT_THRESHOLD = 25
MIDDLE_AGE_THRESHOLD = 55
ADOLESCENT_THRESHOLD = 18
SENIOR_THRESHOLD = 65
ELDERLY_THRESHOLD = 75

# Constants for complexity thresholds
MULTIPLE_CONSIDERATIONS_THRESHOLD = 2
MULTIPLE_STRESSORS_THRESHOLD = 2
MINIMUM_OBJECTIVES = 2
MINIMUM_SYMPTOMS = 2
MINIMUM_TRIGGERS = 1
MINIMUM_GOALS = 2
MINIMUM_COMPLEXITY_FACTORS = 2
MINIMUM_CONSIDERATIONS = 2

# Constants for age grouping
AGE_GROUP_YOUNG = 30
AGE_GROUP_MIDDLE = 50

# Constants for quality scoring
QUALITY_THRESHOLD = 6
MAX_QUALITY_POINTS = 10


class ScenarioType(Enum):
    """Types of client scenarios."""

    INITIAL_ASSESSMENT = "initial_assessment"
    DIAGNOSTIC_INTERVIEW = "diagnostic_interview"
    THERAPEUTIC_SESSION = "therapeutic_session"
    CRISIS_INTERVENTION = "crisis_intervention"
    FOLLOW_UP_SESSION = "follow_up_session"


class SeverityLevel(Enum):
    """Severity levels for client presentations."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRISIS = "crisis"


class DemographicCategory(Enum):
    """Demographic categories for diversity."""

    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    OLDER_ADULT = "older_adult"
    ADOLESCENT = "adolescent"
    COLLEGE_STUDENT = "college_student"
    WORKING_PROFESSIONAL = "working_professional"
    RETIRED = "retired"


@dataclass
class ClientDemographics:
    """Client demographic information."""

    age: int
    gender: str
    occupation: str
    education_level: str
    relationship_status: str
    living_situation: str
    cultural_background: str
    socioeconomic_status: str
    insurance_type: str = "private"
    previous_therapy: bool = False


@dataclass
class PresentingProblem:
    """Client's presenting problem and symptoms."""

    primary_concern: str
    duration: str
    onset: str
    triggers: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)
    functional_impact: list[str] = field(default_factory=list)
    previous_episodes: bool = False
    current_stressors: list[str] = field(default_factory=list)


@dataclass
class ClinicalFormulation:
    """Clinical formulation integrating all psychology knowledge."""

    dsm5_considerations: list[str] = field(default_factory=list)
    attachment_style: str | None = None
    defense_mechanisms: list[str] = field(default_factory=list)
    personality_profile: dict[str, str] = field(default_factory=dict)
    psychodynamic_themes: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    protective_factors: list[str] = field(default_factory=list)
    treatment_goals: list[str] = field(default_factory=list)


@dataclass
class ClientScenario:
    """Complete client scenario for therapeutic training."""

    id: str
    scenario_type: ScenarioType
    severity_level: SeverityLevel
    demographics: ClientDemographics
    presenting_problem: PresentingProblem
    clinical_formulation: ClinicalFormulation
    session_context: dict[str, Any] = field(default_factory=dict)
    therapeutic_considerations: list[str] = field(default_factory=list)
    learning_objectives: list[str] = field(default_factory=list)
    complexity_factors: list[str] = field(default_factory=list)
    created_at: str | None = None


class ClientScenarioGenerator:
    """
    Comprehensive client scenario generator.

    Integrates DSM-5 diagnostic criteria, PDM-2 psychodynamic frameworks,
    and Big Five personality profiles to create realistic client scenarios
    for therapeutic conversation training data generation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the client scenario generator."""
        self.config = config or {}

        # Initialize psychology knowledge parsers
        self.dsm5_parser = DSM5Parser()
        self.pdm2_parser = PDM2Parser()
        self.big_five_processor = BigFiveProcessor()

        # Load demographic templates
        self._initialize_demographic_templates()

        logger.info("Client Scenario Generator initialized")

    def _initialize_demographic_templates(self) -> None:
        """Initialize demographic templates for scenario generation."""
        self.demographic_templates = {
            DemographicCategory.YOUNG_ADULT: {
                "age_range": (18, 25),
                "common_occupations": [
                    "student",
                    "retail worker",
                    "server",
                    "intern",
                    "entry-level professional",
                ],
                "common_concerns": [
                    "academic stress",
                    "relationship issues",
                    "career uncertainty",
                    "identity exploration",
                ],
                "living_situations": [
                    "with parents",
                    "college dorm",
                    "shared apartment",
                    "alone",
                ],
            },
            DemographicCategory.MIDDLE_AGED: {
                "age_range": (35, 55),
                "common_occupations": [
                    "manager",
                    "teacher",
                    "nurse",
                    "engineer",
                    "sales representative",
                ],
                "common_concerns": [
                    "work stress",
                    "parenting challenges",
                    "relationship issues",
                    "health concerns",
                ],
                "living_situations": [
                    "with spouse/partner",
                    "with family",
                    "alone",
                    "divorced parent",
                ],
            },
            DemographicCategory.OLDER_ADULT: {
                "age_range": (65, 85),
                "common_occupations": [
                    "retired",
                    "part-time consultant",
                    "volunteer",
                    "caregiver",
                ],
                "common_concerns": [
                    "health issues",
                    "loss/grief",
                    "isolation",
                    "cognitive concerns",
                ],
                "living_situations": [
                    "with spouse",
                    "alone",
                    "assisted living",
                    "with adult children",
                ],
            },
            DemographicCategory.ADOLESCENT: {
                "age_range": (13, 17),
                "common_occupations": ["student", "part-time worker"],
                "common_concerns": [
                    "identity issues",
                    "peer pressure",
                    "academic stress",
                    "family conflicts",
                ],
                "living_situations": [
                    "with parents",
                    "with guardians",
                    "boarding school",
                ],
            },
            DemographicCategory.COLLEGE_STUDENT: {
                "age_range": (18, 22),
                "common_occupations": ["student", "part-time worker", "intern"],
                "common_concerns": [
                    "academic pressure",
                    "social adjustment",
                    "career uncertainty",
                    "independence",
                ],
                "living_situations": [
                    "college dorm",
                    "shared apartment",
                    "with parents",
                    "alone",
                ],
            },
            DemographicCategory.WORKING_PROFESSIONAL: {
                "age_range": (25, 65),
                "common_occupations": [
                    "manager",
                    "professional",
                    "specialist",
                    "executive",
                ],
                "common_concerns": [
                    "work stress",
                    "career advancement",
                    "work-life balance",
                    "burnout",
                ],
                "living_situations": ["with spouse/partner", "alone", "with family"],
            },
            DemographicCategory.RETIRED: {
                "age_range": (60, 90),
                "common_occupations": [
                    "retired",
                    "volunteer",
                    "part-time consultant",
                    "caregiver",
                ],
                "common_concerns": [
                    "health issues",
                    "financial security",
                    "purpose/meaning",
                    "social isolation",
                ],
                "living_situations": [
                    "with spouse",
                    "alone",
                    "assisted living",
                    "with family",
                ],
            },
        }

        self.cultural_backgrounds = [
            "Caucasian American",
            "African American",
            "Hispanic/Latino",
            "Asian American",
            "Native American",
            "Middle Eastern",
            "Mixed ethnicity",
            "International student",
        ]

        self.education_levels = [
            "High school",
            "Some college",
            "Bachelor's degree",
            "Master's degree",
            "Doctoral degree",
            "Trade school",
            "Professional certification",
        ]

    def generate_client_scenario(
        self,
        scenario_type: ScenarioType | None = None,
        severity_level: SeverityLevel | None = None,
        target_disorder: str | None = None,
        demographic_category: DemographicCategory | None = None,
    ) -> ClientScenario:
        """Generate a single comprehensive client scenario."""

        # Set defaults if not specified
        scenario_type = scenario_type or random.choice(list(ScenarioType))
        severity_level = severity_level or random.choice(list(SeverityLevel))
        demographic_category = (
            demographic_category or random.choice(list(DemographicCategory))
        )

        # Generate scenario ID
        scenario_id = (
            f"scenario_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_"
            f"{random.randint(1000, 9999)}"
        )

        # Generate demographics
        demographics = self._generate_demographics(demographic_category)

        # Select or generate disorder/condition
        if target_disorder:
            disorder = self.dsm5_parser.get_disorder_by_name(target_disorder)
        else:
            disorders = self.dsm5_parser.get_disorders()
            disorder = random.choice(disorders) if disorders else None

        # Generate presenting problem
        presenting_problem = self._generate_presenting_problem(
            disorder, severity_level, demographics
        )

        # Generate clinical formulation
        clinical_formulation = self._generate_clinical_formulation(
            disorder, demographics, presenting_problem
        )

        # Generate session context
        session_context = self._generate_session_context(scenario_type)

        # Generate therapeutic considerations
        therapeutic_considerations = self._generate_therapeutic_considerations(
            disorder, clinical_formulation, demographics
        )

        # Generate learning objectives
        learning_objectives = self._generate_learning_objectives(
            scenario_type, disorder, severity_level
        )

        # Generate complexity factors
        complexity_factors = self._generate_complexity_factors(
            demographics, presenting_problem, clinical_formulation
        )

        scenario = ClientScenario(
            id=scenario_id,
            scenario_type=scenario_type,
            severity_level=severity_level,
            demographics=demographics,
            presenting_problem=presenting_problem,
            clinical_formulation=clinical_formulation,
            session_context=session_context,
            therapeutic_considerations=therapeutic_considerations,
            learning_objectives=learning_objectives,
            complexity_factors=complexity_factors,
            created_at=datetime.now(
                UTC
            ).isoformat(),  # DEBUG: UTC used, flagged by Ruff UP017
        )

        logger.info(
            f"Generated client scenario: {scenario_id} "
            f"({scenario_type.value}, {severity_level.value})"
        )
        return scenario

    def _generate_demographics(
        self, category: DemographicCategory
    ) -> ClientDemographics:
        """Generate realistic demographic information."""
        template = self.demographic_templates[category]

        age_range = template["age_range"]
        age = random.randint(int(age_range[0]), int(age_range[1]))
        gender = random.choice(["Male", "Female", "Non-binary"])
        occupation = str(random.choice(template["common_occupations"]))
        education_level = random.choice(self.education_levels)
        relationship_status = random.choice(
            [
                "Single",
                "In a relationship",
                "Married",
                "Divorced",
                "Widowed",
                "Separated",
            ]
        )
        living_situation = str(random.choice(template["living_situations"]))
        cultural_background = random.choice(self.cultural_backgrounds)
        socioeconomic_status = random.choice(
            ["Lower income", "Middle income", "Upper middle income", "High income"]
        )

        return ClientDemographics(
            age=age,
            gender=gender,
            occupation=occupation,
            education_level=education_level,
            relationship_status=relationship_status,
            living_situation=living_situation,
            cultural_background=cultural_background,
            socioeconomic_status=socioeconomic_status,
            insurance_type=random.choice(
                ["private", "public", "self-pay", "employee assistance"]
            ),
            previous_therapy=random.choice([True, False]),
        )

    def _generate_presenting_problem(
        self,
        disorder: DSMDisorder | None,
        severity: SeverityLevel,
        demographics: ClientDemographics,
    ) -> PresentingProblem:
        """Generate realistic presenting problem based on disorder and demographics."""

        if disorder:
            # Use disorder-specific symptoms
            primary_concern = f"Symptoms consistent with {disorder.name}"
            symptoms = []

            # Extract symptoms from disorder criteria
            for criterion in disorder.criteria[:3]:  # Use first 3 criteria
                if criterion.examples:
                    symptoms.extend(criterion.examples[:2])  # First 2 examples
                else:
                    symptoms.append(criterion.description)

            # Adjust for severity
            if severity == SeverityLevel.MILD:
                duration = random.choice(["2-4 weeks", "1-2 months", "Several weeks"])
                functional_impact = ["Mild interference with daily activities"]
            elif severity == SeverityLevel.MODERATE:
                duration = random.choice(["2-6 months", "Several months", "6 months"])
                functional_impact = [
                    "Moderate impact on work/relationships",
                    "Some difficulty with daily tasks",
                ]
            elif severity == SeverityLevel.SEVERE:
                duration = random.choice(["6+ months", "Over a year", "Chronic"])
                functional_impact = [
                    "Significant impairment in functioning",
                    "Unable to work/maintain relationships",
                ]
            else:  # CRISIS
                duration = random.choice(
                    ["Acute onset", "Recent escalation", "Current crisis"]
                )
                functional_impact = [
                    "Immediate safety concerns",
                    "Unable to function independently",
                ]

        else:
            # Generate general presenting problem
            primary_concern = random.choice(
                [
                    "Feeling overwhelmed and stressed",
                    "Relationship difficulties",
                    "Work-related stress",
                    "Life transition challenges",
                    "Mood changes and emotional difficulties",
                ]
            )
            symptoms = ["Stress", "Worry", "Sleep difficulties", "Mood changes"]
            duration = "Several weeks to months"
            functional_impact = ["Some impact on daily life"]

        # Generate contextual factors
        triggers = self._generate_triggers(demographics, disorder)
        current_stressors = self._generate_current_stressors(demographics)

        onset = random.choice(
            [
                "Gradual onset over time",
                "Sudden onset following stressful event",
                "Worsening of previous symptoms",
                "First-time experience",
            ]
        )

        return PresentingProblem(
            primary_concern=primary_concern,
            duration=duration,
            onset=onset,
            triggers=triggers,
            symptoms=symptoms,
            functional_impact=functional_impact,
            previous_episodes=random.choice([True, False]),
            current_stressors=current_stressors,
        )

    def _generate_triggers(
        self, demographics: ClientDemographics, disorder: DSMDisorder | None
    ) -> list[str]:
        """Generate realistic triggers based on demographics and disorder."""
        triggers = []

        # Age-based triggers
        if demographics.age < YOUNG_ADULT_THRESHOLD:
            triggers.extend(
                ["Academic pressure", "Peer relationships", "Family expectations"]
            )
        elif demographics.age < MIDDLE_AGE_THRESHOLD:
            triggers.extend(
                ["Work stress", "Parenting responsibilities", "Financial pressure"]
            )
        else:
            triggers.extend(
                ["Health concerns", "Retirement adjustment", "Loss of loved ones"]
            )

        # Relationship-based triggers
        if demographics.relationship_status in ["Divorced", "Separated"]:
            triggers.append("Recent relationship changes")
        elif demographics.relationship_status == "Married":
            triggers.append("Relationship conflicts")

        # Disorder-specific triggers
        if disorder and disorder.category == DSMCategory.ANXIETY:
            triggers.extend(
                ["Performance situations", "Social interactions", "Uncertainty"]
            )
        elif disorder and disorder.category == DSMCategory.DEPRESSIVE:
            triggers.extend(["Loss events", "Rejection", "Failure experiences"])

        return random.sample(triggers, min(3, len(triggers)))

    def _generate_current_stressors(
        self, demographics: ClientDemographics
    ) -> list[str]:
        """Generate current stressors based on demographics."""
        stressors = []

        # Occupation-based stressors
        if "student" in demographics.occupation.lower():
            stressors.extend(
                ["Academic deadlines", "Financial concerns", "Future uncertainty"]
            )
        elif demographics.occupation in ["manager", "teacher", "nurse"]:
            stressors.extend(
                ["Work overload", "Responsibility pressure", "Time management"]
            )
        elif demographics.occupation == "retired":
            stressors.extend(
                ["Health concerns", "Financial security", "Social isolation"]
            )

        # Life situation stressors
        if demographics.living_situation == "alone":
            stressors.append("Social isolation")
        elif "family" in demographics.living_situation:
            stressors.append("Family dynamics")

        return random.sample(stressors, min(3, len(stressors)))

    def _generate_clinical_formulation(
        self,
        disorder: DSMDisorder | None,
        demographics: ClientDemographics,
        presenting_problem: PresentingProblem,
    ) -> ClinicalFormulation:
        """
        Generate comprehensive clinical formulation integrating all psychology
        knowledge.
        """

        # DSM-5 considerations
        dsm5_considerations = []
        if disorder:
            dsm5_considerations.append(f"Consider {disorder.name} diagnosis")
            dsm5_considerations.extend(disorder.differential_diagnosis[:2])

        # PDM-2 attachment and defense patterns
        attachment_patterns = self.pdm2_parser.get_attachment_patterns()
        attachment_style = (
            random.choice(attachment_patterns).style.value
            if attachment_patterns
            else None
        )

        defense_mechanisms = []
        all_defenses = self.pdm2_parser.get_defense_mechanisms()
        if all_defenses:
            selected_defenses = random.sample(all_defenses, min(2, len(all_defenses)))
            defense_mechanisms = [defense.name for defense in selected_defenses]

        # Big Five personality profile
        personality_profile = {}
        for factor in PersonalityFactor:
            score_level = random.choice(["low", "average", "high"])
            personality_profile[factor.value] = score_level

        # Psychodynamic themes
        psychodynamic_themes = random.sample(
            [
                "Attachment and relationship patterns",
                "Self-esteem and identity issues",
                "Control and autonomy conflicts",
                "Loss and grief processing",
                "Unconscious conflict resolution",
            ],
            2,
        )

        # Risk and protective factors
        risk_factors = self._generate_risk_factors(demographics, presenting_problem)
        protective_factors = self._generate_protective_factors(demographics)

        # Treatment goals
        treatment_goals = self._generate_treatment_goals(disorder)

        return ClinicalFormulation(
            dsm5_considerations=dsm5_considerations,
            attachment_style=attachment_style,
            defense_mechanisms=defense_mechanisms,
            personality_profile=personality_profile,
            psychodynamic_themes=psychodynamic_themes,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            treatment_goals=treatment_goals,
        )

    def _generate_risk_factors(
        self, demographics: ClientDemographics, presenting_problem: PresentingProblem
    ) -> list[str]:
        """Generate risk factors based on client profile."""
        risk_factors = []

        # Demographic risk factors
        if demographics.age < YOUNG_ADULT_THRESHOLD:
            risk_factors.append("Young age and developmental transitions")
        if demographics.socioeconomic_status == "Lower income":
            risk_factors.append("Financial stress and limited resources")
        if demographics.previous_therapy:
            risk_factors.append("History of mental health treatment")

        # Presenting problem risk factors
        if (
            "severe"
            in presenting_problem.duration.lower()
            or "chronic" in presenting_problem.duration.lower()
        ):
            risk_factors.append("Chronic nature of symptoms")
        if presenting_problem.previous_episodes:
            risk_factors.append("History of previous episodes")

        # Social risk factors
        if demographics.living_situation == "alone":
            risk_factors.append("Social isolation")
        if demographics.relationship_status in ["Divorced", "Separated"]:
            risk_factors.append("Recent relationship disruption")

        return risk_factors[:4]  # Limit to 4 factors

    def _generate_protective_factors(
        self, demographics: ClientDemographics
    ) -> list[str]:
        """Generate protective factors based on client profile."""
        protective_factors = []

        # Education and employment
        if demographics.education_level in [
            "Bachelor's degree",
            "Master's degree",
            "Doctoral degree",
        ]:
            protective_factors.append("Higher education level")
        if demographics.occupation not in ["unemployed", "disabled"]:
            protective_factors.append("Stable employment")

        # Social support
        if demographics.relationship_status in ["Married", "In a relationship"]:
            protective_factors.append("Supportive relationship")
        if "family" in demographics.living_situation:
            protective_factors.append("Family support system")

        # Resources
        if demographics.insurance_type == "private":
            protective_factors.append("Good insurance coverage")
        if demographics.socioeconomic_status in [
            "Middle income",
            "Upper middle income",
            "High income",
        ]:
            protective_factors.append("Financial stability")

        # Personal factors
        protective_factors.append("Motivation for treatment")
        protective_factors.append("Insight into problems")

        return protective_factors[:4]  # Limit to 4 factors

    def _generate_treatment_goals(self, disorder: DSMDisorder | None) -> list[str]:
        """Generate treatment goals based on disorder."""
        goals = []

        # Symptom-specific goals
        if disorder:
            if disorder.category == DSMCategory.DEPRESSIVE:
                goals.extend(
                    [
                        "Improve mood and reduce depressive symptoms",
                        "Increase engagement in meaningful activities",
                        "Develop coping strategies for negative thoughts",
                    ]
                )
            elif disorder.category == DSMCategory.ANXIETY:
                goals.extend(
                    [
                        "Reduce anxiety and worry",
                        "Learn relaxation and grounding techniques",
                        "Gradually face avoided situations",
                    ]
                )
            elif disorder.category == DSMCategory.TRAUMA_STRESSOR:
                goals.extend(
                    [
                        "Process traumatic experiences safely",
                        "Reduce trauma-related symptoms",
                        "Develop healthy coping mechanisms",
                    ]
                )

        # General therapeutic goals
        goals.extend(
            [
                "Improve overall functioning and quality of life",
                "Enhance self-awareness and insight",
                "Strengthen interpersonal relationships",
                "Develop effective problem-solving skills",
            ]
        )

        return goals[:4]  # Limit to 4 goals

    def _generate_session_context(
        self,
        scenario_type: ScenarioType,
    ) -> dict[str, Any]:
        """Generate session context information."""
        context = {
            "session_number": (
                1
                if scenario_type == ScenarioType.INITIAL_ASSESSMENT
                else random.randint(2, 12)
            ),
            "setting": random.choice(
                [
                    "private practice",
                    "community mental health",
                    "hospital",
                    "university counseling",
                ]
            ),
            "modality": random.choice(["individual", "group", "family", "couples"]),
            "duration": random.choice(
                ["45 minutes", "50 minutes", "60 minutes", "90 minutes"]
            ),
            "referral_source": random.choice(
                ["self-referral", "physician", "family", "employer", "court-ordered"]
            ),
        }

        if scenario_type == ScenarioType.CRISIS_INTERVENTION:
            context.update(
                {
                    "urgency_level": "high",
                    "safety_assessment_needed": True,
                    "crisis_type": random.choice(
                        [
                            "suicidal ideation",
                            "panic attack",
                            "psychotic episode",
                            "substance use",
                        ]
                    ),
                }
            )

        return context

    def _generate_therapeutic_considerations(
        self,
        disorder: DSMDisorder | None,
        clinical_formulation: ClinicalFormulation,
        demographics: ClientDemographics,
    ) -> list[str]:
        """Generate therapeutic considerations for the scenario."""
        considerations = []

        # Disorder-specific considerations
        if disorder:
            consequences = list(disorder.functional_consequences)[:2]
            considerations.extend(consequences)

        # Cultural considerations
        if demographics.cultural_background != "Caucasian American":
            considerations.append(
                "Consider cultural factors in assessment and treatment"
            )

        # Age-specific considerations
        if demographics.age < ADOLESCENT_THRESHOLD:
            considerations.append("Developmental considerations for adolescent client")
        elif demographics.age > SENIOR_THRESHOLD:
            considerations.append(
                "Geriatric considerations and potential medical comorbidities"
            )

        # Attachment considerations
        if clinical_formulation.attachment_style:
            considerations.append(
                f"Consider {clinical_formulation.attachment_style} "
                "attachment patterns in therapy"
            )

        # Personality considerations
        profile = clinical_formulation.personality_profile
        high_neuroticism = profile.get("neuroticism") == "high"
        if high_neuroticism:
            considerations.append(
                "High neuroticism may require emotion regulation focus"
            )

        return considerations[:5]  # Limit to 5 considerations

    def _generate_learning_objectives(
        self,
        scenario_type: ScenarioType,
        disorder: DSMDisorder | None,
        severity: SeverityLevel,
    ) -> list[str]:
        """Generate learning objectives for the scenario."""
        objectives = []

        # Scenario-type specific objectives
        if scenario_type == ScenarioType.INITIAL_ASSESSMENT:
            objectives.extend(
                [
                    "Practice initial assessment and rapport building",
                    "Conduct comprehensive intake interview",
                    "Identify presenting problems and symptoms",
                ]
            )
        elif scenario_type == ScenarioType.DIAGNOSTIC_INTERVIEW:
            objectives.extend(
                [
                    "Apply diagnostic criteria systematically",
                    "Conduct differential diagnosis",
                    "Assess symptom severity and functional impact",
                ]
            )
        elif scenario_type == ScenarioType.CRISIS_INTERVENTION:
            objectives.extend(
                [
                    "Assess immediate safety and risk",
                    "Provide crisis stabilization",
                    "Develop safety planning",
                ]
            )

        # Disorder-specific objectives
        if disorder:
            objectives.append(
                f"Apply knowledge of {disorder.name} in clinical practice"
            )

        # Severity-specific objectives
        if severity == SeverityLevel.SEVERE:
            objectives.append("Manage complex and severe symptom presentations")

        return objectives[:4]  # Limit to 4 objectives

    def _generate_complexity_factors(
        self,
        demographics: ClientDemographics,
        presenting_problem: PresentingProblem,
        clinical_formulation: ClinicalFormulation,
    ) -> list[str]:
        """Generate complexity factors that make the scenario challenging."""
        factors = []

        # Demographic complexity
        if demographics.cultural_background != "Caucasian American":
            factors.append("Cross-cultural considerations")
        if (
            demographics.age < ADOLESCENT_THRESHOLD
            or demographics.age > ELDERLY_THRESHOLD
        ):
            factors.append("Age-specific developmental considerations")

        # Clinical complexity
        if (
            len(clinical_formulation.dsm5_considerations)
            > MULTIPLE_CONSIDERATIONS_THRESHOLD
        ):
            factors.append("Multiple diagnostic considerations")
        if presenting_problem.previous_episodes:
            factors.append("Recurrent condition with treatment history")

        # Psychosocial complexity
        if len(presenting_problem.current_stressors) > MULTIPLE_STRESSORS_THRESHOLD:
            factors.append("Multiple concurrent stressors")
        if demographics.socioeconomic_status == "Lower income":
            factors.append("Limited resources and access barriers")

        # Personality complexity
        profile = clinical_formulation.personality_profile
        high_neuroticism = profile.get("neuroticism") == "high"
        low_agreeableness = profile.get("agreeableness") == "low"
        if high_neuroticism and low_agreeableness:
            factors.append("Challenging personality presentation")

        return factors[:4]  # Limit to 4 factors

    def generate_scenario_batch(
        self,
        count: int = 10,
        scenario_types: list[ScenarioType] | None = None,
        severity_levels: list[SeverityLevel] | None = None,
        target_disorders: list[str] | None = None,
    ) -> list[ClientScenario]:
        """Generate a batch of client scenarios with diversity."""

        scenario_types = scenario_types or list(ScenarioType)
        severity_levels = severity_levels or list(SeverityLevel)

        scenarios = []

        for i in range(count):
            # Ensure diversity across scenarios
            scenario_type = scenario_types[i % len(scenario_types)]
            severity_level = severity_levels[i % len(severity_levels)]

            target_disorder = None
            if target_disorders:
                target_disorder = target_disorders[i % len(target_disorders)]

            scenario = self.generate_client_scenario(
                scenario_type=scenario_type,
                severity_level=severity_level,
                target_disorder=target_disorder,
            )
            scenarios.append(scenario)

        logger.info(f"Generated batch of {len(scenarios)} client scenarios")
        return scenarios

    def export_scenarios_to_json(
        self, scenarios: list[ClientScenario], output_path: Path
    ) -> bool:
        """Export scenarios to JSON format."""
        try:
            # Convert scenarios to dictionaries
            export_data: dict[str, Any] = {
                "scenarios": [],
                "metadata": {
                    "total_scenarios": len(scenarios),
                    "generated_at": datetime.now(
                        UTC
                    ).isoformat(),  # DEBUG: UTC used, flagged by Ruff UP017
                    "generator_version": "1.0",
                },
            }

            for scenario in scenarios:
                scenario_dict = asdict(scenario)
                # Convert enums to strings
                scenario_dict["scenario_type"] = scenario.scenario_type.value
                scenario_dict["severity_level"] = scenario.severity_level.value
                scenarios_list = export_data["scenarios"]
                (
                    scenarios_list if isinstance(scenarios_list, list) else []
                ).append(scenario_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(scenarios)} scenarios to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export scenarios: {e}")
            return False

    def generate_conversation_templates(
        self, scenario: ClientScenario
    ) -> list[Conversation]:
        """Generate conversation templates based on client scenario."""
        conversations = []

        # Create initial assessment conversation
        if scenario.scenario_type == ScenarioType.INITIAL_ASSESSMENT:
            messages = [
                Message(
                    role="therapist",
                    content=(
                        "Hello, I'm glad you decided to come in today. "
                        "What brings you to therapy?"
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"type": "opening", "scenario_id": scenario.id},
                ),
                Message(
                    role="client",
                    content=(
                        f"I've been struggling with "
                        f"{scenario.presenting_problem.primary_concern.lower()}. "
                        f"It's been going on for "
                        f"{scenario.presenting_problem.duration.lower()}."
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"presenting_problem": True, "scenario_id": scenario.id},
                ),
                Message(
                    role="therapist",
                    content=(
                        "I can hear that this has been difficult for you. "
                        "Can you tell me more about when these feelings started?"
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"technique": "reflection", "scenario_id": scenario.id},
                ),
            ]

            # Add symptom exploration
            if scenario.presenting_problem.symptoms:
                symptom = scenario.presenting_problem.symptoms[0]
                messages.append(
                    Message(
                        role="client",
                        content=(
                            f"I've been experiencing {symptom.lower()}, "
                            "and it's really affecting my daily life."
                        ),
                        timestamp=datetime.now(
                            UTC
                        ),  # DEBUG: UTC used, flagged by Ruff UP017
                        metadata={
                            "symptom_disclosure": True,
                            "scenario_id": scenario.id,
                        },
                    )
                )

        elif scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            messages = [
                Message(
                    role="therapist",
                    content=(
                        "I understand you're going through a very difficult time "
                        "right now. I want you to know that you're safe here, "
                        "and we're going to work through this together."
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"type": "crisis_opening", "scenario_id": scenario.id},
                ),
                Message(
                    role="client",
                    content=(
                        "I don't know what to do anymore. Everything feels "
                        "overwhelming and I can't handle it."
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"crisis_expression": True, "scenario_id": scenario.id},
                ),
            ]

        else:  # Regular therapeutic session
            messages = [
                Message(
                    role="therapist",
                    content="How have you been since our last session?",
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"type": "check_in", "scenario_id": scenario.id},
                ),
                Message(
                    role="client",
                    content=(
                        "I've been working on what we discussed, but I'm still "
                        "dealing with "
                        f"{scenario.presenting_problem.primary_concern.lower()}."
                    ),
                    timestamp=datetime.now(
                        UTC
                    ),  # DEBUG: UTC used, flagged by Ruff UP017
                    metadata={"progress_update": True, "scenario_id": scenario.id},
                ),
            ]

        conversation = Conversation(
            conversation_id=f"scenario_conversation_{scenario.id}",
            messages=messages,
            source="client_scenario_generator",
            created_at=datetime.now(UTC).isoformat(),
            metadata={
                "learning_objectives": scenario.learning_objectives,
                "therapeutic_considerations": scenario.therapeutic_considerations,
                "complexity_factors": scenario.complexity_factors,
                "scenario_id": scenario.id,
                "scenario_type": scenario.scenario_type.value,
                "severity_level": scenario.severity_level.value,
                "demographics": asdict(scenario.demographics),
                "clinical_formulation": asdict(scenario.clinical_formulation),
            },
        )
        conversations.append(conversation)

        logger.info(f"Generated conversation template for scenario {scenario.id}")
        return conversations

    def get_statistics(self, scenarios: list[ClientScenario]) -> dict[str, Any]:
        """Get statistics about generated scenarios."""
        if not scenarios:
            return {}

        stats: dict[str, Any] = {
            "total_scenarios": len(scenarios),
            "scenario_types": {},
            "severity_levels": {},
            "age_distribution": {},
            "cultural_backgrounds": {},
            "disorders_represented": {},
            "average_complexity_factors": 0,
        }

        total_complexity = 0

        for scenario in scenarios:
            # Scenario types
            scenario_type = scenario.scenario_type.value
            scenario_types_dict = stats["scenario_types"]
            if isinstance(scenario_types_dict, dict):
                scenario_types_dict[scenario_type] = (
                    scenario_types_dict.get(scenario_type, 0) + 1
                )

            # Severity levels
            severity = scenario.severity_level.value
            severity_levels_dict = stats["severity_levels"]
            if isinstance(severity_levels_dict, dict):
                severity_levels_dict[severity] = (
                    severity_levels_dict.get(severity, 0) + 1
                )

            # Age distribution
            age_group = (
                "18-30"
                if scenario.demographics.age <= AGE_GROUP_YOUNG
                else "31-50"
                if scenario.demographics.age <= AGE_GROUP_MIDDLE
                else "51+"
            )
            age_distribution_dict = stats["age_distribution"]
            if isinstance(age_distribution_dict, dict):
                age_distribution_dict[age_group] = (
                    age_distribution_dict.get(age_group, 0) + 1
                )

            # Cultural backgrounds
            culture = scenario.demographics.cultural_background
            cultural_backgrounds_dict = stats["cultural_backgrounds"]
            if isinstance(cultural_backgrounds_dict, dict):
                cultural_backgrounds_dict[culture] = (
                    cultural_backgrounds_dict.get(culture, 0) + 1
                )

            # Disorders
            if scenario.clinical_formulation.dsm5_considerations:
                disorder = scenario.clinical_formulation.dsm5_considerations[0]
                disorders_represented_dict = stats["disorders_represented"]
                if isinstance(disorders_represented_dict, dict):
                    disorders_represented_dict[disorder] = (
                        disorders_represented_dict.get(disorder, 0) + 1
                    )

            # Complexity
            total_complexity += len(scenario.complexity_factors)

        stats["average_complexity_factors"] = (
            total_complexity / len(scenarios) if scenarios else 0
        )

        return stats

    def load_scenarios_from_json(self, input_path: Path) -> list[ClientScenario]:
        """Load scenarios from JSON format."""
        try:
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return []

            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)

            scenarios = []
            for scenario_data in data.get("scenarios", []):
                # Convert string enums back to enum objects
                scenario_data["scenario_type"] = ScenarioType(
                    scenario_data["scenario_type"]
                )
                scenario_data["severity_level"] = SeverityLevel(
                    scenario_data["severity_level"]
                )

                # Convert nested dataclasses
                scenario_data["demographics"] = ClientDemographics(
                    **scenario_data["demographics"]
                )
                scenario_data["presenting_problem"] = PresentingProblem(
                    **scenario_data["presenting_problem"]
                )
                scenario_data["clinical_formulation"] = ClinicalFormulation(
                    **scenario_data["clinical_formulation"]
                )

                scenarios.append(ClientScenario(**scenario_data))

            logger.info(f"Loaded {len(scenarios)} scenarios from {input_path}")
            return scenarios

        except Exception as e:
            logger.error(f"Failed to load scenarios: {e}")
            return []

    def validate_scenario_quality(self, scenario: ClientScenario) -> dict[str, Any]:
        """Validate the quality and completeness of a client scenario."""
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "issues": [],
            "strengths": [],
        }
        quality_points = 0
        max_points = MAX_QUALITY_POINTS

        # Demographic completeness
        (
            demo_points,
            demo_issues,
            demo_strengths,
        ) = self._check_demographic_completeness(scenario)
        quality_points += demo_points
        (
            validation_results["issues"]
            if isinstance(validation_results["issues"], list)
            else []
        ).extend(demo_issues)
        (
            validation_results["strengths"]
            if isinstance(validation_results["strengths"], list)
            else []
        ).extend(demo_strengths)

        # Presenting problem detail
        (
            pp_points,
            pp_issues,
            pp_strengths,
        ) = self._check_presenting_problem_detail(scenario)
        quality_points += pp_points
        (
            validation_results["issues"]
            if isinstance(validation_results["issues"], list)
            else []
        ).extend(pp_issues)
        (
            validation_results["strengths"]
            if isinstance(validation_results["strengths"], list)
            else []
        ).extend(pp_strengths)

        # Clinical formulation depth
        (
            cf_points,
            cf_issues,
            cf_strengths,
        ) = self._check_clinical_formulation_depth(scenario)
        quality_points += cf_points
        (
            validation_results["issues"]
            if isinstance(validation_results["issues"], list)
            else []
        ).extend(cf_issues)
        (
            validation_results["strengths"]
            if isinstance(validation_results["strengths"], list)
            else []
        ).extend(cf_strengths)

        # Learning objectives
        lo_points, lo_issues, lo_strengths = self._check_learning_objectives(scenario)
        quality_points += lo_points
        (
            validation_results["issues"]
            if isinstance(validation_results["issues"], list)
            else []
        ).extend(lo_issues)
        (
            validation_results["strengths"]
            if isinstance(validation_results["strengths"], list)
            else []
        ).extend(lo_strengths)

        # Complexity and realism
        (
            cr_points,
            cr_issues,
            cr_strengths,
        ) = self._check_complexity_and_realism(scenario)
        quality_points += cr_points
        (
            validation_results["issues"]
            if isinstance(validation_results["issues"], list)
            else []
        ).extend(cr_issues)
        (
            validation_results["strengths"]
            if isinstance(validation_results["strengths"], list)
            else []
        ).extend(cr_strengths)

        validation_results["quality_score"] = quality_points / max_points
        # 60% threshold
        validation_results["is_valid"] = (
            quality_points >= QUALITY_THRESHOLD
        )

        return validation_results

    def _check_demographic_completeness(self, scenario: ClientScenario):
        logger.info(
            f"Checking demographic completeness for scenario {scenario.id}: "
            f"{scenario.demographics}"
        )
        points = 0
        issues = []
        strengths = []
        if all(
            [
                scenario.demographics.age > 0,
                scenario.demographics.gender,
                scenario.demographics.occupation,
                scenario.demographics.cultural_background,
            ]
        ):
            points += 2
            strengths.append("Complete demographic information")
        else:
            issues.append("Incomplete demographic information")
        return points, issues, strengths

    def _check_presenting_problem_detail(self, scenario: ClientScenario):
        logger.info(
            f"Checking presenting problem detail for scenario {scenario.id}: "
            f"{scenario.presenting_problem}"
        )
        points = 0
        issues = []
        strengths = []
        if (
            len(scenario.presenting_problem.symptoms) >= MINIMUM_SYMPTOMS
            and len(scenario.presenting_problem.triggers) >= MINIMUM_TRIGGERS
            and scenario.presenting_problem.duration
        ):
            points += 2
            strengths.append("Detailed presenting problem")
        else:
            issues.append("Insufficient presenting problem detail")
        return points, issues, strengths

    def _check_clinical_formulation_depth(self, scenario: ClientScenario):
        logger.info(
            f"Checking clinical formulation depth for scenario {scenario.id}: "
            f"{scenario.clinical_formulation}"
        )
        points = 0
        issues = []
        strengths = []
        if (
            len(scenario.clinical_formulation.dsm5_considerations) >= 1
            and len(scenario.clinical_formulation.treatment_goals) >= MINIMUM_GOALS
        ):
            points += 2
            strengths.append("Comprehensive clinical formulation")
        else:
            issues.append("Limited clinical formulation")
        return points, issues, strengths

    def _check_learning_objectives(self, scenario: ClientScenario):
        logger.info(
            f"Checking learning objectives for scenario {scenario.id}: "
            f"{scenario.learning_objectives}"
        )
        points = 0
        issues = []
        strengths = []
        if len(scenario.learning_objectives) >= MINIMUM_OBJECTIVES:
            points += 2
            strengths.append("Clear learning objectives")
        else:
            issues.append("Insufficient learning objectives")
        return points, issues, strengths

    def _check_complexity_and_realism(self, scenario: ClientScenario):
        logger.info(
            f"Checking complexity and realism for scenario {scenario.id}: "
            f"complexity_factors={scenario.complexity_factors}, "
            f"therapeutic_considerations={scenario.therapeutic_considerations}"
        )
        points = 0
        issues = []
        strengths = []
        if (
            len(scenario.complexity_factors) >= MINIMUM_COMPLEXITY_FACTORS
            and len(scenario.therapeutic_considerations) >= MINIMUM_CONSIDERATIONS
        ):
            points += 2
            strengths.append("Realistic complexity and considerations")
        else:
            issues.append("Limited complexity or therapeutic considerations")
        return points, issues, strengths

    def generate_scenario_variations(
        self, base_scenario: ClientScenario, count: int = 3
    ) -> list[ClientScenario]:
        """Generate variations of a base scenario with different parameters."""
        variations = []

        for i in range(count):
            # Create variation with different severity or demographic
            variation_severity = random.choice(
                [
                    level
                    for level in SeverityLevel
                    if level != base_scenario.severity_level
                ]
            )
            variation_demo_category = random.choice(list(DemographicCategory))

            variation = self.generate_client_scenario(
                scenario_type=base_scenario.scenario_type,
                severity_level=variation_severity,
                demographic_category=variation_demo_category,
            )

            # Modify ID to indicate it's a variation
            variation.id = f"{base_scenario.id}_var_{i + 1}"
            variations.append(variation)

        logger.info(
            f"Generated {len(variations)} variations of scenario {base_scenario.id}"
        )
        return variations
