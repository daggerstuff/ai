"""
DSM-5 diagnostic criteria parser for psychology knowledge integration pipeline.

This module provides comprehensive parsing and structuring of DSM-5 diagnostic criteria
into standardized format for therapeutic conversation generation and training data creation.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger("dataset_pipeline.dsm5_parser")


class DSMCategory(Enum):
    """DSM-5 diagnostic categories."""
    NEURODEVELOPMENTAL = "neurodevelopmental_disorders"
    SCHIZOPHRENIA_SPECTRUM = "schizophrenia_spectrum_psychotic_disorders"
    BIPOLAR = "bipolar_related_disorders"
    DEPRESSIVE = "depressive_disorders"
    ANXIETY = "anxiety_disorders"
    OBSESSIVE_COMPULSIVE = "obsessive_compulsive_related_disorders"
    TRAUMA_STRESSOR = "trauma_stressor_related_disorders"
    DISSOCIATIVE = "dissociative_disorders"
    SOMATIC_SYMPTOM = "somatic_symptom_related_disorders"
    FEEDING_EATING = "feeding_eating_disorders"
    ELIMINATION = "elimination_disorders"
    SLEEP_WAKE = "sleep_wake_disorders"
    SEXUAL_DYSFUNCTIONS = "sexual_dysfunctions"
    GENDER_DYSPHORIA = "gender_dysphoria"
    DISRUPTIVE_IMPULSE = "disruptive_impulse_control_conduct_disorders"
    SUBSTANCE_RELATED = "substance_related_addictive_disorders"
    NEUROCOGNITIVE = "neurocognitive_disorders"
    PERSONALITY = "personality_disorders"
    PARAPHILIC = "paraphilic_disorders"
    OTHER = "other_mental_disorders"


class SeverityLevel(Enum):
    """Severity levels for disorders."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNSPECIFIED = "unspecified"


@dataclass
class DSMCriterion:
    """Individual DSM-5 diagnostic criterion."""
    id: str
    description: str
    category: str = ""
    required: bool = True
    examples: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)


@dataclass
class DSMSpecifier:
    """DSM-5 specifiers for disorders."""
    name: str
    description: str
    options: list[str] = field(default_factory=list)
    required: bool = False


@dataclass
class DSMDisorder:
    """Complete DSM-5 disorder definition."""
    code: str
    name: str
    category: DSMCategory
    criteria: list[DSMCriterion]
    minimum_criteria_count: int
    duration_requirement: str
    severity_levels: list[SeverityLevel] = field(default_factory=list)
    specifiers: list[DSMSpecifier] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    differential_diagnosis: list[str] = field(default_factory=list)
    prevalence: str | None = None
    onset_pattern: str | None = None
    course: str | None = None
    risk_factors: list[str] = field(default_factory=list)
    functional_consequences: list[str] = field(default_factory=list)


@dataclass
class DSMKnowledgeBase:
    """Complete DSM-5 knowledge base."""
    disorders: list[DSMDisorder]
    categories: dict[str, list[str]] = field(default_factory=dict)
    cross_references: dict[str, list[str]] = field(default_factory=dict)
    version: str = "DSM-5-TR"
    created_at: str | None = None


class DSM5Parser:
    """
    Comprehensive DSM-5 diagnostic criteria parser.

    Provides structured parsing of DSM-5 diagnostic criteria into standardized format
    for therapeutic conversation generation and clinical training data creation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the DSM-5 parser with configuration."""
        self.config = config or {}
        self.knowledge_base: DSMKnowledgeBase | None = None

        # Initialize sample disorders for demonstration
        self._initialize_sample_disorders()

        logger.info("DSM-5 Parser initialized")

    def _initialize_sample_disorders(self) -> None:
        """Initialize sample DSM-5 disorders for demonstration and testing."""
        sample_disorders = [
            self._create_major_depressive_disorder(),
            self._create_generalized_anxiety_disorder(),
            self._create_panic_disorder(),
            self._create_ptsd(),
            self._create_ocd(),
            self._create_bipolar_disorder(),
            self._create_adhd(),
            self._create_autism_spectrum_disorder()
        ]

        self.knowledge_base = DSMKnowledgeBase(
            disorders=sample_disorders,
            categories=self._build_category_mapping(sample_disorders),
            version="DSM-5-TR Sample",
            created_at="2024-01-01"
        )

        logger.info(f"Initialized {len(sample_disorders)} sample DSM-5 disorders")

    def _create_major_depressive_disorder(self) -> DSMDisorder:
        """Create Major Depressive Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A1",
                description="Depressed mood most of the day, nearly every day",
                category="core_symptoms",
                examples=[
                    "Feeling sad, empty, or hopeless",
                    "Appears tearful to others",
                    "Irritable mood in children and adolescents"
                ]
            ),
            DSMCriterion(
                id="A2",
                description="Markedly diminished interest or pleasure in all, or almost all, activities",
                category="core_symptoms",
                examples=[
                    "Loss of interest in hobbies",
                    "Withdrawal from social activities",
                    "Decreased sexual interest"
                ]
            ),
            DSMCriterion(
                id="A3",
                description="Significant weight loss when not dieting or weight gain, or decrease or increase in appetite",
                category="physical_symptoms",
                examples=[
                    "Weight change of more than 5% in a month",
                    "Decreased or increased appetite nearly every day"
                ]
            ),
            DSMCriterion(
                id="A4",
                description="Insomnia or hypersomnia nearly every day",
                category="sleep_symptoms",
                examples=[
                    "Difficulty falling asleep",
                    "Early morning awakening",
                    "Sleeping too much"
                ]
            ),
            DSMCriterion(
                id="A5",
                description="Psychomotor agitation or retardation nearly every day",
                category="motor_symptoms",
                examples=[
                    "Restlessness or feeling slowed down",
                    "Observable by others, not merely subjective"
                ]
            ),
            DSMCriterion(
                id="A6",
                description="Fatigue or loss of energy nearly every day",
                category="energy_symptoms",
                examples=[
                    "Feeling tired without physical exertion",
                    "Decreased efficiency in tasks"
                ]
            ),
            DSMCriterion(
                id="A7",
                description="Feelings of worthlessness or excessive or inappropriate guilt",
                category="cognitive_symptoms",
                examples=[
                    "Self-blame for things beyond control",
                    "Feelings of personal inadequacy"
                ]
            ),
            DSMCriterion(
                id="A8",
                description="Diminished ability to think or concentrate, or indecisiveness",
                category="cognitive_symptoms",
                examples=[
                    "Difficulty making decisions",
                    "Problems with memory or concentration"
                ]
            ),
            DSMCriterion(
                id="A9",
                description="Recurrent thoughts of death, recurrent suicidal ideation, or suicide attempt",
                category="suicidal_symptoms",
                examples=[
                    "Fear of dying",
                    "Suicidal thoughts without specific plan",
                    "Suicide attempt or specific plan"
                ]
            )
        ]

        specifiers = [
            DSMSpecifier(
                name="severity",
                description="Severity of current episode",
                options=["mild", "moderate", "severe"],
                required=True
            ),
            DSMSpecifier(
                name="course",
                description="Course specifiers",
                options=["single_episode", "recurrent", "in_partial_remission", "in_full_remission"]
            ),
            DSMSpecifier(
                name="features",
                description="Additional features",
                options=["with_anxious_distress", "with_mixed_features", "with_melancholic_features",
                        "with_atypical_features", "with_mood_congruent_psychotic_features",
                        "with_mood_incongruent_psychotic_features", "with_catatonia",
                        "with_peripartum_onset", "with_seasonal_pattern"]
            )
        ]

        return DSMDisorder(
            code="296.2x",
            name="Major Depressive Disorder",
            category=DSMCategory.DEPRESSIVE,
            criteria=criteria,
            minimum_criteria_count=5,
            duration_requirement="2 weeks",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE],
            specifiers=specifiers,
            exclusions=[
                "Symptoms not attributable to physiological effects of substance or medical condition",
                "Not better explained by schizoaffective, schizophrenia, or other psychotic disorders",
                "No history of manic or hypomanic episodes"
            ],
            differential_diagnosis=[
                "Bipolar Disorder",
                "Persistent Depressive Disorder",
                "Adjustment Disorder with Depressed Mood",
                "Substance-Induced Mood Disorder"
            ],
            prevalence="Approximately 7% in the United States",
            onset_pattern="Can occur at any age, with peak in 20s",
            course="Variable; may be single episode or recurrent",
            risk_factors=[
                "Family history of depression",
                "Stressful life events",
                "Medical conditions",
                "Substance use"
            ],
            functional_consequences=[
                "Impaired work performance",
                "Relationship difficulties",
                "Increased medical morbidity",
                "Suicide risk"
            ]
        )

    def _create_generalized_anxiety_disorder(self) -> DSMDisorder:
        """Create Generalized Anxiety Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Excessive anxiety and worry about a number of events or activities",
                category="core_symptoms",
                examples=[
                    "Worry about work performance",
                    "Concern about family safety",
                    "Anxiety about future events"
                ]
            ),
            DSMCriterion(
                id="B",
                description="Difficulty controlling the worry",
                category="control_symptoms",
                examples=[
                    "Unable to stop worrying",
                    "Worry interferes with concentration",
                    "Worry feels overwhelming"
                ]
            ),
            DSMCriterion(
                id="C1",
                description="Restlessness or feeling keyed up or on edge",
                category="physical_symptoms"
            ),
            DSMCriterion(
                id="C2",
                description="Being easily fatigued",
                category="physical_symptoms"
            ),
            DSMCriterion(
                id="C3",
                description="Difficulty concentrating or mind going blank",
                category="cognitive_symptoms"
            ),
            DSMCriterion(
                id="C4",
                description="Irritability",
                category="emotional_symptoms"
            ),
            DSMCriterion(
                id="C5",
                description="Muscle tension",
                category="physical_symptoms"
            ),
            DSMCriterion(
                id="C6",
                description="Sleep disturbance",
                category="sleep_symptoms",
                examples=[
                    "Difficulty falling asleep",
                    "Staying asleep",
                    "Restless, unsatisfying sleep"
                ]
            )
        ]

        return DSMDisorder(
            code="300.02",
            name="Generalized Anxiety Disorder",
            category=DSMCategory.ANXIETY,
            criteria=criteria,
            minimum_criteria_count=3,  # 3 of the C criteria
            duration_requirement="6 months",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE],
            exclusions=[
                "Not attributable to physiological effects of substance or medical condition",
                "Not better explained by another mental disorder"
            ],
            differential_diagnosis=[
                "Panic Disorder",
                "Social Anxiety Disorder",
                "Obsessive-Compulsive Disorder",
                "Adjustment Disorder with Anxiety"
            ],
            prevalence="Approximately 2.9% annually in the United States",
            onset_pattern="Can begin in childhood, adolescence, or adulthood",
            course="Chronic and fluctuating",
            risk_factors=[
                "Family history of anxiety",
                "Stressful life events",
                "Chronic medical conditions",
                "Substance use"
            ]
        )

    def _create_panic_disorder(self) -> DSMDisorder:
        """Create Panic Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Recurrent unexpected panic attacks",
                category="core_symptoms",
                examples=[
                    "Sudden onset of intense fear",
                    "Peak within minutes",
                    "Four or more panic attack symptoms"
                ]
            ),
            DSMCriterion(
                id="B1",
                description="Persistent concern about additional panic attacks",
                category="worry_symptoms"
            ),
            DSMCriterion(
                id="B2",
                description="Worry about implications or consequences of panic attacks",
                category="worry_symptoms",
                examples=[
                    "Fear of losing control",
                    "Fear of having a heart attack",
                    "Fear of going crazy"
                ]
            ),
            DSMCriterion(
                id="B3",
                description="Significant maladaptive change in behavior related to attacks",
                category="behavioral_symptoms",
                examples=[
                    "Avoidance of exercise",
                    "Avoidance of unfamiliar situations",
                    "Avoidance of being alone"
                ]
            )
        ]

        return DSMDisorder(
            code="300.01",
            name="Panic Disorder",
            category=DSMCategory.ANXIETY,
            criteria=criteria,
            minimum_criteria_count=2,  # A + at least one B criterion
            duration_requirement="1 month",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE],
            exclusions=[
                "Not attributable to physiological effects of substance or medical condition",
                "Not better explained by another mental disorder"
            ],
            differential_diagnosis=[
                "Generalized Anxiety Disorder",
                "Specific Phobia",
                "Social Anxiety Disorder",
                "Agoraphobia"
            ],
            prevalence="Approximately 2-3% annually in the United States",
            onset_pattern="Typically begins in late adolescence or early adulthood",
            course="Variable; may be episodic or persistent"
        )

    def _create_ptsd(self) -> DSMDisorder:
        """Create Post-Traumatic Stress Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Exposure to actual or threatened death, serious injury, or sexual violence",
                category="trauma_exposure",
                examples=[
                    "Directly experiencing traumatic event",
                    "Witnessing traumatic event",
                    "Learning of traumatic event to close family/friend",
                    "Repeated exposure to aversive details"
                ]
            ),
            DSMCriterion(
                id="B",
                description="Intrusion symptoms associated with traumatic event",
                category="intrusion_symptoms",
                examples=[
                    "Recurrent distressing memories",
                    "Recurrent distressing dreams",
                    "Dissociative reactions (flashbacks)",
                    "Intense psychological distress at cues",
                    "Marked physiological reactions to cues"
                ]
            ),
            DSMCriterion(
                id="C",
                description="Persistent avoidance of stimuli associated with traumatic event",
                category="avoidance_symptoms",
                examples=[
                    "Avoidance of distressing memories",
                    "Avoidance of external reminders"
                ]
            ),
            DSMCriterion(
                id="D",
                description="Negative alterations in cognitions and mood",
                category="cognitive_mood_symptoms",
                examples=[
                    "Inability to remember important aspects",
                    "Persistent negative beliefs about self/world",
                    "Distorted blame of self or others",
                    "Persistent negative emotional state",
                    "Diminished interest in activities",
                    "Feelings of detachment from others",
                    "Inability to experience positive emotions"
                ]
            ),
            DSMCriterion(
                id="E",
                description="Marked alterations in arousal and reactivity",
                category="arousal_symptoms",
                examples=[
                    "Irritable behavior and angry outbursts",
                    "Reckless or self-destructive behavior",
                    "Hypervigilance",
                    "Exaggerated startle response",
                    "Problems with concentration",
                    "Sleep disturbance"
                ]
            )
        ]

        return DSMDisorder(
            code="309.81",
            name="Post-Traumatic Stress Disorder",
            category=DSMCategory.TRAUMA_STRESSOR,
            criteria=criteria,
            minimum_criteria_count=5,  # All criteria A-E must be met
            duration_requirement="1 month",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE],
            specifiers=[
                DSMSpecifier(
                    name="dissociative_symptoms",
                    description="With dissociative symptoms",
                    options=["depersonalization", "derealization"]
                ),
                DSMSpecifier(
                    name="delayed_expression",
                    description="With delayed expression",
                    options=["delayed_onset"]
                )
            ],
            prevalence="Approximately 3.5% annually in the United States",
            onset_pattern="Can occur at any age",
            course="Variable; may be acute or chronic"
        )

    def _create_ocd(self) -> DSMDisorder:
        """Create Obsessive-Compulsive Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Presence of obsessions, compulsions, or both",
                category="core_symptoms",
                examples=[
                    "Obsessions: recurrent, persistent thoughts/urges/images",
                    "Compulsions: repetitive behaviors or mental acts"
                ]
            )
        ]

        return DSMDisorder(
            code="300.3",
            name="Obsessive-Compulsive Disorder",
            category=DSMCategory.OBSESSIVE_COMPULSIVE,
            criteria=criteria,
            minimum_criteria_count=1,
            duration_requirement="Time-consuming or cause distress",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE]
        )

    def _create_bipolar_disorder(self) -> DSMDisorder:
        """Create Bipolar I Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Criteria have been met for at least one manic episode",
                category="manic_episode"
            )
        ]

        return DSMDisorder(
            code="296.4x",
            name="Bipolar I Disorder",
            category=DSMCategory.BIPOLAR,
            criteria=criteria,
            minimum_criteria_count=1,
            duration_requirement="Variable",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE]
        )

    def _create_adhd(self) -> DSMDisorder:
        """Create ADHD definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Persistent pattern of inattention and/or hyperactivity-impulsivity",
                category="core_symptoms"
            )
        ]

        return DSMDisorder(
            code="314.0x",
            name="Attention-Deficit/Hyperactivity Disorder",
            category=DSMCategory.NEURODEVELOPMENTAL,
            criteria=criteria,
            minimum_criteria_count=1,
            duration_requirement="6 months",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE]
        )

    def _create_autism_spectrum_disorder(self) -> DSMDisorder:
        """Create Autism Spectrum Disorder definition."""
        criteria = [
            DSMCriterion(
                id="A",
                description="Persistent deficits in social communication and social interaction",
                category="social_communication"
            ),
            DSMCriterion(
                id="B",
                description="Restricted, repetitive patterns of behavior, interests, or activities",
                category="restricted_repetitive"
            )
        ]

        return DSMDisorder(
            code="299.00",
            name="Autism Spectrum Disorder",
            category=DSMCategory.NEURODEVELOPMENTAL,
            criteria=criteria,
            minimum_criteria_count=2,
            duration_requirement="Early developmental period",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE, SeverityLevel.SEVERE]
        )

    def _build_category_mapping(self, disorders: list[DSMDisorder]) -> dict[str, list[str]]:
        """Build mapping of categories to disorder names."""
        mapping = {}
        for disorder in disorders:
            category_name = disorder.category.value
            if category_name not in mapping:
                mapping[category_name] = []
            mapping[category_name].append(disorder.name)
        return mapping

    def get_disorders(self) -> list[DSMDisorder]:
        """Get all disorders in the knowledge base."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.disorders

    def get_disorder_by_name(self, name: str) -> DSMDisorder | None:
        """Get a specific disorder by name."""
        if not self.knowledge_base:
            return None

        for disorder in self.knowledge_base.disorders:
            if disorder.name.lower() == name.lower():
                return disorder
        return None

    def get_disorder_by_code(self, code: str) -> DSMDisorder | None:
        """Get a specific disorder by DSM code."""
        if not self.knowledge_base:
            return None

        for disorder in self.knowledge_base.disorders:
            if disorder.code == code:
                return disorder
        return None

    def get_disorders_by_category(self, category: DSMCategory) -> list[DSMDisorder]:
        """Get all disorders in a specific category."""
        if not self.knowledge_base:
            return []

        return [d for d in self.knowledge_base.disorders if d.category == category]

    def create_sample_disorders(self) -> list[dict[str, Any]]:
        """Create sample disorders for demonstration and testing."""
        if not self.knowledge_base:
            return []

        sample_data = []
        for disorder in self.knowledge_base.disorders:
            disorder_dict = asdict(disorder)
            # Convert enums to strings for JSON serialization
            disorder_dict["category"] = disorder.category.value
            disorder_dict["severity_levels"] = [level.value for level in disorder.severity_levels]
            sample_data.append(disorder_dict)

        logger.info(f"Created {len(sample_data)} sample disorders")
        return sample_data

    def export_to_json(self, output_path: Path) -> bool:
        """Export the knowledge base to JSON format."""
        try:
            if not self.knowledge_base:
                logger.error("No knowledge base to export")
                return False

            # Convert to dictionary for JSON serialization
            export_data = {
                "version": self.knowledge_base.version,
                "created_at": self.knowledge_base.created_at,
                "disorders": [],
                "categories": self.knowledge_base.categories,
                "cross_references": self.knowledge_base.cross_references
            }

            for disorder in self.knowledge_base.disorders:
                disorder_dict = asdict(disorder)
                # Convert enums to strings
                disorder_dict["category"] = disorder.category.value
                disorder_dict["severity_levels"] = [level.value for level in disorder.severity_levels]
                export_data["disorders"].append(disorder_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported DSM-5 knowledge base to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export DSM-5 knowledge base: {e}")
            return False

    def load_from_json(self, input_path: Path) -> bool:
        """Load knowledge base from JSON format."""
        try:
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return False

            with open(input_path, encoding="utf-8") as f:
                data = json.load(f)

            # Convert back to objects
            disorders = []
            for disorder_data in data.get("disorders", []):
                # Convert string enums back to enum objects
                disorder_data["category"] = DSMCategory(disorder_data["category"])
                disorder_data["severity_levels"] = [SeverityLevel(level) for level in disorder_data.get("severity_levels", [])]

                # Convert criteria
                criteria = []
                for criterion_data in disorder_data.get("criteria", []):
                    criteria.append(DSMCriterion(**criterion_data))
                disorder_data["criteria"] = criteria

                # Convert specifiers
                specifiers = []
                for specifier_data in disorder_data.get("specifiers", []):
                    specifiers.append(DSMSpecifier(**specifier_data))
                disorder_data["specifiers"] = specifiers

                disorders.append(DSMDisorder(**disorder_data))

            self.knowledge_base = DSMKnowledgeBase(
                disorders=disorders,
                categories=data.get("categories", {}),
                cross_references=data.get("cross_references", {}),
                version=data.get("version", "Unknown"),
                created_at=data.get("created_at")
            )

            logger.info(f"Loaded DSM-5 knowledge base from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load DSM-5 knowledge base: {e}")
            return False

    def generate_conversation_templates(self, disorder_name: str) -> list[Conversation]:
        """Generate conversation templates for a specific disorder."""
        disorder = self.get_disorder_by_name(disorder_name)
        if not disorder:
            logger.warning(f"Disorder not found: {disorder_name}")
            return []

        conversations = []

        # Create diagnostic conversation
        diagnostic_messages = [
            Message(
                role="therapist",
                content=f"I'd like to ask you some questions to better understand what you've been experiencing. These questions relate to {disorder.name}.",
                meta={"type": "introduction", "disorder": disorder.name}
            )
        ]

        for criterion in disorder.criteria:
            # Therapist question
            diagnostic_messages.append(Message(
                role="therapist",
                content=f"Have you experienced: {criterion.description}?",
                meta={"criterion_id": criterion.id, "category": criterion.category}
            ))

            # Sample client response
            if criterion.examples:
                example = criterion.examples[0]
                diagnostic_messages.append(Message(
                    role="client",
                    content=f"Yes, I have been experiencing {example.lower()}.",
                    meta={"criterion_id": criterion.id, "example": True}
                ))

        diagnostic_conversation = Conversation(
            id=f"dsm5_diagnostic_{disorder.code}",
            messages=diagnostic_messages,
            context={
                "disorder": disorder.name,
                "code": disorder.code,
                "category": disorder.category.value,
                "type": "diagnostic_assessment"
            },
            source="dsm5_parser",
            meta={
                "minimum_criteria": disorder.minimum_criteria_count,
                "duration_requirement": disorder.duration_requirement
            }
        )

        conversations.append(diagnostic_conversation)

        logger.info(f"Generated {len(conversations)} conversation templates for {disorder.name}")
        return conversations

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.knowledge_base:
            return {}

        stats = {
            "total_disorders": len(self.knowledge_base.disorders),
            "categories": {},
            "total_criteria": 0,
            "disorders_by_severity": {},
            "version": self.knowledge_base.version
        }

        for disorder in self.knowledge_base.disorders:
            # Category stats
            category = disorder.category.value
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1

            # Criteria count
            stats["total_criteria"] += len(disorder.criteria)

            # Severity levels
            for level in disorder.severity_levels:
                level_name = level.value
                if level_name not in stats["disorders_by_severity"]:
                    stats["disorders_by_severity"][level_name] = 0
                stats["disorders_by_severity"][level_name] += 1

        return stats
