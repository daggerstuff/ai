"""
Big Five personality assessment processor for psychology knowledge integration pipeline.

This module provides comprehensive processing and structuring of Big Five personality
assessments, clinical guidelines, and personality psychology frameworks for therapeutic
conversation generation and personality-based training data creation.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger("dataset_pipeline.big_five_processor")


class PersonalityFactor(Enum):
    """Big Five personality factors (OCEAN model)."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class AssessmentType(Enum):
    """Types of Big Five assessments."""
    NEO_PI_R = "neo_pi_r"  # NEO Personality Inventory-Revised
    BFI = "big_five_inventory"  # Big Five Inventory
    TIPI = "ten_item_personality_inventory"  # Ten-Item Personality Inventory
    BFI_2 = "big_five_inventory_2"  # Big Five Inventory-2
    IPIP = "international_personality_item_pool"  # IPIP Big Five


class ScoreLevel(Enum):
    """Personality trait score levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    AVERAGE = "average"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class PersonalityFacet:
    """Individual facet within a Big Five factor."""
    name: str
    factor: PersonalityFactor
    description: str
    high_score_characteristics: list[str] = field(default_factory=list)
    low_score_characteristics: list[str] = field(default_factory=list)
    assessment_items: list[str] = field(default_factory=list)
    clinical_implications: list[str] = field(default_factory=list)


@dataclass
class AssessmentItem:
    """Individual assessment item/question."""
    id: str
    text: str
    factor: PersonalityFactor
    facet: str | None = None
    reverse_scored: bool = False
    response_scale: str = "1-5 Likert"
    examples: list[str] = field(default_factory=list)


@dataclass
class PersonalityProfile:
    """Complete Big Five personality profile."""
    factor: PersonalityFactor
    name: str
    description: str
    facets: list[PersonalityFacet]
    score_interpretations: dict[str, list[str]] = field(default_factory=dict)
    clinical_considerations: list[str] = field(default_factory=list)
    therapeutic_implications: list[str] = field(default_factory=list)
    developmental_aspects: list[str] = field(default_factory=list)
    cultural_considerations: list[str] = field(default_factory=list)
    research_findings: list[str] = field(default_factory=list)


@dataclass
class BigFiveAssessment:
    """Complete Big Five assessment instrument."""
    name: str
    type: AssessmentType
    description: str
    items: list[AssessmentItem]
    administration_time: str
    target_population: list[str] = field(default_factory=list)
    reliability_data: dict[str, float] = field(default_factory=dict)
    validity_data: dict[str, str] = field(default_factory=dict)
    scoring_guidelines: list[str] = field(default_factory=list)
    interpretation_guidelines: list[str] = field(default_factory=list)
    clinical_applications: list[str] = field(default_factory=list)


@dataclass
class BigFiveKnowledgeBase:
    """Complete Big Five knowledge base."""
    personality_profiles: list[PersonalityProfile]
    assessments: list[BigFiveAssessment]
    clinical_guidelines: dict[str, list[str]] = field(default_factory=dict)
    research_findings: dict[str, list[str]] = field(default_factory=dict)
    version: str = "Big Five Clinical Framework v1.0"
    created_at: str | None = None


class BigFiveProcessor:
    """
    Comprehensive Big Five personality assessment processor.

    Provides structured processing of Big Five personality assessments, clinical
    guidelines, and personality psychology frameworks for therapeutic conversation
    generation and personality-based training data creation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the Big Five processor with configuration."""
        self.config = config or {}
        self.knowledge_base: BigFiveKnowledgeBase | None = None

        # Initialize sample assessments and profiles
        self._initialize_sample_data()

        logger.info("Big Five Processor initialized")

    def _initialize_sample_data(self) -> None:
        """Initialize sample Big Five data for demonstration and testing."""
        personality_profiles = [
            self._create_openness_profile(),
            self._create_conscientiousness_profile(),
            self._create_extraversion_profile(),
            self._create_agreeableness_profile(),
            self._create_neuroticism_profile()
        ]

        assessments = [
            self._create_bfi_assessment(),
            self._create_tipi_assessment()
        ]

        clinical_guidelines = self._create_clinical_guidelines()
        research_findings = self._create_research_findings()

        self.knowledge_base = BigFiveKnowledgeBase(
            personality_profiles=personality_profiles,
            assessments=assessments,
            clinical_guidelines=clinical_guidelines,
            research_findings=research_findings,
            version="Big Five Clinical Framework v1.0",
            created_at="2024-01-01"
        )

        logger.info(f"Initialized {len(personality_profiles)} personality profiles and {len(assessments)} assessments")

    def _create_openness_profile(self) -> PersonalityProfile:
        """Create Openness to Experience personality profile."""
        facets = [
            PersonalityFacet(
                name="Fantasy",
                factor=PersonalityFactor.OPENNESS,
                description="Receptivity to inner world of imagination",
                high_score_characteristics=[
                    "Vivid imagination and fantasy life",
                    "Daydreaming and creative thinking",
                    "Appreciation for unrealistic or unconventional ideas"
                ],
                low_score_characteristics=[
                    "Practical and realistic thinking",
                    "Preference for concrete over abstract",
                    "Limited imagination or fantasy"
                ],
                clinical_implications=[
                    "High fantasy may indicate creative potential or escapism",
                    "Low fantasy may suggest practical problem-solving style"
                ]
            ),
            PersonalityFacet(
                name="Aesthetics",
                factor=PersonalityFactor.OPENNESS,
                description="Appreciation for art, beauty, and aesthetic experiences",
                high_score_characteristics=[
                    "Deep appreciation for art and beauty",
                    "Moved by poetry, music, and artistic expression",
                    "Aesthetic sensitivity and artistic interests"
                ],
                low_score_characteristics=[
                    "Limited interest in art or aesthetic experiences",
                    "Practical rather than aesthetic focus",
                    "May find artistic pursuits unimportant"
                ]
            ),
            PersonalityFacet(
                name="Feelings",
                factor=PersonalityFactor.OPENNESS,
                description="Openness to inner emotional experiences",
                high_score_characteristics=[
                    "Access to and awareness of feelings",
                    "Values emotional experiences",
                    "Differentiated emotional states"
                ],
                low_score_characteristics=[
                    "Limited emotional awareness",
                    "May suppress or ignore feelings",
                    "Preference for rational over emotional"
                ]
            )
        ]

        return PersonalityProfile(
            factor=PersonalityFactor.OPENNESS,
            name="Openness to Experience",
            description="Reflects the degree of intellectual curiosity, creativity, and preference for novelty and variety",
            facets=facets,
            score_interpretations={
                "high": [
                    "Creative and imaginative",
                    "Intellectually curious",
                    "Open to new experiences",
                    "Appreciates art and beauty",
                    "Values intellectual pursuits"
                ],
                "low": [
                    "Conventional and traditional",
                    "Prefers familiar and routine",
                    "Practical and down-to-earth",
                    "Conservative in beliefs",
                    "Resistant to change"
                ]
            },
            clinical_considerations=[
                "High openness may correlate with creativity but also unconventional thinking",
                "Low openness may indicate stability but potential rigidity",
                "Consider cultural factors in interpretation"
            ],
            therapeutic_implications=[
                "High openness clients may benefit from creative therapeutic approaches",
                "Low openness clients may prefer structured, traditional interventions",
                "Explore balance between openness and stability"
            ]
        )

    def _create_conscientiousness_profile(self) -> PersonalityProfile:
        """Create Conscientiousness personality profile."""
        facets = [
            PersonalityFacet(
                name="Competence",
                factor=PersonalityFactor.CONSCIENTIOUSNESS,
                description="Sense of personal efficacy and capability",
                high_score_characteristics=[
                    "Feels capable and effective",
                    "Confident in ability to accomplish tasks",
                    "Self-efficacious and prepared"
                ],
                low_score_characteristics=[
                    "Feels unprepared or inadequate",
                    "Lacks confidence in abilities",
                    "May avoid challenging tasks"
                ]
            ),
            PersonalityFacet(
                name="Order",
                factor=PersonalityFactor.CONSCIENTIOUSNESS,
                description="Preference for organization and structure",
                high_score_characteristics=[
                    "Well-organized and methodical",
                    "Keeps things neat and tidy",
                    "Plans and structures activities"
                ],
                low_score_characteristics=[
                    "Disorganized and scattered",
                    "Comfortable with mess and chaos",
                    "Spontaneous rather than planned"
                ]
            )
        ]

        return PersonalityProfile(
            factor=PersonalityFactor.CONSCIENTIOUSNESS,
            name="Conscientiousness",
            description="Reflects the degree of organization, persistence, and motivation in goal-directed behavior",
            facets=facets,
            score_interpretations={
                "high": [
                    "Organized and responsible",
                    "Strong self-discipline",
                    "Achievement-oriented",
                    "Reliable and dependable",
                    "Plans ahead and follows through"
                ],
                "low": [
                    "Spontaneous and flexible",
                    "May be disorganized",
                    "Difficulty with long-term goals",
                    "Prefers immediate gratification",
                    "May procrastinate"
                ]
            },
            clinical_considerations=[
                "High conscientiousness associated with better health outcomes",
                "Low conscientiousness may indicate ADHD or impulse control issues",
                "Consider perfectionism in high scorers"
            ],
            therapeutic_implications=[
                "High conscientiousness clients may benefit from goal-setting approaches",
                "Low conscientiousness clients may need structure and accountability",
                "Address perfectionism or self-criticism in high scorers"
            ]
        )

    def _create_extraversion_profile(self) -> PersonalityProfile:
        """Create Extraversion personality profile."""
        facets = [
            PersonalityFacet(
                name="Warmth",
                factor=PersonalityFactor.EXTRAVERSION,
                description="Capacity for close, warm relationships",
                high_score_characteristics=[
                    "Affectionate and friendly",
                    "Forms close attachments easily",
                    "Genuinely likes people"
                ],
                low_score_characteristics=[
                    "Reserved and formal",
                    "Difficulty forming close relationships",
                    "May appear cold or distant"
                ]
            ),
            PersonalityFacet(
                name="Gregariousness",
                factor=PersonalityFactor.EXTRAVERSION,
                description="Preference for company of others",
                high_score_characteristics=[
                    "Seeks out social situations",
                    "Enjoys being around people",
                    "Feels comfortable in groups"
                ],
                low_score_characteristics=[
                    "Prefers solitude",
                    "Avoids social gatherings",
                    "Comfortable being alone"
                ]
            )
        ]

        return PersonalityProfile(
            factor=PersonalityFactor.EXTRAVERSION,
            name="Extraversion",
            description="Reflects the degree of sociability, assertiveness, and positive emotionality",
            facets=facets,
            score_interpretations={
                "high": [
                    "Outgoing and sociable",
                    "Assertive and energetic",
                    "Seeks stimulation and excitement",
                    "Talkative and expressive",
                    "Optimistic and positive"
                ],
                "low": [
                    "Reserved and quiet",
                    "Prefers solitude",
                    "Thoughtful and reflective",
                    "Independent and self-sufficient",
                    "May appear shy or withdrawn"
                ]
            },
            clinical_considerations=[
                "High extraversion may mask underlying issues",
                "Low extraversion not necessarily problematic",
                "Consider social anxiety vs. introversion"
            ],
            therapeutic_implications=[
                "High extraversion clients may benefit from group therapy",
                "Low extraversion clients may prefer individual therapy",
                "Respect introverted processing styles"
            ]
        )

    def _create_agreeableness_profile(self) -> PersonalityProfile:
        """Create Agreeableness personality profile."""
        facets = [
            PersonalityFacet(
                name="Trust",
                factor=PersonalityFactor.AGREEABLENESS,
                description="Tendency to believe others are honest and well-intentioned",
                high_score_characteristics=[
                    "Assumes others have good intentions",
                    "Trusting and forgiving",
                    "Believes people are fundamentally good"
                ],
                low_score_characteristics=[
                    "Skeptical of others' motives",
                    "Suspicious and cynical",
                    "Assumes others are selfish or dishonest"
                ]
            ),
            PersonalityFacet(
                name="Altruism",
                factor=PersonalityFactor.AGREEABLENESS,
                description="Active concern for others' welfare",
                high_score_characteristics=[
                    "Genuinely concerned for others",
                    "Willing to help those in need",
                    "Self-sacrificing and generous"
                ],
                low_score_characteristics=[
                    "Self-centered and selfish",
                    "Reluctant to help others",
                    "Puts own needs first"
                ]
            )
        ]

        return PersonalityProfile(
            factor=PersonalityFactor.AGREEABLENESS,
            name="Agreeableness",
            description="Reflects the degree of cooperation, trust, and concern for others",
            facets=facets,
            score_interpretations={
                "high": [
                    "Cooperative and trusting",
                    "Sympathetic and helpful",
                    "Considerate of others",
                    "Avoids conflict",
                    "Forgiving and generous"
                ],
                "low": [
                    "Competitive and skeptical",
                    "Direct and frank",
                    "Self-interested",
                    "May be argumentative",
                    "Tough-minded"
                ]
            },
            clinical_considerations=[
                "High agreeableness may indicate people-pleasing",
                "Low agreeableness may suggest interpersonal difficulties",
                "Consider cultural factors in interpretation"
            ],
            therapeutic_implications=[
                "High agreeableness clients may need assertiveness training",
                "Low agreeableness clients may benefit from empathy building",
                "Address boundary issues in high scorers"
            ]
        )

    def _create_neuroticism_profile(self) -> PersonalityProfile:
        """Create Neuroticism personality profile."""
        facets = [
            PersonalityFacet(
                name="Anxiety",
                factor=PersonalityFactor.NEUROTICISM,
                description="Tendency to experience anxiety, worry, and nervousness",
                high_score_characteristics=[
                    "Frequently anxious and worried",
                    "Anticipates problems and dangers",
                    "Feels tense and jittery"
                ],
                low_score_characteristics=[
                    "Generally calm and relaxed",
                    "Rarely feels anxious",
                    "Handles stress well"
                ]
            ),
            PersonalityFacet(
                name="Depression",
                factor=PersonalityFactor.NEUROTICISM,
                description="Tendency to experience sadness, hopelessness, and discouragement",
                high_score_characteristics=[
                    "Prone to feelings of sadness",
                    "Often feels hopeless or discouraged",
                    "May experience guilt and loneliness"
                ],
                low_score_characteristics=[
                    "Generally optimistic and hopeful",
                    "Rarely feels sad or discouraged",
                    "Resilient to setbacks"
                ]
            )
        ]

        return PersonalityProfile(
            factor=PersonalityFactor.NEUROTICISM,
            name="Neuroticism",
            description="Reflects the degree of emotional instability and tendency to experience negative emotions",
            facets=facets,
            score_interpretations={
                "high": [
                    "Emotionally reactive and sensitive",
                    "Prone to anxiety and worry",
                    "May experience mood swings",
                    "Stress-sensitive",
                    "Self-conscious and vulnerable"
                ],
                "low": [
                    "Emotionally stable and calm",
                    "Resilient to stress",
                    "Even-tempered",
                    "Self-confident",
                    "Rarely experiences negative emotions"
                ]
            },
            clinical_considerations=[
                "High neuroticism strongly associated with mental health issues",
                "Low neuroticism indicates emotional resilience",
                "Consider as risk factor for anxiety and depression"
            ],
            therapeutic_implications=[
                "High neuroticism clients may need emotion regulation skills",
                "Focus on stress management and coping strategies",
                "Address underlying anxiety and mood issues"
            ]
        )

    def _create_bfi_assessment(self) -> BigFiveAssessment:
        """Create Big Five Inventory (BFI) assessment."""
        items = [
            AssessmentItem(
                id="BFI_1",
                text="I see myself as someone who is talkative",
                factor=PersonalityFactor.EXTRAVERSION,
                facet="gregariousness",
                reverse_scored=False
            ),
            AssessmentItem(
                id="BFI_2",
                text="I see myself as someone who tends to find fault with others",
                factor=PersonalityFactor.AGREEABLENESS,
                facet="trust",
                reverse_scored=True
            ),
            AssessmentItem(
                id="BFI_3",
                text="I see myself as someone who does a thorough job",
                factor=PersonalityFactor.CONSCIENTIOUSNESS,
                facet="competence",
                reverse_scored=False
            ),
            AssessmentItem(
                id="BFI_4",
                text="I see myself as someone who is depressed, blue",
                factor=PersonalityFactor.NEUROTICISM,
                facet="depression",
                reverse_scored=False
            ),
            AssessmentItem(
                id="BFI_5",
                text="I see myself as someone who is original, comes up with new ideas",
                factor=PersonalityFactor.OPENNESS,
                facet="fantasy",
                reverse_scored=False
            )
        ]

        return BigFiveAssessment(
            name="Big Five Inventory (BFI)",
            type=AssessmentType.BFI,
            description="44-item self-report measure of Big Five personality dimensions",
            items=items,
            administration_time="5-10 minutes",
            target_population=["adults", "adolescents"],
            reliability_data={
                "openness": 0.81,
                "conscientiousness": 0.82,
                "extraversion": 0.88,
                "agreeableness": 0.79,
                "neuroticism": 0.84
            },
            scoring_guidelines=[
                "Rate each item on 1-5 scale (strongly disagree to strongly agree)",
                "Reverse score negatively keyed items",
                "Sum items for each factor",
                "Convert to percentile scores using norms"
            ],
            clinical_applications=[
                "Personality assessment in therapy",
                "Treatment planning and matching",
                "Research on personality and psychopathology"
            ]
        )

    def _create_tipi_assessment(self) -> BigFiveAssessment:
        """Create Ten-Item Personality Inventory (TIPI) assessment."""
        items = [
            AssessmentItem(
                id="TIPI_1",
                text="I see myself as extraverted, enthusiastic",
                factor=PersonalityFactor.EXTRAVERSION,
                reverse_scored=False
            ),
            AssessmentItem(
                id="TIPI_2",
                text="I see myself as critical, quarrelsome",
                factor=PersonalityFactor.AGREEABLENESS,
                reverse_scored=True
            ),
            AssessmentItem(
                id="TIPI_3",
                text="I see myself as dependable, self-disciplined",
                factor=PersonalityFactor.CONSCIENTIOUSNESS,
                reverse_scored=False
            ),
            AssessmentItem(
                id="TIPI_4",
                text="I see myself as anxious, easily upset",
                factor=PersonalityFactor.NEUROTICISM,
                reverse_scored=False
            ),
            AssessmentItem(
                id="TIPI_5",
                text="I see myself as open to new experiences, complex",
                factor=PersonalityFactor.OPENNESS,
                reverse_scored=False
            )
        ]

        return BigFiveAssessment(
            name="Ten-Item Personality Inventory (TIPI)",
            type=AssessmentType.TIPI,
            description="Brief 10-item measure of Big Five personality dimensions",
            items=items,
            administration_time="1-2 minutes",
            target_population=["adults"],
            reliability_data={
                "openness": 0.45,
                "conscientiousness": 0.50,
                "extraversion": 0.68,
                "agreeableness": 0.40,
                "neuroticism": 0.73
            },
            scoring_guidelines=[
                "Rate each item on 1-7 scale (strongly disagree to strongly agree)",
                "Average paired items for each factor",
                "Reverse score negatively keyed items"
            ],
            clinical_applications=[
                "Quick personality screening",
                "Research applications",
                "Time-limited assessments"
            ]
        )

    def _create_clinical_guidelines(self) -> dict[str, list[str]]:
        """Create clinical guidelines for Big Five assessment."""
        return {
            "assessment_best_practices": [
                "Use multiple assessment methods when possible",
                "Consider cultural and developmental factors",
                "Interpret scores in context of client's life circumstances",
                "Avoid pathologizing normal personality variations",
                "Use personality data to inform treatment planning"
            ],
            "interpretation_guidelines": [
                "Scores represent relative standing, not absolute traits",
                "Consider measurement error and confidence intervals",
                "Look for patterns across factors, not isolated scores",
                "Integrate with clinical interview and behavioral observations",
                "Consider stability and change over time"
            ],
            "therapeutic_applications": [
                "Match therapeutic approach to personality style",
                "Use personality insights for rapport building",
                "Address personality-related treatment barriers",
                "Incorporate strengths-based interventions",
                "Monitor personality change during treatment"
            ],
            "ethical_considerations": [
                "Obtain informed consent for personality assessment",
                "Protect confidentiality of personality data",
                "Avoid discriminatory use of personality information",
                "Provide clear, understandable feedback",
                "Consider cultural bias in assessment instruments"
            ]
        }

    def _create_research_findings(self) -> dict[str, list[str]]:
        """Create research findings about Big Five personality."""
        return {
            "mental_health_correlations": [
                "High neuroticism strongly predicts anxiety and depression",
                "Low conscientiousness associated with ADHD and substance use",
                "High openness may correlate with creativity but also psychosis risk",
                "Low agreeableness linked to antisocial and narcissistic traits",
                "Extraversion shows complex relationship with well-being"
            ],
            "treatment_outcomes": [
                "Personality factors predict therapy engagement and outcomes",
                "High conscientiousness associated with better treatment adherence",
                "High neuroticism may require longer treatment duration",
                "Personality-matched interventions show improved effectiveness",
                "Big Five factors moderate response to different therapy types"
            ],
            "developmental_patterns": [
                "Personality shows moderate stability across lifespan",
                "Mean-level changes occur with age and life experiences",
                "Conscientiousness and agreeableness tend to increase with age",
                "Neuroticism typically decreases in adulthood",
                "Openness may decline in later life"
            ],
            "cultural_considerations": [
                "Big Five structure replicated across many cultures",
                "Mean levels vary significantly between cultures",
                "Cultural values influence personality expression",
                "Assessment instruments may show cultural bias",
                "Interpretation requires cultural competence"
            ]
        }

    def get_personality_profiles(self) -> list[PersonalityProfile]:
        """Get all personality profiles in the knowledge base."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.personality_profiles

    def get_profile_by_factor(self, factor: PersonalityFactor) -> PersonalityProfile | None:
        """Get a specific personality profile by factor."""
        if not self.knowledge_base:
            return None

        for profile in self.knowledge_base.personality_profiles:
            if profile.factor == factor:
                return profile
        return None

    def get_assessments(self) -> list[BigFiveAssessment]:
        """Get all assessments in the knowledge base."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.assessments

    def get_assessment_by_type(self, assessment_type: AssessmentType) -> BigFiveAssessment | None:
        """Get a specific assessment by type."""
        if not self.knowledge_base:
            return None

        for assessment in self.knowledge_base.assessments:
            if assessment.type == assessment_type:
                return assessment
        return None

    def generate_conversation_templates(self, factor: PersonalityFactor) -> list[Conversation]:
        """Generate conversation templates for a specific personality factor."""
        profile = self.get_profile_by_factor(factor)
        if not profile:
            logger.warning(f"Personality profile not found: {factor}")
            return []

        conversations = []

        # Create assessment conversation
        assessment_messages = [
            Message(
                role="therapist",
                content=f"I'd like to explore your {profile.name.lower()} with you. This will help me understand your personality style better.",
                meta={"type": "introduction", "factor": factor.value}
            )
        ]

        # Add questions about high and low characteristics
        high_chars = profile.score_interpretations.get("high", [])
        profile.score_interpretations.get("low", [])

        if high_chars:
            assessment_messages.append(Message(
                role="therapist",
                content=f"Some people describe themselves as {high_chars[0].lower()}. How would you describe yourself in this area?",
                meta={"characteristic_type": "high", "factor": factor.value}
            ))

            assessment_messages.append(Message(
                role="client",
                content=f"I think I am fairly {high_chars[0].lower()}. For example, I often find myself engaging in activities that reflect this trait.",
                meta={"response_type": "high_agreement", "factor": factor.value}
            ))

        # Create therapeutic conversation
        therapeutic_messages = [
            Message(
                role="therapist",
                content=f"Based on our discussion about your {profile.name.lower()}, I'd like to explore how this affects your daily life and relationships.",
                meta={"type": "therapeutic_exploration", "factor": factor.value}
            )
        ]

        if profile.therapeutic_implications:
            implication = profile.therapeutic_implications[0]
            therapeutic_messages.append(Message(
                role="therapist",
                content=f"Given your personality style, {implication.lower()}. How does this resonate with your experience?",
                meta={"therapeutic_implication": True, "factor": factor.value}
            ))

        # Create assessment conversation
        assessment_conversation = Conversation(
            id=f"bigfive_assessment_{factor.value}",
            messages=assessment_messages,
            context={
                "factor": factor.value,
                "profile_name": profile.name,
                "type": "personality_assessment"
            },
            source="big_five_processor",
            meta={
                "clinical_considerations": profile.clinical_considerations,
                "therapeutic_implications": profile.therapeutic_implications
            }
        )

        # Create therapeutic conversation
        therapeutic_conversation = Conversation(
            id=f"bigfive_therapeutic_{factor.value}",
            messages=therapeutic_messages,
            context={
                "factor": factor.value,
                "profile_name": profile.name,
                "type": "therapeutic_exploration"
            },
            source="big_five_processor",
            meta={
                "clinical_considerations": profile.clinical_considerations,
                "therapeutic_implications": profile.therapeutic_implications
            }
        )

        conversations.extend([assessment_conversation, therapeutic_conversation])

        logger.info(f"Generated {len(conversations)} conversation templates for {profile.name}")
        return conversations

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
                "personality_profiles": [],
                "assessments": [],
                "clinical_guidelines": self.knowledge_base.clinical_guidelines,
                "research_findings": self.knowledge_base.research_findings
            }

            # Convert personality profiles
            for profile in self.knowledge_base.personality_profiles:
                profile_dict = asdict(profile)
                profile_dict["factor"] = profile.factor.value

                # Convert facets
                facets_list = []
                for facet in profile.facets:
                    facet_dict = asdict(facet)
                    facet_dict["factor"] = facet.factor.value
                    facets_list.append(facet_dict)
                profile_dict["facets"] = facets_list

                export_data["personality_profiles"].append(profile_dict)

            # Convert assessments
            for assessment in self.knowledge_base.assessments:
                assessment_dict = asdict(assessment)
                assessment_dict["type"] = assessment.type.value

                # Convert items
                items_list = []
                for item in assessment.items:
                    item_dict = asdict(item)
                    item_dict["factor"] = item.factor.value
                    items_list.append(item_dict)
                assessment_dict["items"] = items_list

                export_data["assessments"].append(assessment_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported Big Five knowledge base to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export Big Five knowledge base: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.knowledge_base:
            return {}

        stats = {
            "total_profiles": len(self.knowledge_base.personality_profiles),
            "total_assessments": len(self.knowledge_base.assessments),
            "factors_covered": [],
            "assessment_types": [],
            "total_facets": 0,
            "total_assessment_items": 0,
            "version": self.knowledge_base.version
        }

        for profile in self.knowledge_base.personality_profiles:
            stats["factors_covered"].append(profile.factor.value)
            stats["total_facets"] += len(profile.facets)

        for assessment in self.knowledge_base.assessments:
            stats["assessment_types"].append(assessment.type.value)
            stats["total_assessment_items"] += len(assessment.items)

        return stats
