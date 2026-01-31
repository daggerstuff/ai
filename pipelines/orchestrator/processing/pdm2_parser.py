"""
PDM-2 (Psychodynamic Diagnostic Manual) parser for psychology knowledge integration pipeline.

This module provides comprehensive parsing and structuring of PDM-2 psychodynamic frameworks,
attachment styles, defense mechanisms, and personality patterns for therapeutic conversation
generation and psychodynamic training data creation.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.utils.logger import get_logger

logger = get_logger("dataset_pipeline.pdm2_parser")


class AttachmentStyle(Enum):
    """Attachment styles from attachment theory."""

    SECURE = "secure"
    ANXIOUS_PREOCCUPIED = "anxious_preoccupied"
    DISMISSIVE_AVOIDANT = "dismissive_avoidant"
    DISORGANIZED_FEARFUL_AVOIDANT = "disorganized_fearful_avoidant"


class DefenseMechanismLevel(Enum):
    """Defense mechanism maturity levels."""

    MATURE = "mature"
    NEUROTIC = "neurotic"
    IMMATURE = "immature"
    PATHOLOGICAL = "pathological"


class PsychodynamicDomain(Enum):
    """PDM-2 personality domains."""

    ATTACHMENT_RELATIONSHIPS = "attachment_relationships"
    AFFECT_REGULATION = "affect_regulation"
    COGNITIVE_PATTERNS = "cognitive_patterns"
    IDENTITY_SELF_CONCEPT = "identity_self_concept"
    DEFENSIVE_PATTERNS = "defensive_patterns"
    INTERPERSONAL_FUNCTIONING = "interpersonal_functioning"


class DevelopmentalLevel(Enum):
    """Developmental considerations."""

    EARLY_CHILDHOOD = "early_childhood"
    MIDDLE_CHILDHOOD = "middle_childhood"
    ADOLESCENCE = "adolescence"
    EARLY_ADULTHOOD = "early_adulthood"
    MIDDLE_ADULTHOOD = "middle_adulthood"
    LATER_ADULTHOOD = "later_adulthood"


@dataclass
class AttachmentPattern:
    """Attachment style pattern definition."""

    style: AttachmentStyle
    name: str
    description: str
    characteristics: list[str] = field(default_factory=list)
    behavioral_indicators: list[str] = field(default_factory=list)
    therapeutic_considerations: list[str] = field(default_factory=list)
    developmental_origins: list[str] = field(default_factory=list)
    relationship_patterns: list[str] = field(default_factory=list)
    emotional_regulation: list[str] = field(default_factory=list)


@dataclass
class DefenseMechanism:
    """Defense mechanism definition."""

    name: str
    level: DefenseMechanismLevel
    description: str
    function: str
    examples: list[str] = field(default_factory=list)
    adaptive_aspects: list[str] = field(default_factory=list)
    maladaptive_aspects: list[str] = field(default_factory=list)
    therapeutic_approach: list[str] = field(default_factory=list)
    developmental_context: str | None = None


@dataclass
class PsychodynamicPattern:
    """Psychodynamic personality pattern."""

    name: str
    domain: PsychodynamicDomain
    description: str
    core_features: list[str] = field(default_factory=list)
    unconscious_conflicts: list[str] = field(default_factory=list)
    object_relations: list[str] = field(default_factory=list)
    transference_patterns: list[str] = field(default_factory=list)
    countertransference_patterns: list[str] = field(default_factory=list)
    therapeutic_goals: list[str] = field(default_factory=list)
    treatment_considerations: list[str] = field(default_factory=list)


@dataclass
class PDM2KnowledgeBase:
    """Complete PDM-2 knowledge base."""

    attachment_patterns: list[AttachmentPattern]
    defense_mechanisms: list[DefenseMechanism]
    psychodynamic_patterns: list[PsychodynamicPattern]
    theoretical_frameworks: dict[str, list[str]] = field(default_factory=dict)
    developmental_considerations: dict[str, list[str]] = field(default_factory=dict)
    version: str = "PDM-2"
    created_at: str | None = None


class PDM2Parser:
    """
    Comprehensive PDM-2 psychodynamic frameworks parser.

    Provides structured parsing of PDM-2 psychodynamic concepts including attachment styles,
    defense mechanisms, and personality patterns for therapeutic conversation generation
    and psychodynamic training data creation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the PDM-2 parser with configuration."""
        self.config = config or {}
        self.knowledge_base: PDM2KnowledgeBase | None = None

        # Initialize sample psychodynamic concepts
        self._initialize_sample_concepts()

        logger.info("PDM-2 Parser initialized")

    def _initialize_sample_concepts(self) -> None:
        """Initialize sample PDM-2 concepts for demonstration and testing."""
        attachment_patterns = self._create_attachment_patterns()
        defense_mechanisms = self._create_defense_mechanisms()
        psychodynamic_patterns = self._create_psychodynamic_patterns()

        self.knowledge_base = PDM2KnowledgeBase(
            attachment_patterns=attachment_patterns,
            defense_mechanisms=defense_mechanisms,
            psychodynamic_patterns=psychodynamic_patterns,
            theoretical_frameworks=self._build_theoretical_frameworks(),
            developmental_considerations=self._build_developmental_considerations(),
            version="PDM-2 Sample",
            created_at="2024-01-01",
        )

        total_concepts = (
            len(attachment_patterns) + len(defense_mechanisms) + len(psychodynamic_patterns)
        )
        logger.info(f"Initialized {total_concepts} PDM-2 psychodynamic concepts")

    def _create_attachment_patterns(self) -> list[AttachmentPattern]:
        """Create attachment style patterns."""
        patterns = []

        # Secure Attachment
        patterns.append(
            AttachmentPattern(
                style=AttachmentStyle.SECURE,
                name="Secure Attachment",
                description="Comfortable with intimacy and autonomy, positive view of self and others",
                characteristics=[
                    "Comfortable with closeness and independence",
                    "Positive view of self and others",
                    "Effective emotion regulation",
                    "Clear communication of needs",
                    "Trusting and supportive relationships",
                ],
                behavioral_indicators=[
                    "Seeks comfort when distressed",
                    "Explores environment confidently",
                    "Maintains stable relationships",
                    "Expresses emotions appropriately",
                    "Shows empathy and understanding",
                ],
                therapeutic_considerations=[
                    "Generally good therapeutic alliance",
                    "Able to explore difficult emotions",
                    "Responds well to insight-oriented approaches",
                    "Can tolerate therapeutic challenges",
                ],
                developmental_origins=[
                    "Consistent, responsive caregiving",
                    "Emotional attunement from caregivers",
                    "Safe, predictable environment",
                    "Balanced autonomy and support",
                ],
                relationship_patterns=[
                    "Seeks mutual support and intimacy",
                    "Comfortable with interdependence",
                    "Effective conflict resolution",
                    "Maintains individual identity in relationships",
                ],
                emotional_regulation=[
                    "Adaptive coping strategies",
                    "Ability to self-soothe",
                    "Seeks appropriate support",
                    "Balanced emotional expression",
                ],
            )
        )

        # Anxious-Preoccupied Attachment
        patterns.append(
            AttachmentPattern(
                style=AttachmentStyle.ANXIOUS_PREOCCUPIED,
                name="Anxious-Preoccupied Attachment",
                description="High anxiety about relationships, seeks excessive closeness, fear of abandonment",
                characteristics=[
                    "High relationship anxiety",
                    "Fear of abandonment",
                    "Seeks excessive reassurance",
                    "Preoccupied with relationships",
                    "Negative self-view, positive view of others",
                ],
                behavioral_indicators=[
                    "Clingy or demanding behavior",
                    "Hypervigilant to relationship threats",
                    "Difficulty with partner's independence",
                    "Emotional dysregulation in relationships",
                    "Seeks constant validation",
                ],
                therapeutic_considerations=[
                    "May form intense therapeutic attachment",
                    "Fear of therapeutic abandonment",
                    "Benefits from consistent, reliable therapy",
                    "May test therapeutic boundaries",
                ],
                developmental_origins=[
                    "Inconsistent caregiving",
                    "Unpredictable emotional availability",
                    "Anxious or overwhelmed caregivers",
                    "Early separation or loss experiences",
                ],
                relationship_patterns=[
                    "Pursues closeness intensely",
                    "Difficulty tolerating partner's autonomy",
                    "Jealousy and possessiveness",
                    "Emotional volatility in relationships",
                ],
                emotional_regulation=[
                    "Difficulty self-soothing",
                    "Seeks external regulation",
                    "Intense emotional reactions",
                    "Rumination and worry",
                ],
            )
        )

        # Dismissive-Avoidant Attachment
        patterns.append(
            AttachmentPattern(
                style=AttachmentStyle.DISMISSIVE_AVOIDANT,
                name="Dismissive-Avoidant Attachment",
                description="Discomfort with closeness, values independence, minimizes emotional needs",
                characteristics=[
                    "Discomfort with intimacy",
                    "Values self-reliance",
                    "Minimizes emotional needs",
                    "Positive self-view, negative view of others",
                    "Difficulty accessing emotions",
                ],
                behavioral_indicators=[
                    "Avoids emotional conversations",
                    "Maintains emotional distance",
                    "Difficulty expressing vulnerability",
                    "Self-sufficient presentation",
                    "Minimizes importance of relationships",
                ],
                therapeutic_considerations=[
                    "May be reluctant to engage emotionally",
                    "Benefits from gradual trust building",
                    "Cognitive approaches may be initially preferred",
                    "Needs respect for autonomy",
                ],
                developmental_origins=[
                    "Emotionally unavailable caregivers",
                    "Rejection of emotional needs",
                    "Emphasis on self-reliance",
                    "Lack of emotional attunement",
                ],
                relationship_patterns=[
                    "Maintains emotional distance",
                    "Difficulty with partner's emotional needs",
                    "Values independence over intimacy",
                    "May withdraw during conflict",
                ],
                emotional_regulation=[
                    "Suppression of emotions",
                    "Self-reliant coping",
                    "Difficulty seeking support",
                    "Intellectualization of feelings",
                ],
            )
        )

        # Disorganized/Fearful-Avoidant Attachment
        patterns.append(
            AttachmentPattern(
                style=AttachmentStyle.DISORGANIZED_FEARFUL_AVOIDANT,
                name="Disorganized/Fearful-Avoidant Attachment",
                description="Simultaneous desire for and fear of close relationships, chaotic attachment patterns",
                characteristics=[
                    "Simultaneous approach and avoidance",
                    "Fear of intimacy and abandonment",
                    "Chaotic relationship patterns",
                    "Negative view of self and others",
                    "Emotional dysregulation",
                ],
                behavioral_indicators=[
                    "Inconsistent relationship behavior",
                    "Push-pull dynamics",
                    "Emotional volatility",
                    "Self-sabotaging behaviors",
                    "Difficulty trusting others",
                ],
                therapeutic_considerations=[
                    "Complex therapeutic relationship",
                    "May recreate chaotic patterns in therapy",
                    "Needs trauma-informed approaches",
                    "Requires patience and consistency",
                ],
                developmental_origins=[
                    "Traumatic or abusive caregiving",
                    "Caregiver as source of fear and comfort",
                    "Unresolved caregiver trauma",
                    "Chaotic family environment",
                ],
                relationship_patterns=[
                    "Unstable relationship patterns",
                    "Fear of both intimacy and abandonment",
                    "Intense but chaotic connections",
                    "Difficulty maintaining boundaries",
                ],
                emotional_regulation=[
                    "Severe dysregulation",
                    "Dissociation under stress",
                    "Self-harm or destructive behaviors",
                    "Difficulty identifying emotions",
                ],
            )
        )

        return patterns

    def _create_defense_mechanisms(self) -> list[DefenseMechanism]:
        """Create defense mechanism definitions."""
        mechanisms = []

        # Mature Defenses
        mechanisms.extend(
            [
                DefenseMechanism(
                    name="Sublimation",
                    level=DefenseMechanismLevel.MATURE,
                    description="Channeling unacceptable impulses into socially acceptable activities",
                    function="Transform primitive impulses into constructive outlets",
                    examples=[
                        "Channeling aggression into competitive sports",
                        "Transforming sexual energy into creative work",
                        "Using analytical skills to understand personal conflicts",
                    ],
                    adaptive_aspects=[
                        "Promotes personal growth",
                        "Socially beneficial outcomes",
                        "Maintains psychological balance",
                    ],
                    therapeutic_approach=[
                        "Encourage healthy outlets",
                        "Explore underlying impulses",
                        "Support creative expression",
                    ],
                ),
                DefenseMechanism(
                    name="Humor",
                    level=DefenseMechanismLevel.MATURE,
                    description="Using comedy to express thoughts and feelings without discomfort",
                    function="Reduce anxiety and maintain social connections",
                    examples=[
                        "Making light of difficult situations",
                        "Self-deprecating humor about personal flaws",
                        "Using wit to address uncomfortable topics",
                    ],
                    adaptive_aspects=[
                        "Maintains social bonds",
                        "Reduces tension",
                        "Provides perspective on problems",
                    ],
                    therapeutic_approach=[
                        "Appreciate adaptive function",
                        "Explore what lies beneath humor",
                        "Balance humor with serious processing",
                    ],
                ),
            ]
        )

        # Neurotic Defenses
        mechanisms.extend(
            [
                DefenseMechanism(
                    name="Repression",
                    level=DefenseMechanismLevel.NEUROTIC,
                    description="Unconsciously blocking unacceptable thoughts or memories",
                    function="Protect ego from anxiety-provoking material",
                    examples=[
                        "Forgetting traumatic childhood events",
                        "Unable to recall angry feelings toward loved ones",
                        "Blocking awareness of sexual desires",
                    ],
                    adaptive_aspects=[
                        "Protects from overwhelming anxiety",
                        "Allows functioning in daily life",
                    ],
                    maladaptive_aspects=[
                        "Prevents processing of important experiences",
                        "Can lead to symptoms and dysfunction",
                        "Limits self-awareness",
                    ],
                    therapeutic_approach=[
                        "Gentle exploration of blocked material",
                        "Create safety for emergence of repressed content",
                        "Work through underlying conflicts",
                    ],
                ),
                DefenseMechanism(
                    name="Rationalization",
                    level=DefenseMechanismLevel.NEUROTIC,
                    description="Creating logical explanations for unacceptable behavior or feelings",
                    function="Maintain self-esteem and reduce cognitive dissonance",
                    examples=[
                        "Explaining away failures with external factors",
                        "Justifying harmful behavior with good intentions",
                        "Creating elaborate reasons for avoiding challenges",
                    ],
                    adaptive_aspects=[
                        "Maintains psychological stability",
                        "Allows continued functioning",
                    ],
                    maladaptive_aspects=[
                        "Prevents learning from mistakes",
                        "Blocks authentic self-reflection",
                        "Can damage relationships",
                    ],
                    therapeutic_approach=[
                        "Gently challenge rationalizations",
                        "Explore underlying fears and shame",
                        "Encourage honest self-examination",
                    ],
                ),
            ]
        )

        # Immature Defenses
        mechanisms.extend(
            [
                DefenseMechanism(
                    name="Projection",
                    level=DefenseMechanismLevel.IMMATURE,
                    description="Attributing one's own unacceptable thoughts or feelings to others",
                    function="Avoid awareness of uncomfortable aspects of self",
                    examples=[
                        "Accusing others of being angry when you are angry",
                        "Believing others are judging you when you judge yourself",
                        "Seeing others as untrustworthy when feeling untrustworthy",
                    ],
                    adaptive_aspects=["Protects from overwhelming self-criticism"],
                    maladaptive_aspects=[
                        "Damages relationships",
                        "Prevents self-awareness",
                        "Creates interpersonal conflict",
                    ],
                    therapeutic_approach=[
                        "Help recognize projection patterns",
                        "Explore disowned aspects of self",
                        "Develop tolerance for difficult feelings",
                    ],
                ),
                DefenseMechanism(
                    name="Splitting",
                    level=DefenseMechanismLevel.IMMATURE,
                    description="Viewing people or situations as all good or all bad",
                    function="Manage ambivalence and complexity",
                    examples=[
                        "Idealizing then devaluing relationships",
                        "Seeing therapist as perfect or terrible",
                        "Black-and-white thinking about self and others",
                    ],
                    adaptive_aspects=["Simplifies complex emotional situations"],
                    maladaptive_aspects=[
                        "Creates unstable relationships",
                        "Prevents integrated view of self and others",
                        "Leads to emotional volatility",
                    ],
                    therapeutic_approach=[
                        "Help integrate good and bad aspects",
                        "Explore fear of ambivalence",
                        "Develop tolerance for complexity",
                    ],
                ),
            ]
        )

        # Pathological Defenses
        mechanisms.extend(
            [
                DefenseMechanism(
                    name="Denial",
                    level=DefenseMechanismLevel.PATHOLOGICAL,
                    description="Refusing to acknowledge obvious reality or experience",
                    function="Avoid overwhelming anxiety or pain",
                    examples=[
                        "Refusing to acknowledge serious illness",
                        "Denying substance abuse problems",
                        "Ignoring obvious relationship problems",
                    ],
                    adaptive_aspects=["May provide temporary protection from trauma"],
                    maladaptive_aspects=[
                        "Prevents necessary action",
                        "Can be life-threatening",
                        "Blocks reality testing",
                    ],
                    therapeutic_approach=[
                        "Very gentle reality testing",
                        "Build capacity to tolerate difficult truths",
                        "Address underlying trauma or fear",
                    ],
                ),
                DefenseMechanism(
                    name="Dissociation",
                    level=DefenseMechanismLevel.PATHOLOGICAL,
                    description="Disconnection from thoughts, feelings, memories, or sense of identity",
                    function="Escape from overwhelming psychological pain",
                    examples=[
                        "Feeling detached from one's body",
                        "Memory gaps during stressful events",
                        "Feeling like observing oneself from outside",
                    ],
                    adaptive_aspects=["Protects from unbearable trauma"],
                    maladaptive_aspects=[
                        "Interferes with daily functioning",
                        "Prevents integration of experience",
                        "Can be disorienting and frightening",
                    ],
                    therapeutic_approach=[
                        "Trauma-informed treatment",
                        "Grounding and stabilization techniques",
                        "Gradual integration of dissociated material",
                    ],
                ),
            ]
        )

        return mechanisms

    def _create_psychodynamic_patterns(self) -> list[PsychodynamicPattern]:
        """Create psychodynamic personality patterns."""
        patterns = []

        # Attachment-Related Pattern
        patterns.append(
            PsychodynamicPattern(
                name="Anxious-Attachment Pattern",
                domain=PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS,
                description="Chronic fear of abandonment with intense relationship preoccupation",
                core_features=[
                    "Pervasive fear of abandonment",
                    "Intense need for reassurance",
                    "Difficulty tolerating aloneness",
                    "Hypervigilance to relationship threats",
                ],
                unconscious_conflicts=[
                    "Desire for closeness vs. fear of engulfment",
                    "Need for independence vs. fear of abandonment",
                    "Love vs. anger toward attachment figures",
                ],
                object_relations=[
                    "Internal objects are unpredictable",
                    "Self as needy and vulnerable",
                    "Others as potentially abandoning",
                ],
                transference_patterns=[
                    "Intense attachment to therapist",
                    "Fear of therapeutic abandonment",
                    "Testing of therapeutic commitment",
                ],
                countertransference_patterns=[
                    "Feeling overwhelmed by client's needs",
                    "Urge to rescue or reject",
                    "Anxiety about setting boundaries",
                ],
                therapeutic_goals=[
                    "Develop secure internal working models",
                    "Increase tolerance for aloneness",
                    "Build capacity for self-soothing",
                ],
                treatment_considerations=[
                    "Consistent, reliable therapeutic frame",
                    "Gradual exploration of abandonment fears",
                    "Work with transference-countertransference",
                ],
            )
        )

        # Affect Regulation Pattern
        patterns.append(
            PsychodynamicPattern(
                name="Emotional Dysregulation Pattern",
                domain=PsychodynamicDomain.AFFECT_REGULATION,
                description="Difficulty managing and modulating emotional experiences",
                core_features=[
                    "Intense, overwhelming emotions",
                    "Rapid mood fluctuations",
                    "Difficulty identifying emotions",
                    "Impulsive emotional responses",
                ],
                unconscious_conflicts=[
                    "Need for emotional expression vs. fear of losing control",
                    "Desire for emotional connection vs. fear of vulnerability",
                    "Anger vs. guilt about having needs",
                ],
                object_relations=[
                    "Emotions as dangerous and overwhelming",
                    "Self as emotionally chaotic",
                    "Others as unable to contain emotions",
                ],
                transference_patterns=[
                    "Emotional storms in therapy",
                    "Testing therapist's capacity to contain",
                    "Fear of overwhelming therapist",
                ],
                countertransference_patterns=[
                    "Feeling emotionally flooded",
                    "Urge to calm or control client",
                    "Anxiety about emotional intensity",
                ],
                therapeutic_goals=[
                    "Develop emotional awareness and vocabulary",
                    "Build capacity for affect tolerance",
                    "Learn healthy emotional expression",
                ],
                treatment_considerations=[
                    "Containment and stabilization focus",
                    "Gradual affect tolerance building",
                    "Psychoeducation about emotions",
                ],
            )
        )

        # Identity Pattern
        patterns.append(
            PsychodynamicPattern(
                name="Identity Diffusion Pattern",
                domain=PsychodynamicDomain.IDENTITY_SELF_CONCEPT,
                description="Unstable sense of self with shifting identity and values",
                core_features=[
                    "Unclear sense of identity",
                    "Shifting values and goals",
                    "Dependence on others for self-definition",
                    "Chronic feelings of emptiness",
                ],
                unconscious_conflicts=[
                    "Desire for authentic self vs. need for approval",
                    "Independence vs. merger with others",
                    "Grandiosity vs. worthlessness",
                ],
                object_relations=[
                    "Self as fragmented and unclear",
                    "Others as sources of identity",
                    "Relationships as identity-defining",
                ],
                transference_patterns=[
                    "Adopting therapist's values and perspectives",
                    "Seeking therapist's definition of self",
                    "Identity shifts based on therapeutic relationship",
                ],
                countertransference_patterns=[
                    "Feeling responsible for client's identity",
                    "Confusion about client's authentic self",
                    "Urge to provide structure and definition",
                ],
                therapeutic_goals=[
                    "Develop coherent sense of identity",
                    "Increase self-awareness and authenticity",
                    "Build capacity for self-definition",
                ],
                treatment_considerations=[
                    "Long-term therapy often needed",
                    "Focus on self-exploration and discovery",
                    "Work with identity-related transference",
                ],
            )
        )

        return patterns

    def _build_theoretical_frameworks(self) -> dict[str, list[str]]:
        """Build theoretical frameworks mapping."""
        return {
            "psychoanalytic": [
                "Drive theory",
                "Ego psychology",
                "Object relations theory",
                "Self psychology",
            ],
            "attachment_theory": [
                "Bowlby's attachment theory",
                "Ainsworth's attachment styles",
                "Adult attachment theory",
                "Mentalization-based treatment",
            ],
            "developmental": [
                "Erikson's psychosocial stages",
                "Mahler's separation-individuation",
                "Winnicott's transitional phenomena",
                "Stern's sense of self",
            ],
        }

    def _build_developmental_considerations(self) -> dict[str, list[str]]:
        """Build developmental considerations mapping."""
        return {
            "early_childhood": [
                "Attachment formation",
                "Basic trust vs. mistrust",
                "Separation-individuation",
                "Language and symbolic development",
            ],
            "adolescence": [
                "Identity formation",
                "Peer relationships",
                "Sexual development",
                "Autonomy struggles",
            ],
            "adulthood": [
                "Intimate relationships",
                "Career development",
                "Generativity",
                "Midlife transitions",
            ],
        }

    def get_attachment_patterns(self) -> list[AttachmentPattern]:
        """Get all attachment patterns."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.attachment_patterns

    def get_attachment_pattern_by_style(self, style: AttachmentStyle) -> AttachmentPattern | None:
        """Get attachment pattern by style."""
        if not self.knowledge_base:
            return None

        for pattern in self.knowledge_base.attachment_patterns:
            if pattern.style == style:
                return pattern
        return None

    def get_defense_mechanisms(self) -> list[DefenseMechanism]:
        """Get all defense mechanisms."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.defense_mechanisms

    def get_defense_mechanisms_by_level(
        self, level: DefenseMechanismLevel
    ) -> list[DefenseMechanism]:
        """Get defense mechanisms by maturity level."""
        if not self.knowledge_base:
            return []

        return [dm for dm in self.knowledge_base.defense_mechanisms if dm.level == level]

    def get_defense_mechanism_by_name(self, name: str) -> DefenseMechanism | None:
        """Get defense mechanism by name."""
        if not self.knowledge_base:
            return None

        for mechanism in self.knowledge_base.defense_mechanisms:
            if mechanism.name.lower() == name.lower():
                return mechanism
        return None

    def get_psychodynamic_patterns(self) -> list[PsychodynamicPattern]:
        """Get all psychodynamic patterns."""
        if not self.knowledge_base:
            return []
        return self.knowledge_base.psychodynamic_patterns

    def get_psychodynamic_patterns_by_domain(
        self, domain: PsychodynamicDomain
    ) -> list[PsychodynamicPattern]:
        """Get psychodynamic patterns by domain."""
        if not self.knowledge_base:
            return []

        return [pp for pp in self.knowledge_base.psychodynamic_patterns if pp.domain == domain]

    def get_psychodynamic_pattern_by_name(self, name: str) -> PsychodynamicPattern | None:
        """Get psychodynamic pattern by name."""
        if not self.knowledge_base:
            return None

        for pattern in self.knowledge_base.psychodynamic_patterns:
            if pattern.name.lower() == name.lower():
                return pattern
        return None

    def create_sample_concepts(self) -> list[dict[str, Any]]:
        """Create sample psychodynamic concepts for demonstration and testing."""
        if not self.knowledge_base:
            return []

        sample_data = []

        # Add attachment patterns
        for pattern in self.knowledge_base.attachment_patterns:
            pattern_dict = asdict(pattern)
            pattern_dict["style"] = pattern.style.value
            pattern_dict["type"] = "attachment_pattern"
            sample_data.append(pattern_dict)

        # Add defense mechanisms
        for mechanism in self.knowledge_base.defense_mechanisms:
            mechanism_dict = asdict(mechanism)
            mechanism_dict["level"] = mechanism.level.value
            mechanism_dict["type"] = "defense_mechanism"
            sample_data.append(mechanism_dict)

        # Add psychodynamic patterns
        for pattern in self.knowledge_base.psychodynamic_patterns:
            pattern_dict = asdict(pattern)
            pattern_dict["domain"] = pattern.domain.value
            pattern_dict["type"] = "psychodynamic_pattern"
            sample_data.append(pattern_dict)

        logger.info(f"Created {len(sample_data)} sample psychodynamic concepts")
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
                "attachment_patterns": [],
                "defense_mechanisms": [],
                "psychodynamic_patterns": [],
                "theoretical_frameworks": self.knowledge_base.theoretical_frameworks,
                "developmental_considerations": self.knowledge_base.developmental_considerations,
            }

            # Convert attachment patterns
            for pattern in self.knowledge_base.attachment_patterns:
                pattern_dict = asdict(pattern)
                pattern_dict["style"] = pattern.style.value
                export_data["attachment_patterns"].append(pattern_dict)

            # Convert defense mechanisms
            for mechanism in self.knowledge_base.defense_mechanisms:
                mechanism_dict = asdict(mechanism)
                mechanism_dict["level"] = mechanism.level.value
                export_data["defense_mechanisms"].append(mechanism_dict)

            # Convert psychodynamic patterns
            for pattern in self.knowledge_base.psychodynamic_patterns:
                pattern_dict = asdict(pattern)
                pattern_dict["domain"] = pattern.domain.value
                export_data["psychodynamic_patterns"].append(pattern_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported PDM-2 knowledge base to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export PDM-2 knowledge base: {e}")
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
            attachment_patterns = []
            for pattern_data in data.get("attachment_patterns", []):
                pattern_data["style"] = AttachmentStyle(pattern_data["style"])
                attachment_patterns.append(AttachmentPattern(**pattern_data))

            defense_mechanisms = []
            for mechanism_data in data.get("defense_mechanisms", []):
                mechanism_data["level"] = DefenseMechanismLevel(mechanism_data["level"])
                defense_mechanisms.append(DefenseMechanism(**mechanism_data))

            psychodynamic_patterns = []
            for pattern_data in data.get("psychodynamic_patterns", []):
                pattern_data["domain"] = PsychodynamicDomain(pattern_data["domain"])
                psychodynamic_patterns.append(PsychodynamicPattern(**pattern_data))

            self.knowledge_base = PDM2KnowledgeBase(
                attachment_patterns=attachment_patterns,
                defense_mechanisms=defense_mechanisms,
                psychodynamic_patterns=psychodynamic_patterns,
                theoretical_frameworks=data.get("theoretical_frameworks", {}),
                developmental_considerations=data.get("developmental_considerations", {}),
                version=data.get("version", "Unknown"),
                created_at=data.get("created_at"),
            )

            logger.info(f"Loaded PDM-2 knowledge base from {input_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load PDM-2 knowledge base: {e}")
            return False

    def generate_conversation_templates(
        self, concept_type: str, concept_name: str
    ) -> list[Conversation]:
        """Generate conversation templates for psychodynamic assessment."""
        conversations = []

        if concept_type == "attachment":
            pattern = self.get_attachment_pattern_by_style(AttachmentStyle(concept_name))
            if pattern:
                conversations.extend(self._generate_attachment_conversations(pattern))
        elif concept_type == "defense":
            mechanism = self.get_defense_mechanism_by_name(concept_name)
            if mechanism:
                conversations.extend(self._generate_defense_conversations(mechanism))
        elif concept_type == "pattern":
            pattern = self.get_psychodynamic_pattern_by_name(concept_name)
            if pattern:
                conversations.extend(self._generate_pattern_conversations(pattern))

        logger.info(
            f"Generated {len(conversations)} conversation templates for {concept_type}: {concept_name}"
        )
        return conversations

    def _generate_attachment_conversations(self, pattern: AttachmentPattern) -> list[Conversation]:
        """Generate conversations for attachment pattern assessment."""
        messages = [
            Message(
                role="therapist",
                content=f"I'd like to explore your relationship patterns and attachment style. This relates to {pattern.name}.",
                meta={"type": "introduction", "attachment_style": pattern.style.value},
            )
        ]

        # Add questions about attachment characteristics
        for characteristic in pattern.characteristics[:3]:
            messages.append(
                Message(
                    role="therapist",
                    content=f"How do you experience: {characteristic.lower()}?",
                    meta={
                        "category": "attachment_characteristic",
                        "attachment_style": pattern.style.value,
                    },
                )
            )

            # Sample client response
            if pattern.behavioral_indicators:
                indicator = pattern.behavioral_indicators[0]
                messages.append(
                    Message(
                        role="client",
                        content=f"I notice that I tend to {indicator.lower()}.",
                        meta={"attachment_style": pattern.style.value, "example": True},
                    )
                )

        conversation = Conversation(
            id=f"pdm2_attachment_{pattern.style.value}",
            messages=messages,
            context={
                "attachment_style": pattern.style.value,
                "pattern_name": pattern.name,
                "type": "attachment_assessment",
            },
            source="pdm2_parser",
            meta={
                "characteristics_count": len(pattern.characteristics),
                "behavioral_indicators_count": len(pattern.behavioral_indicators),
            },
        )

        return [conversation]

    def _generate_defense_conversations(self, mechanism: DefenseMechanism) -> list[Conversation]:
        """Generate conversations for defense mechanism exploration."""
        messages = [
            Message(
                role="therapist",
                content=f"Let's explore how you cope with difficult situations. This relates to {mechanism.name}.",
                meta={"type": "introduction", "defense_mechanism": mechanism.name},
            ),
            Message(
                role="therapist",
                content=f"When you're stressed, do you notice: {mechanism.description.lower()}?",
                meta={"category": "defense_exploration", "level": mechanism.level.value},
            ),
        ]

        # Add example exploration
        if mechanism.examples:
            example = mechanism.examples[0]
            messages.append(
                Message(
                    role="client",
                    content=f"Yes, I find myself {example.lower()}.",
                    meta={"defense_mechanism": mechanism.name, "example": True},
                )
            )

        conversation = Conversation(
            id=f"pdm2_defense_{mechanism.name.lower().replace(' ', '_')}",
            messages=messages,
            context={
                "defense_mechanism": mechanism.name,
                "level": mechanism.level.value,
                "type": "defense_assessment",
            },
            source="pdm2_parser",
            meta={"function": mechanism.function, "examples_count": len(mechanism.examples)},
        )

        return [conversation]

    def _generate_pattern_conversations(self, pattern: PsychodynamicPattern) -> list[Conversation]:
        """Generate conversations for psychodynamic pattern exploration."""
        messages = [
            Message(
                role="therapist",
                content=f"I'd like to understand your inner world better. This relates to {pattern.name}.",
                meta={"type": "introduction", "pattern": pattern.name},
            )
        ]

        # Add questions about core features
        for feature in pattern.core_features[:2]:
            messages.append(
                Message(
                    role="therapist",
                    content=f"Do you experience: {feature.lower()}?",
                    meta={"category": "core_feature", "domain": pattern.domain.value},
                )
            )

        conversation = Conversation(
            id=f"pdm2_pattern_{pattern.name.lower().replace(' ', '_').replace('-', '_')}",
            messages=messages,
            context={
                "pattern_name": pattern.name,
                "domain": pattern.domain.value,
                "type": "psychodynamic_assessment",
            },
            source="pdm2_parser",
            meta={
                "core_features_count": len(pattern.core_features),
                "unconscious_conflicts_count": len(pattern.unconscious_conflicts),
            },
        )

        return [conversation]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.knowledge_base:
            return {}

        stats = {
            "total_attachment_patterns": len(self.knowledge_base.attachment_patterns),
            "total_defense_mechanisms": len(self.knowledge_base.defense_mechanisms),
            "total_psychodynamic_patterns": len(self.knowledge_base.psychodynamic_patterns),
            "defense_mechanisms_by_level": {},
            "psychodynamic_patterns_by_domain": {},
            "attachment_styles": {},
            "version": self.knowledge_base.version,
        }

        # Defense mechanisms by level
        for mechanism in self.knowledge_base.defense_mechanisms:
            level = mechanism.level.value
            if level not in stats["defense_mechanisms_by_level"]:
                stats["defense_mechanisms_by_level"][level] = 0
            stats["defense_mechanisms_by_level"][level] += 1

        # Psychodynamic patterns by domain
        for pattern in self.knowledge_base.psychodynamic_patterns:
            domain = pattern.domain.value
            if domain not in stats["psychodynamic_patterns_by_domain"]:
                stats["psychodynamic_patterns_by_domain"][domain] = 0
            stats["psychodynamic_patterns_by_domain"][domain] += 1

        # Attachment styles
        for pattern in self.knowledge_base.attachment_patterns:
            style = pattern.style.value
            stats["attachment_styles"][style] = pattern.name

        return stats
