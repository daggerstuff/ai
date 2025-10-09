"""
Therapeutic Alliance and Rapport-Building Conversations System

This module provides comprehensive therapeutic alliance and rapport-building conversation
generation for psychology knowledge-based training. It focuses on the foundational
therapeutic relationship components that underpin successful therapy outcomes.

Key Features:
- Therapeutic alliance components (task, bond, goals)
- Rapport-building techniques and strategies
- Alliance rupture detection and repair processes
- Cultural considerations in alliance building
- Stage-specific alliance development
- Alliance strength assessment and tracking
- Population-specific alliance strategies
- Alliance conversation generation
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .client_scenario_generator import ClientScenario


class AllianceComponent(Enum):
    """Core components of therapeutic alliance (Bordin's model)."""
    TASK_ALLIANCE = "task_alliance"  # Agreement on therapeutic tasks and activities
    BOND_ALLIANCE = "bond_alliance"  # Emotional connection and trust
    GOAL_ALLIANCE = "goal_alliance"  # Shared understanding of therapy goals


class AllianceStage(Enum):
    """Stages of therapeutic alliance development."""
    INITIAL_ENGAGEMENT = "initial_engagement"  # First sessions, trust building
    WORKING_ALLIANCE = "working_alliance"  # Established therapeutic relationship
    ALLIANCE_MAINTENANCE = "alliance_maintenance"  # Ongoing relationship work
    ALLIANCE_REPAIR = "alliance_repair"  # Addressing ruptures and strains
    TERMINATION_ALLIANCE = "termination_alliance"  # Ending phase considerations


class RapportTechnique(Enum):
    """Specific rapport-building techniques."""
    EMPATHIC_REFLECTION = "empathic_reflection"
    ACTIVE_LISTENING = "active_listening"
    GENUINENESS = "genuineness"
    UNCONDITIONAL_POSITIVE_REGARD = "unconditional_positive_regard"
    CULTURAL_RESPONSIVENESS = "cultural_responsiveness"
    NONVERBAL_ATTUNEMENT = "nonverbal_attunement"
    COMMUNICATION_MATCHING = "communication_matching"
    VALIDATION = "validation"
    COLLABORATIVE_APPROACH = "collaborative_approach"
    TRANSPARENCY = "transparency"


class AllianceRuptureType(Enum):
    """Types of therapeutic alliance ruptures."""
    WITHDRAWAL_RUPTURE = "withdrawal_rupture"  # Client becomes distant or disengaged
    CONFRONTATION_RUPTURE = "confrontation_rupture"  # Direct conflict or disagreement
    TASK_DISAGREEMENT = "task_disagreement"  # Disagreement about therapeutic activities
    GOAL_MISALIGNMENT = "goal_misalignment"  # Different views on therapy goals
    CULTURAL_MISUNDERSTANDING = "cultural_misunderstanding"  # Cultural factors affecting alliance
    TRANSFERENCE_RUPTURE = "transference_rupture"  # Past relationship patterns affecting alliance


class AllianceStrength(Enum):
    """Levels of therapeutic alliance strength."""
    WEAK = "weak"  # Poor alliance, significant challenges
    DEVELOPING = "developing"  # Alliance forming but fragile
    MODERATE = "moderate"  # Solid working alliance
    STRONG = "strong"  # Excellent therapeutic relationship
    EXCEPTIONAL = "exceptional"  # Outstanding alliance with deep trust


@dataclass
class AllianceAssessment:
    """Assessment of therapeutic alliance strength and components."""
    id: str
    overall_strength: AllianceStrength
    task_alliance_score: float  # 0-10 scale
    bond_alliance_score: float  # 0-10 scale
    goal_alliance_score: float  # 0-10 scale
    rapport_indicators: list[str]
    rupture_indicators: list[str]
    cultural_factors: list[str]
    improvement_areas: list[str]
    strengths: list[str]
    assessment_date: str


@dataclass
class AllianceStrategy:
    """Strategy for building or repairing therapeutic alliance."""
    id: str
    alliance_component: AllianceComponent
    target_stage: AllianceStage
    rapport_techniques: list[RapportTechnique]
    cultural_adaptations: list[str]
    specific_interventions: list[str]
    expected_outcomes: list[str]
    success_indicators: list[str]
    potential_challenges: list[str]
    monitoring_plan: list[str]


@dataclass
class AllianceConversation:
    """Therapeutic alliance-focused conversation with rapport-building elements."""
    id: str
    alliance_strategy: AllianceStrategy
    client_scenario: ClientScenario
    conversation_exchanges: list[dict[str, Any]]
    alliance_focus: AllianceComponent
    stage: AllianceStage
    rapport_techniques_used: list[RapportTechnique]
    cultural_adaptations_applied: list[str]
    alliance_building_moments: list[str]
    rupture_repair_elements: list[str]
    nonverbal_considerations: list[str]
    alliance_assessment: AllianceAssessment
    therapeutic_gains: list[str]
    next_session_focus: str


class TherapeuticAllianceGenerator:
    """
    Comprehensive therapeutic alliance and rapport-building conversation system.

    This class provides generation of alliance-focused therapeutic conversations,
    rapport-building strategies, and alliance rupture repair processes.
    """

    def __init__(self):
        """Initialize the therapeutic alliance generator."""
        self.alliance_strategies = self._initialize_alliance_strategies()
        self.rapport_frameworks = self._initialize_rapport_frameworks()
        self.rupture_repair_protocols = self._initialize_rupture_repair_protocols()
        self.cultural_alliance_considerations = self._initialize_cultural_considerations()

    def _initialize_alliance_strategies(self) -> dict[AllianceComponent, list[AllianceStrategy]]:
        """Initialize alliance-building strategies for each component."""
        strategies = {}

        # Task Alliance Strategies
        strategies[AllianceComponent.TASK_ALLIANCE] = [
            AllianceStrategy(
                id="collaborative_task_setting",
                alliance_component=AllianceComponent.TASK_ALLIANCE,
                target_stage=AllianceStage.INITIAL_ENGAGEMENT,
                rapport_techniques=[
                    RapportTechnique.COLLABORATIVE_APPROACH,
                    RapportTechnique.TRANSPARENCY,
                    RapportTechnique.VALIDATION
                ],
                cultural_adaptations=[
                    "Respect for cultural decision-making processes",
                    "Family involvement when culturally appropriate",
                    "Adaptation to communication styles"
                ],
                specific_interventions=[
                    "Explain therapy process clearly",
                    "Invite client input on therapeutic activities",
                    "Negotiate homework and between-session tasks",
                    "Check for understanding and agreement"
                ],
                expected_outcomes=[
                    "Clear understanding of therapy process",
                    "Mutual agreement on therapeutic activities",
                    "Increased client engagement in tasks",
                    "Reduced resistance to interventions"
                ],
                success_indicators=[
                    "Client asks clarifying questions",
                    "Client suggests modifications to tasks",
                    "Completion of between-session activities",
                    "Positive feedback about therapy process"
                ],
                potential_challenges=[
                    "Cultural differences in authority relationships",
                    "Previous negative therapy experiences",
                    "Ambivalence about change",
                    "Cognitive or emotional barriers"
                ],
                monitoring_plan=[
                    "Regular check-ins about task relevance",
                    "Feedback on task difficulty and usefulness",
                    "Adjustment of tasks based on client response",
                    "Documentation of task completion patterns"
                ]
            )
        ]

        # Bond Alliance Strategies
        strategies[AllianceComponent.BOND_ALLIANCE] = [
            AllianceStrategy(
                id="empathic_connection_building",
                alliance_component=AllianceComponent.BOND_ALLIANCE,
                target_stage=AllianceStage.INITIAL_ENGAGEMENT,
                rapport_techniques=[
                    RapportTechnique.EMPATHIC_REFLECTION,
                    RapportTechnique.ACTIVE_LISTENING,
                    RapportTechnique.GENUINENESS,
                    RapportTechnique.UNCONDITIONAL_POSITIVE_REGARD
                ],
                cultural_adaptations=[
                    "Culturally appropriate expressions of empathy",
                    "Respect for emotional expression norms",
                    "Adaptation to relationship building styles",
                    "Awareness of power dynamics"
                ],
                specific_interventions=[
                    "Reflect client emotions accurately",
                    "Share appropriate self-disclosure",
                    "Demonstrate genuine care and concern",
                    "Maintain consistent warmth and acceptance"
                ],
                expected_outcomes=[
                    "Client feels understood and accepted",
                    "Increased emotional openness",
                    "Development of trust and safety",
                    "Positive therapeutic relationship"
                ],
                success_indicators=[
                    "Client shares personal information",
                    "Emotional expression increases",
                    "Client seeks therapist's opinion",
                    "Positive comments about therapist"
                ],
                potential_challenges=[
                    "Cultural barriers to emotional expression",
                    "Trust issues from past trauma",
                    "Therapist-client demographic differences",
                    "Attachment style influences"
                ],
                monitoring_plan=[
                    "Observe client comfort level",
                    "Monitor emotional expression patterns",
                    "Check for signs of trust development",
                    "Assess client feedback about relationship"
                ]
            )
        ]

        # Goal Alliance Strategies
        strategies[AllianceComponent.GOAL_ALLIANCE] = [
            AllianceStrategy(
                id="collaborative_goal_setting",
                alliance_component=AllianceComponent.GOAL_ALLIANCE,
                target_stage=AllianceStage.INITIAL_ENGAGEMENT,
                rapport_techniques=[
                    RapportTechnique.COLLABORATIVE_APPROACH,
                    RapportTechnique.VALIDATION,
                    RapportTechnique.ACTIVE_LISTENING
                ],
                cultural_adaptations=[
                    "Respect for cultural values in goal setting",
                    "Family and community considerations",
                    "Culturally relevant outcome measures",
                    "Adaptation to time orientation"
                ],
                specific_interventions=[
                    "Explore client's hopes and expectations",
                    "Identify meaningful and achievable goals",
                    "Negotiate timeline and milestones",
                    "Regularly review and adjust goals"
                ],
                expected_outcomes=[
                    "Clear, mutually agreed-upon goals",
                    "Client ownership of therapeutic objectives",
                    "Realistic expectations about outcomes",
                    "Motivation for therapeutic work"
                ],
                success_indicators=[
                    "Client articulates personal goals",
                    "Goals are specific and measurable",
                    "Client expresses commitment to goals",
                    "Regular progress toward objectives"
                ],
                potential_challenges=[
                    "Unrealistic expectations",
                    "Conflicting goals (client vs. others)",
                    "Ambivalence about change",
                    "Cultural differences in goal orientation"
                ],
                monitoring_plan=[
                    "Regular goal review sessions",
                    "Progress tracking and measurement",
                    "Goal modification as needed",
                    "Celebration of achievements"
                ]
            )
        ]

        return strategies

    def _initialize_rapport_frameworks(self) -> dict[RapportTechnique, dict[str, Any]]:
        """Initialize rapport-building frameworks and techniques."""
        return {
            RapportTechnique.EMPATHIC_REFLECTION: {
                "description": "Accurately reflecting client's emotional experience",
                "key_elements": [
                    "Emotional labeling",
                    "Feeling validation",
                    "Perspective taking",
                    "Emotional attunement"
                ],
                "verbal_techniques": [
                    "It sounds like you're feeling...",
                    "I can hear the [emotion] in your voice",
                    "That must have been [emotion] for you",
                    "I'm sensing that you feel..."
                ],
                "nonverbal_elements": [
                    "Facial expression matching",
                    "Appropriate eye contact",
                    "Open body posture",
                    "Vocal tone matching"
                ],
                "cultural_considerations": [
                    "Cultural norms for emotional expression",
                    "Eye contact appropriateness",
                    "Physical space preferences",
                    "Emotional vocabulary differences"
                ]
            },
            RapportTechnique.ACTIVE_LISTENING: {
                "description": "Fully attending to and engaging with client communication",
                "key_elements": [
                    "Full attention",
                    "Minimal encouragers",
                    "Paraphrasing",
                    "Summarizing"
                ],
                "verbal_techniques": [
                    "Mm-hmm", "I see", "Go on",
                    "What I'm hearing is...",
                    "Let me make sure I understand...",
                    "So you're saying..."
                ],
                "nonverbal_elements": [
                    "Leaning forward",
                    "Nodding appropriately",
                    "Maintaining eye contact",
                    "Eliminating distractions"
                ],
                "cultural_considerations": [
                    "Silence comfort levels",
                    "Turn-taking patterns",
                    "Storytelling traditions",
                    "Indirect communication styles"
                ]
            },
            RapportTechnique.GENUINENESS: {
                "description": "Being authentic and real in the therapeutic relationship",
                "key_elements": [
                    "Authenticity",
                    "Congruence",
                    "Appropriate self-disclosure",
                    "Honest communication"
                ],
                "verbal_techniques": [
                    "I notice I'm feeling...",
                    "I want to be honest with you...",
                    "My sense is...",
                    "I'm wondering if..."
                ],
                "nonverbal_elements": [
                    "Natural expressions",
                    "Relaxed posture",
                    "Consistent verbal/nonverbal",
                    "Spontaneous responses"
                ],
                "cultural_considerations": [
                    "Professional boundary expectations",
                    "Authority relationship norms",
                    "Self-disclosure appropriateness",
                    "Formality preferences"
                ]
            },
            RapportTechnique.CULTURAL_RESPONSIVENESS: {
                "description": "Adapting therapeutic approach to client's cultural background",
                "key_elements": [
                    "Cultural awareness",
                    "Adaptation flexibility",
                    "Respectful inquiry",
                    "Cultural humility"
                ],
                "verbal_techniques": [
                    "Help me understand your cultural perspective...",
                    "In your culture, how is this typically handled?",
                    "I want to be respectful of your background...",
                    "Please correct me if I misunderstand..."
                ],
                "nonverbal_elements": [
                    "Culturally appropriate distance",
                    "Respectful gestures",
                    "Appropriate formality level",
                    "Cultural greeting styles"
                ],
                "cultural_considerations": [
                    "Power distance preferences",
                    "Collectivist vs individualist values",
                    "Religious/spiritual considerations",
                    "Language and communication styles"
                ]
            }
        }

    def _initialize_rupture_repair_protocols(self) -> dict[AllianceRuptureType, dict[str, Any]]:
        """Initialize alliance rupture repair protocols."""
        return {
            AllianceRuptureType.WITHDRAWAL_RUPTURE: {
                "description": "Client becomes distant, disengaged, or withdrawn",
                "warning_signs": [
                    "Decreased verbal participation",
                    "Minimal eye contact",
                    "Short responses",
                    "Missed appointments",
                    "Emotional flatness"
                ],
                "repair_strategies": [
                    "Gentle exploration of distance",
                    "Validation of client experience",
                    "Therapist self-reflection",
                    "Process focus on relationship"
                ],
                "repair_interventions": [
                    "I notice you seem quieter today...",
                    "I'm wondering if something has shifted between us...",
                    "It feels like you might be pulling back...",
                    "Help me understand what's happening for you..."
                ],
                "prevention_strategies": [
                    "Regular alliance check-ins",
                    "Attention to nonverbal cues",
                    "Cultural sensitivity",
                    "Pacing awareness"
                ]
            },
            AllianceRuptureType.CONFRONTATION_RUPTURE: {
                "description": "Direct conflict or disagreement between therapist and client",
                "warning_signs": [
                    "Direct criticism of therapist",
                    "Challenging therapeutic approach",
                    "Angry or hostile tone",
                    "Questioning therapist competence",
                    "Threatening to quit therapy"
                ],
                "repair_strategies": [
                    "Non-defensive listening",
                    "Validation of client concerns",
                    "Collaborative problem-solving",
                    "Therapist accountability"
                ],
                "repair_interventions": [
                    "I can hear your frustration...",
                    "You're right to bring this up...",
                    "Help me understand your concerns...",
                    "What would be more helpful for you?"
                ],
                "prevention_strategies": [
                    "Regular feedback solicitation",
                    "Collaborative approach",
                    "Cultural sensitivity",
                    "Flexibility in approach"
                ]
            },
            AllianceRuptureType.CULTURAL_MISUNDERSTANDING: {
                "description": "Cultural factors creating alliance strain or misunderstanding",
                "warning_signs": [
                    "Cultural references missed",
                    "Inappropriate interventions",
                    "Value conflicts",
                    "Communication style mismatches",
                    "Family/community concerns"
                ],
                "repair_strategies": [
                    "Cultural humility and learning",
                    "Acknowledgment of mistakes",
                    "Cultural consultation",
                    "Approach modification"
                ],
                "repair_interventions": [
                    "I realize I may have misunderstood...",
                    "Help me learn about your cultural perspective...",
                    "I want to be more culturally responsive...",
                    "How can I better honor your background?"
                ],
                "prevention_strategies": [
                    "Cultural assessment",
                    "Ongoing cultural learning",
                    "Cultural consultation",
                    "Flexible approaches"
                ]
            }
        }

    def _initialize_cultural_considerations(self) -> dict[str, Any]:
        """Initialize cultural considerations for alliance building."""
        return {
            "communication_styles": {
                "direct_vs_indirect": [
                    "Adapt questioning style",
                    "Respect indirect communication",
                    "Allow for storytelling",
                    "Read between the lines"
                ],
                "high_context_vs_low_context": [
                    "Attend to nonverbal cues",
                    "Understand implicit messages",
                    "Respect silence and pauses",
                    "Consider relationship context"
                ]
            },
            "relationship_patterns": {
                "hierarchical_vs_egalitarian": [
                    "Respect authority expectations",
                    "Adapt formality level",
                    "Consider power dynamics",
                    "Honor cultural roles"
                ],
                "individual_vs_collective": [
                    "Include family considerations",
                    "Respect group decision-making",
                    "Consider community impact",
                    "Balance individual and group needs"
                ]
            },
            "trust_building": {
                "time_orientation": [
                    "Respect different pacing",
                    "Allow for relationship building",
                    "Consider cultural time concepts",
                    "Adapt session structure"
                ],
                "disclosure_patterns": [
                    "Respect privacy norms",
                    "Understand shame/honor concepts",
                    "Adapt self-disclosure",
                    "Honor family loyalty"
                ]
            }
        }

    def generate_alliance_conversation(
        self,
        client_scenario: ClientScenario,
        alliance_focus: AllianceComponent,
        stage: AllianceStage = AllianceStage.INITIAL_ENGAGEMENT,
        num_exchanges: int = 10
    ) -> AllianceConversation:
        """
        Generate therapeutic alliance-focused conversation.

        Args:
            client_scenario: Client scenario for alliance building
            alliance_focus: Primary alliance component to focus on
            stage: Stage of alliance development
            num_exchanges: Number of conversation exchanges

        Returns:
            AllianceConversation with rapport-building dialogue
        """
        # Select appropriate strategy
        strategies = self.alliance_strategies.get(alliance_focus, [])
        strategy = strategies[0] if strategies else self._create_generic_strategy(alliance_focus, stage)

        # Generate conversation exchanges
        exchanges = self._generate_alliance_exchanges(
            client_scenario, strategy, num_exchanges
        )

        # Identify rapport techniques used
        rapport_techniques = self._identify_rapport_techniques_used(exchanges)

        # Identify cultural adaptations applied
        cultural_adaptations = self._identify_cultural_adaptations_applied(exchanges, client_scenario)

        # Identify alliance-building moments
        alliance_moments = self._identify_alliance_building_moments(exchanges)

        # Identify rupture repair elements
        rupture_repair = self._identify_rupture_repair_elements(exchanges)

        # Consider nonverbal elements
        nonverbal_considerations = self._generate_nonverbal_considerations(exchanges)

        # Assess alliance strength
        alliance_assessment = self._assess_alliance_strength(exchanges, strategy, client_scenario)

        # Identify therapeutic gains
        therapeutic_gains = self._identify_therapeutic_gains(exchanges, alliance_assessment)

        # Plan next session focus
        next_session_focus = self._plan_next_session_focus(alliance_assessment, strategy)

        conversation_id = f"alliance_{alliance_focus.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return AllianceConversation(
            id=conversation_id,
            alliance_strategy=strategy,
            client_scenario=client_scenario,
            conversation_exchanges=exchanges,
            alliance_focus=alliance_focus,
            stage=stage,
            rapport_techniques_used=rapport_techniques,
            cultural_adaptations_applied=cultural_adaptations,
            alliance_building_moments=alliance_moments,
            rupture_repair_elements=rupture_repair,
            nonverbal_considerations=nonverbal_considerations,
            alliance_assessment=alliance_assessment,
            therapeutic_gains=therapeutic_gains,
            next_session_focus=next_session_focus
        )

    def _create_generic_strategy(self, alliance_focus: AllianceComponent, stage: AllianceStage) -> AllianceStrategy:
        """Create generic alliance strategy for unspecified focus areas."""
        return AllianceStrategy(
            id=f"generic_{alliance_focus.value}_{stage.value}",
            alliance_component=alliance_focus,
            target_stage=stage,
            rapport_techniques=[
                RapportTechnique.EMPATHIC_REFLECTION,
                RapportTechnique.ACTIVE_LISTENING,
                RapportTechnique.VALIDATION
            ],
            cultural_adaptations=["Cultural responsiveness", "Respectful approach"],
            specific_interventions=["Build rapport", "Establish trust", "Collaborate"],
            expected_outcomes=["Improved alliance", "Better engagement"],
            success_indicators=["Client participation", "Positive feedback"],
            potential_challenges=["Resistance", "Cultural barriers"],
            monitoring_plan=["Regular assessment", "Feedback collection"]
        )

    def _generate_alliance_exchanges(
        self,
        client_scenario: ClientScenario,
        strategy: AllianceStrategy,
        num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate alliance-focused conversation exchanges."""
        exchanges = []

        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_alliance_response(
                    client_scenario, strategy, i // 2
                )
            else:  # Client turn
                exchange = self._generate_client_alliance_response(
                    client_scenario, strategy, i // 2
                )

            exchanges.append(exchange)

        return exchanges

    def _generate_therapist_alliance_response(
        self,
        client_scenario: ClientScenario,
        strategy: AllianceStrategy,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate therapist response focused on alliance building."""

        # Select appropriate rapport technique
        if exchange_index < len(strategy.rapport_techniques):
            technique = strategy.rapport_techniques[exchange_index]
        else:
            technique = random.choice(strategy.rapport_techniques)

        # Generate content based on alliance component and technique
        content = self._generate_alliance_content(
            strategy.alliance_component, technique, exchange_index
        )

        return {
            "speaker": "therapist",
            "content": content,
            "rapport_technique": technique.value,
            "alliance_component": strategy.alliance_component.value,
            "alliance_building_element": self._determine_alliance_building_element(technique),
            "cultural_adaptation": self._determine_cultural_adaptation_used(client_scenario, technique),
            "nonverbal_focus": self._determine_nonverbal_focus(technique),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_client_alliance_response(
        self,
        client_scenario: ClientScenario,
        strategy: AllianceStrategy,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate client response in alliance-building context."""

        # Generate response based on alliance development
        content = self._generate_client_alliance_content(
            client_scenario, strategy, exchange_index
        )

        return {
            "speaker": "client",
            "content": content,
            "engagement_level": self._assess_client_engagement(exchange_index),
            "trust_indicators": self._identify_trust_indicators(content),
            "alliance_response": self._assess_alliance_response(content, strategy.alliance_component),
            "emotional_openness": self._assess_emotional_openness(content, exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_alliance_content(
        self,
        alliance_component: AllianceComponent,
        technique: RapportTechnique,
        exchange_index: int
    ) -> str:
        """Generate alliance-building content based on component and technique."""

        if alliance_component == AllianceComponent.TASK_ALLIANCE:
            if technique == RapportTechnique.COLLABORATIVE_APPROACH:
                return "I'd like us to work together to figure out what would be most helpful for you. What are your thoughts on how we might approach this?"
            if technique == RapportTechnique.TRANSPARENCY:
                return "Let me explain what I'm thinking about our work together and get your input on whether this feels right to you."
            return "I want to make sure the things we do in therapy feel meaningful and helpful to you. How does this approach feel so far?"

        if alliance_component == AllianceComponent.BOND_ALLIANCE:
            if technique == RapportTechnique.EMPATHIC_REFLECTION:
                return "I can really hear the pain in your voice when you talk about this. It sounds like this has been incredibly difficult for you."
            if technique == RapportTechnique.GENUINENESS:
                return "I find myself feeling moved by your courage in sharing this with me. It takes real strength to be this open."
            if technique == RapportTechnique.UNCONDITIONAL_POSITIVE_REGARD:
                return "I want you to know that there's nothing you could share that would change my respect for you as a person."
            return "I'm really glad you felt comfortable enough to share that with me. It helps me understand you better."

        if alliance_component == AllianceComponent.GOAL_ALLIANCE:
            if technique == RapportTechnique.COLLABORATIVE_APPROACH:
                return "What would you most like to see change in your life? Let's think together about what goals would be most meaningful for you."
            if technique == RapportTechnique.VALIDATION:
                return "Those sound like really important goals. I can understand why those changes would make such a difference in your life."
            return "Help me understand what success would look like for you. What would tell you that our work together is making a difference?"

        return "I'm here to support you in whatever way feels most helpful. What do you need from me right now?"

    def _generate_client_alliance_content(
        self,
        client_scenario: ClientScenario,
        strategy: AllianceStrategy,
        exchange_index: int
    ) -> str:
        """Generate client response content for alliance building."""

        # Alliance development progresses over exchanges
        if exchange_index == 0:
            # Initial cautious response
            responses = [
                "I'm not really sure what to expect from this.",
                "I've never done therapy before, so this is all new to me.",
                "I hope this can help, but I'm a little nervous about opening up.",
                "I'm willing to try, but I don't know if I'm doing this right."
            ]
        elif exchange_index == 1:
            # Slightly more open
            responses = [
                "That actually makes me feel a bit more comfortable.",
                "I appreciate you explaining that. It helps to know what to expect.",
                "I'm starting to feel like maybe I can trust you with this.",
                "It's nice to feel like we're working together on this."
            ]
        elif exchange_index == 2:
            # Increasing trust and engagement
            responses = [
                "I feel like you really understand what I'm going through.",
                "This feels different from what I expected. In a good way.",
                "I'm surprised by how comfortable I'm starting to feel here.",
                "You make it feel safe to talk about these things."
            ]
        else:
            # Strong alliance development
            responses = [
                "I really value our work together. It's making a difference.",
                "I feel like I can be completely honest with you.",
                "This relationship feels really important to me.",
                "I trust your guidance and feel supported by you."
            ]

        return random.choice(responses)

    def _determine_alliance_building_element(self, technique: RapportTechnique) -> str:
        """Determine specific alliance-building element being used."""

        elements = {
            RapportTechnique.EMPATHIC_REFLECTION: "emotional_attunement",
            RapportTechnique.ACTIVE_LISTENING: "full_presence",
            RapportTechnique.GENUINENESS: "authentic_connection",
            RapportTechnique.UNCONDITIONAL_POSITIVE_REGARD: "acceptance_and_warmth",
            RapportTechnique.CULTURAL_RESPONSIVENESS: "cultural_attunement",
            RapportTechnique.COLLABORATIVE_APPROACH: "partnership_building",
            RapportTechnique.VALIDATION: "experience_validation",
            RapportTechnique.TRANSPARENCY: "process_clarity"
        }

        return elements.get(technique, "rapport_building")

    def _determine_cultural_adaptation_used(
        self,
        client_scenario: ClientScenario,
        technique: RapportTechnique
    ) -> str:
        """Determine cultural adaptation being used."""

        # Consider client demographics for cultural adaptations
        demographics = client_scenario.demographics

        if "ethnicity" in demographics:
            return "culturally_responsive_approach"
        if technique == RapportTechnique.CULTURAL_RESPONSIVENESS:
            return "cultural_humility_demonstration"
        return "universal_rapport_building"

    def _determine_nonverbal_focus(self, technique: RapportTechnique) -> str:
        """Determine nonverbal focus for technique."""

        nonverbal_focus = {
            RapportTechnique.EMPATHIC_REFLECTION: "facial_expression_matching",
            RapportTechnique.ACTIVE_LISTENING: "attentive_body_language",
            RapportTechnique.GENUINENESS: "natural_expressions",
            RapportTechnique.UNCONDITIONAL_POSITIVE_REGARD: "warm_presence",
            RapportTechnique.CULTURAL_RESPONSIVENESS: "culturally_appropriate_distance"
        }

        return nonverbal_focus.get(technique, "open_and_welcoming")

    def _assess_client_engagement(self, exchange_index: int) -> str:
        """Assess client engagement level based on exchange progression."""

        if exchange_index == 0:
            return "cautious"
        if exchange_index <= 1:
            return "warming_up"
        if exchange_index <= 2:
            return "engaged"
        return "highly_engaged"

    def _identify_trust_indicators(self, content: str) -> list[str]:
        """Identify trust indicators in client response."""
        indicators = []
        content_lower = content.lower()

        if any(word in content_lower for word in ["comfortable", "safe", "trust"]):
            indicators.append("explicit_trust_expression")
        if any(word in content_lower for word in ["appreciate", "value", "grateful"]):
            indicators.append("positive_regard")
        if any(word in content_lower for word in ["honest", "open", "share"]):
            indicators.append("willingness_to_disclose")
        if any(word in content_lower for word in ["together", "we", "our"]):
            indicators.append("collaborative_language")

        return indicators

    def _assess_alliance_response(self, content: str, alliance_component: AllianceComponent) -> str:
        """Assess client response to alliance-building efforts."""
        content_lower = content.lower()

        if alliance_component == AllianceComponent.TASK_ALLIANCE:
            if any(word in content_lower for word in ["makes sense", "helpful", "willing"]):
                return "positive_task_response"
            return "neutral_task_response"
        if alliance_component == AllianceComponent.BOND_ALLIANCE:
            if any(word in content_lower for word in ["comfortable", "safe", "understand"]):
                return "positive_bond_response"
            return "developing_bond_response"
        if alliance_component == AllianceComponent.GOAL_ALLIANCE:
            if any(word in content_lower for word in ["important", "want", "hope"]):
                return "positive_goal_response"
            return "exploring_goal_response"

        return "neutral_response"

    def _assess_emotional_openness(self, content: str, exchange_index: int) -> str:
        """Assess level of emotional openness in client response."""

        content_lower = content.lower()

        # Emotional words indicate openness
        emotional_words = ["feel", "emotion", "scared", "happy", "sad", "angry", "worried", "hopeful"]
        emotional_count = sum(1 for word in emotional_words if word in content_lower)

        if emotional_count >= 2 or exchange_index >= 3:
            return "high_openness"
        if emotional_count >= 1 or exchange_index >= 1:
            return "moderate_openness"
        return "guarded"

    def _identify_rapport_techniques_used(self, exchanges: list[dict[str, Any]]) -> list[RapportTechnique]:
        """Identify rapport techniques used in conversation."""
        techniques = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "rapport_technique" in exchange:
                technique_value = exchange["rapport_technique"]
                for technique in RapportTechnique:
                    if technique.value == technique_value:
                        techniques.add(technique)

        return list(techniques)

    def _identify_cultural_adaptations_applied(
        self,
        exchanges: list[dict[str, Any]],
        client_scenario: ClientScenario
    ) -> list[str]:
        """Identify cultural adaptations applied in conversation."""
        adaptations = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "cultural_adaptation" in exchange:
                adaptations.add(exchange["cultural_adaptation"])

        return list(adaptations)

    def _identify_alliance_building_moments(self, exchanges: list[dict[str, Any]]) -> list[str]:
        """Identify specific alliance-building moments in conversation."""
        moments = []

        for i, exchange in enumerate(exchanges):
            if exchange.get("speaker") == "therapist":
                element = exchange.get("alliance_building_element", "")
                if element:
                    moments.append(f"Exchange {i+1}: {element}")

        return moments

    def _identify_rupture_repair_elements(self, exchanges: list[dict[str, Any]]) -> list[str]:
        """Identify rupture repair elements (if any) in conversation."""
        # For this basic implementation, assume no ruptures in alliance-building conversations
        # In practice, this would analyze for signs of rupture and repair
        return []

    def _generate_nonverbal_considerations(self, exchanges: list[dict[str, Any]]) -> list[str]:
        """Generate nonverbal considerations for conversation."""
        considerations = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "nonverbal_focus" in exchange:
                considerations.add(exchange["nonverbal_focus"])

        return list(considerations)

    def _assess_alliance_strength(
        self,
        exchanges: list[dict[str, Any]],
        strategy: AllianceStrategy,
        client_scenario: ClientScenario
    ) -> AllianceAssessment:
        """Assess alliance strength based on conversation."""

        # Analyze client responses for alliance indicators
        client_exchanges = [e for e in exchanges if e.get("speaker") == "client"]

        # Calculate component scores based on client responses
        task_score = self._calculate_task_alliance_score(client_exchanges)
        bond_score = self._calculate_bond_alliance_score(client_exchanges)
        goal_score = self._calculate_goal_alliance_score(client_exchanges)

        # Determine overall strength
        overall_score = (task_score + bond_score + goal_score) / 3
        if overall_score >= 8:
            strength = AllianceStrength.STRONG
        elif overall_score >= 6:
            strength = AllianceStrength.MODERATE
        elif overall_score >= 4:
            strength = AllianceStrength.DEVELOPING
        else:
            strength = AllianceStrength.WEAK

        # Identify indicators
        rapport_indicators = self._extract_rapport_indicators(client_exchanges)
        rupture_indicators = self._extract_rupture_indicators(client_exchanges)
        cultural_factors = self._extract_cultural_factors(client_scenario)

        # Determine improvement areas and strengths
        improvement_areas = self._determine_improvement_areas(task_score, bond_score, goal_score)
        strengths = self._determine_alliance_strengths(task_score, bond_score, goal_score)

        assessment_id = f"alliance_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return AllianceAssessment(
            id=assessment_id,
            overall_strength=strength,
            task_alliance_score=task_score,
            bond_alliance_score=bond_score,
            goal_alliance_score=goal_score,
            rapport_indicators=rapport_indicators,
            rupture_indicators=rupture_indicators,
            cultural_factors=cultural_factors,
            improvement_areas=improvement_areas,
            strengths=strengths,
            assessment_date=datetime.now().isoformat()
        )

    def _calculate_task_alliance_score(self, client_exchanges: list[dict[str, Any]]) -> float:
        """Calculate task alliance score from client responses."""
        score = 5.0  # Base score

        for exchange in client_exchanges:
            alliance_response = exchange.get("alliance_response", "")
            if "positive_task_response" in alliance_response:
                score += 1.0
            elif "neutral_task_response" in alliance_response:
                score += 0.5

        return min(10.0, score)

    def _calculate_bond_alliance_score(self, client_exchanges: list[dict[str, Any]]) -> float:
        """Calculate bond alliance score from client responses."""
        score = 5.0  # Base score

        for exchange in client_exchanges:
            trust_indicators = exchange.get("trust_indicators", [])
            emotional_openness = exchange.get("emotional_openness", "")

            score += len(trust_indicators) * 0.5
            if emotional_openness == "high_openness":
                score += 1.0
            elif emotional_openness == "moderate_openness":
                score += 0.5

        return min(10.0, score)

    def _calculate_goal_alliance_score(self, client_exchanges: list[dict[str, Any]]) -> float:
        """Calculate goal alliance score from client responses."""
        score = 5.0  # Base score

        for exchange in client_exchanges:
            alliance_response = exchange.get("alliance_response", "")
            if "positive_goal_response" in alliance_response:
                score += 1.0
            elif "exploring_goal_response" in alliance_response:
                score += 0.5

        return min(10.0, score)

    def _extract_rapport_indicators(self, client_exchanges: list[dict[str, Any]]) -> list[str]:
        """Extract rapport indicators from client responses."""
        indicators = set()

        for exchange in client_exchanges:
            trust_indicators = exchange.get("trust_indicators", [])
            indicators.update(trust_indicators)

        return list(indicators)

    def _extract_rupture_indicators(self, client_exchanges: list[dict[str, Any]]) -> list[str]:
        """Extract rupture indicators from client responses."""
        # For alliance-building conversations, assume minimal ruptures
        return []

    def _extract_cultural_factors(self, client_scenario: ClientScenario) -> list[str]:
        """Extract cultural factors from client scenario."""
        factors = []
        demographics = client_scenario.demographics

        if "ethnicity" in demographics:
            factors.append("ethnic_background")
        if "age" in demographics:
            age = demographics["age"]
            if age < 18:
                factors.append("adolescent_considerations")
            elif age > 65:
                factors.append("elderly_considerations")

        return factors

    def _determine_improvement_areas(self, task_score: float, bond_score: float, goal_score: float) -> list[str]:
        """Determine areas for alliance improvement."""
        areas = []

        if task_score < 6:
            areas.append("task_alliance_development")
        if bond_score < 6:
            areas.append("emotional_connection_building")
        if goal_score < 6:
            areas.append("goal_clarification_and_agreement")

        return areas

    def _determine_alliance_strengths(self, task_score: float, bond_score: float, goal_score: float) -> list[str]:
        """Determine alliance strengths."""
        strengths = []

        if task_score >= 7:
            strengths.append("strong_task_collaboration")
        if bond_score >= 7:
            strengths.append("positive_emotional_connection")
        if goal_score >= 7:
            strengths.append("clear_goal_alignment")

        return strengths

    def _identify_therapeutic_gains(
        self,
        exchanges: list[dict[str, Any]],
        alliance_assessment: AllianceAssessment
    ) -> list[str]:
        """Identify therapeutic gains from alliance building."""
        gains = []

        if alliance_assessment.overall_strength in [AllianceStrength.MODERATE, AllianceStrength.STRONG]:
            gains.append("improved_therapeutic_relationship")

        if "explicit_trust_expression" in alliance_assessment.rapport_indicators:
            gains.append("increased_trust_and_safety")

        if "collaborative_language" in alliance_assessment.rapport_indicators:
            gains.append("enhanced_collaboration")

        return gains

    def _plan_next_session_focus(
        self,
        alliance_assessment: AllianceAssessment,
        strategy: AllianceStrategy
    ) -> str:
        """Plan focus for next session based on alliance assessment."""

        if alliance_assessment.improvement_areas:
            return f"Continue building {alliance_assessment.improvement_areas[0]}"
        if alliance_assessment.overall_strength == AllianceStrength.STRONG:
            return "Maintain strong alliance while deepening therapeutic work"
        return "Continue alliance development and rapport building"

    def export_alliance_conversations(
        self,
        conversations: list[AllianceConversation],
        output_file: str
    ) -> dict[str, Any]:
        """Export alliance conversations to JSON file."""

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
                "alliance_components_covered": list({
                    conv.alliance_focus.value for conv in conversations
                }),
                "rapport_techniques_used": list({
                    technique.value for conv in conversations
                    for technique in conv.rapport_techniques_used
                }),
                "average_alliance_strength": self._calculate_average_alliance_strength(conversations)
            },
            "conversations": serializable_conversations
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "alliance_components": len(export_data["metadata"]["alliance_components_covered"]),
            "rapport_techniques": len(export_data["metadata"]["rapport_techniques_used"]),
            "average_strength": export_data["metadata"]["average_alliance_strength"]
        }

    def _calculate_average_alliance_strength(self, conversations: list[AllianceConversation]) -> float:
        """Calculate average alliance strength across conversations."""
        if not conversations:
            return 0.0

        strength_values = {
            AllianceStrength.WEAK: 1,
            AllianceStrength.DEVELOPING: 2,
            AllianceStrength.MODERATE: 3,
            AllianceStrength.STRONG: 4,
            AllianceStrength.EXCEPTIONAL: 5
        }

        total_strength = sum(
            strength_values.get(conv.alliance_assessment.overall_strength, 0)
            for conv in conversations
        )

        return total_strength / len(conversations)

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
