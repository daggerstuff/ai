"""
Therapeutic response generator for psychology knowledge integration pipeline.

This module generates appropriate therapeutic responses based on client scenarios,
psychology knowledge (DSM-5, PDM-2, Big Five), and evidence-based therapeutic
techniques to create realistic therapeutic conversation training data.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_five_processor import BigFiveProcessor
from client_scenario_generator import ClientScenario, ScenarioType, SeverityLevel
from conversation_schema import Conversation, Message
from dsm5_parser import DSM5Parser
from logger import get_logger
from pdm2_parser import PDM2Parser

logger = get_logger("dataset_pipeline.therapeutic_response_generator")


class TherapeuticTechnique(Enum):
    """Evidence-based therapeutic techniques."""
    ACTIVE_LISTENING = "active_listening"
    EMPATHIC_REFLECTION = "empathic_reflection"
    OPEN_ENDED_QUESTIONING = "open_ended_questioning"
    SUMMARIZATION = "summarization"
    CLARIFICATION = "clarification"
    VALIDATION = "validation"
    PSYCHOEDUCATION = "psychoeducation"
    COGNITIVE_RESTRUCTURING = "cognitive_restructuring"
    BEHAVIORAL_ACTIVATION = "behavioral_activation"
    MINDFULNESS = "mindfulness"
    GROUNDING_TECHNIQUES = "grounding_techniques"
    SAFETY_PLANNING = "safety_planning"
    THERAPEUTIC_CHALLENGE = "therapeutic_challenge"


class ResponseType(Enum):
    """Types of therapeutic responses."""
    ASSESSMENT_QUESTION = "assessment_question"
    EMPATHIC_RESPONSE = "empathic_response"
    PSYCHOEDUCATIONAL = "psychoeducational"
    INTERVENTION_SUGGESTION = "intervention_suggestion"
    CRISIS_RESPONSE = "crisis_response"
    THERAPEUTIC_CHALLENGE = "therapeutic_challenge"
    SUPPORTIVE_STATEMENT = "supportive_statement"
    TREATMENT_PLANNING = "treatment_planning"


class TherapeuticModality(Enum):
    """Therapeutic modalities and approaches."""
    COGNITIVE_BEHAVIORAL = "cognitive_behavioral"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    DIALECTICAL_BEHAVIORAL = "dialectical_behavioral"
    ACCEPTANCE_COMMITMENT = "acceptance_commitment"
    MINDFULNESS_BASED = "mindfulness_based"
    TRAUMA_INFORMED = "trauma_informed"
    SOLUTION_FOCUSED = "solution_focused"


@dataclass
class TherapeuticResponse:
    """Individual therapeutic response with metadata."""
    id: str
    content: str
    technique: TherapeuticTechnique
    response_type: ResponseType
    modality: TherapeuticModality
    clinical_rationale: str
    target_symptoms: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(default_factory=list)
    effectiveness_indicators: list[str] = field(default_factory=list)


@dataclass
class ResponseContext:
    """Context for generating therapeutic responses."""
    client_scenario: ClientScenario
    conversation_history: list[Message] = field(default_factory=list)
    session_phase: str = "opening"  # opening, exploration, intervention, closing
    therapeutic_goals: list[str] = field(default_factory=list)
    client_readiness: str = "moderate"  # low, moderate, high
    cultural_considerations: list[str] = field(default_factory=list)


class TherapeuticResponseGenerator:
    """
    Comprehensive therapeutic response generator.

    Generates evidence-based therapeutic responses tailored to client scenarios,
    incorporating DSM-5 diagnostic knowledge, PDM-2 psychodynamic insights,
    Big Five personality considerations, and established therapeutic techniques.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the therapeutic response generator."""
        self.config = config or {}

        # Initialize psychology knowledge parsers
        self.dsm5_parser = DSM5Parser()
        self.pdm2_parser = PDM2Parser()
        self.big_five_processor = BigFiveProcessor()

        # Initialize therapeutic knowledge
        self._initialize_therapeutic_knowledge()

        logger.info("Therapeutic Response Generator initialized")

    def _initialize_therapeutic_knowledge(self) -> None:
        """Initialize therapeutic techniques and response patterns."""

        # Technique-specific response templates
        self.technique_templates = {
            TherapeuticTechnique.ACTIVE_LISTENING: [
                "I hear you saying that {client_statement}. That sounds {emotion_reflection}.",
                "It sounds like {client_concern} is really important to you.",
                "I'm listening. Please tell me more about {topic}."
            ],
            TherapeuticTechnique.EMPATHIC_REFLECTION: [
                "It sounds like you're feeling {emotion} about {situation}.",
                "I can sense that this is {intensity_word} for you.",
                "That must be {emotion_validation} to experience."
            ],
            TherapeuticTechnique.OPEN_ENDED_QUESTIONING: [
                "Can you help me understand what {topic} means to you?",
                "What has your experience been like with {situation}?",
                "How do you make sense of {client_concern}?"
            ],
            TherapeuticTechnique.VALIDATION: [
                "Your feelings about {situation} make complete sense.",
                "It's understandable that you would feel {emotion} given what you've experienced.",
                "Many people in your situation would feel similarly."
            ],
            TherapeuticTechnique.PSYCHOEDUCATION: [
                "What you're describing sounds consistent with {clinical_concept}.",
                "It might be helpful to understand that {educational_content}.",
                "Research shows that {evidence_based_information}."
            ]
        }

        # Modality-specific approaches
        self.modality_approaches = {
            TherapeuticModality.COGNITIVE_BEHAVIORAL: {
                "focus": ["thoughts", "behaviors", "thought patterns", "behavioral changes"],
                "techniques": [TherapeuticTechnique.COGNITIVE_RESTRUCTURING, TherapeuticTechnique.BEHAVIORAL_ACTIVATION],
                "language_style": "structured and goal-oriented"
            },
            TherapeuticModality.PSYCHODYNAMIC: {
                "focus": ["unconscious patterns", "early relationships", "defense mechanisms", "transference"],
                "techniques": [TherapeuticTechnique.EMPATHIC_REFLECTION, TherapeuticTechnique.CLARIFICATION],
                "language_style": "exploratory and insight-oriented"
            },
            TherapeuticModality.HUMANISTIC: {
                "focus": ["personal growth", "self-acceptance", "authentic self", "present experience"],
                "techniques": [TherapeuticTechnique.EMPATHIC_REFLECTION, TherapeuticTechnique.VALIDATION],
                "language_style": "warm and person-centered"
            }
        }

        # Crisis response protocols
        self.crisis_responses = {
            "suicidal_ideation": [
                "I'm concerned about your safety. Are you having thoughts of hurting yourself?",
                "Thank you for sharing that with me. Let's work together to keep you safe.",
                "It sounds like you're in a lot of pain right now. We need to make sure you're safe."
            ],
            "panic_attack": [
                "Let's focus on your breathing right now. Can you take slow, deep breaths with me?",
                "You're safe here. This feeling will pass. Let's use some grounding techniques.",
                "I can see you're experiencing intense anxiety. Let's work on calming your nervous system."
            ],
            "psychotic_episode": [
                "I want to understand your experience. Can you tell me what you're noticing right now?",
                "It sounds like you're having some unusual experiences. Let's talk about what's happening.",
                "I'm here to help you feel more grounded and safe."
            ]
        }

        logger.info("Initialized therapeutic knowledge base with techniques, modalities, and crisis protocols")

    def generate_therapeutic_response(
        self,
        context: ResponseContext,
        target_technique: TherapeuticTechnique = None,
        target_modality: TherapeuticModality = None
    ) -> TherapeuticResponse:
        """Generate a single therapeutic response based on context."""

        # Determine appropriate technique and modality
        technique = target_technique or self._select_appropriate_technique(context)
        modality = target_modality or self._select_appropriate_modality(context)

        # Generate response content
        content = self._generate_response_content(context, technique, modality)

        # Determine response type
        response_type = self._determine_response_type(context, technique)

        # Generate clinical rationale
        rationale = self._generate_clinical_rationale(context, technique, modality)

        # Create response ID
        response_id = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"

        # Extract target symptoms and considerations
        target_symptoms = self._extract_target_symptoms(context)
        contraindications = self._identify_contraindications(context, technique)
        follow_ups = self._suggest_follow_ups(context, technique)
        effectiveness_indicators = self._define_effectiveness_indicators(technique)

        response = TherapeuticResponse(
            id=response_id,
            content=content,
            technique=technique,
            response_type=response_type,
            modality=modality,
            clinical_rationale=rationale,
            target_symptoms=target_symptoms,
            contraindications=contraindications,
            follow_up_suggestions=follow_ups,
            effectiveness_indicators=effectiveness_indicators
        )

        logger.info(f"Generated therapeutic response: {technique.value} - {response_type.value}")
        return response

    def _select_appropriate_technique(self, context: ResponseContext) -> TherapeuticTechnique:
        """Select appropriate therapeutic technique based on context."""

        # Crisis scenarios require specific techniques
        if context.client_scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            return random.choice([
                TherapeuticTechnique.SAFETY_PLANNING,
                TherapeuticTechnique.GROUNDING_TECHNIQUES,
                TherapeuticTechnique.VALIDATION
            ])

        # Initial assessment scenarios
        if context.client_scenario.scenario_type == ScenarioType.INITIAL_ASSESSMENT:
            return random.choice([
                TherapeuticTechnique.ACTIVE_LISTENING,
                TherapeuticTechnique.OPEN_ENDED_QUESTIONING,
                TherapeuticTechnique.EMPATHIC_REFLECTION
            ])

        # Consider severity level
        if context.client_scenario.severity_level == SeverityLevel.SEVERE:
            return random.choice([
                TherapeuticTechnique.VALIDATION,
                TherapeuticTechnique.PSYCHOEDUCATION,
                TherapeuticTechnique.GROUNDING_TECHNIQUES
            ])

        # Default to common therapeutic techniques
        return random.choice([
            TherapeuticTechnique.EMPATHIC_REFLECTION,
            TherapeuticTechnique.OPEN_ENDED_QUESTIONING,
            TherapeuticTechnique.CLARIFICATION,
            TherapeuticTechnique.SUMMARIZATION
        ])

    def _select_appropriate_modality(self, context: ResponseContext) -> TherapeuticModality:
        """Select appropriate therapeutic modality based on context."""

        # Consider DSM-5 diagnostic categories
        dsm5_considerations = context.client_scenario.clinical_formulation.dsm5_considerations

        if any("anxiety" in consideration.lower() for consideration in dsm5_considerations):
            return random.choice([
                TherapeuticModality.COGNITIVE_BEHAVIORAL,
                TherapeuticModality.MINDFULNESS_BASED
            ])

        if any("depression" in consideration.lower() for consideration in dsm5_considerations):
            return random.choice([
                TherapeuticModality.COGNITIVE_BEHAVIORAL,
                TherapeuticModality.BEHAVIORAL_ACTIVATION
            ])

        if any("trauma" in consideration.lower() for consideration in dsm5_considerations):
            return TherapeuticModality.TRAUMA_INFORMED

        # Consider attachment style from PDM-2
        attachment_style = context.client_scenario.clinical_formulation.attachment_style
        if attachment_style and "insecure" in attachment_style.lower():
            return TherapeuticModality.PSYCHODYNAMIC

        # Default modality
        return random.choice([
            TherapeuticModality.HUMANISTIC,
            TherapeuticModality.COGNITIVE_BEHAVIORAL,
            TherapeuticModality.PSYCHODYNAMIC
        ])

    def _generate_response_content(
        self,
        context: ResponseContext,
        technique: TherapeuticTechnique,
        modality: TherapeuticModality
    ) -> str:
        """Generate the actual response content."""

        # Handle crisis scenarios first
        if context.client_scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            crisis_type = context.client_scenario.session_context.get("crisis_type", "general")
            if crisis_type in self.crisis_responses:
                return random.choice(self.crisis_responses[crisis_type])

        # Get base template for technique
        if technique in self.technique_templates:
            template = random.choice(self.technique_templates[technique])
        else:
            template = "I hear what you're saying. Can you tell me more about that?"

        # Extract context variables for template filling
        variables = self._extract_template_variables(context)

        # Fill template with context-appropriate content
        try:
            content = template.format(**variables)
        except KeyError:
            # Fallback if template variables don't match
            content = template

        # Adjust language style based on modality
        return self._adjust_for_modality(content, modality)


    def _extract_template_variables(self, context: ResponseContext) -> dict[str, str]:
        """Extract variables for template filling from context."""
        variables = {}

        # Extract from presenting problem
        presenting_problem = context.client_scenario.presenting_problem
        variables["client_concern"] = presenting_problem.primary_concern.lower()
        variables["situation"] = presenting_problem.primary_concern.lower()
        variables["topic"] = presenting_problem.primary_concern.lower()

        # Extract emotions and intensity
        if presenting_problem.symptoms:
            primary_symptom = presenting_problem.symptoms[0].lower()
            if any(word in primary_symptom for word in ["anxious", "worry", "nervous"]):
                variables["emotion"] = "anxious"
                variables["emotion_reflection"] = "really concerning"
                variables["emotion_validation"] = "overwhelming"
            elif any(word in primary_symptom for word in ["sad", "depressed", "down"]):
                variables["emotion"] = "sad"
                variables["emotion_reflection"] = "really difficult"
                variables["emotion_validation"] = "painful"
            else:
                variables["emotion"] = "distressed"
                variables["emotion_reflection"] = "challenging"
                variables["emotion_validation"] = "difficult"

        # Intensity based on severity
        if context.client_scenario.severity_level == SeverityLevel.SEVERE:
            variables["intensity_word"] = "overwhelming"
        elif context.client_scenario.severity_level == SeverityLevel.MODERATE:
            variables["intensity_word"] = "challenging"
        else:
            variables["intensity_word"] = "difficult"

        # Clinical concepts from DSM-5
        if context.client_scenario.clinical_formulation.dsm5_considerations:
            variables["clinical_concept"] = context.client_scenario.clinical_formulation.dsm5_considerations[0]

        # Educational content based on symptoms
        variables["educational_content"] = "these symptoms are treatable with proper support"
        variables["evidence_based_information"] = "effective treatments are available for these concerns"

        # Client statement (placeholder for conversation context)
        variables["client_statement"] = "you're experiencing these difficulties"

        return variables

    def _adjust_for_modality(self, content: str, modality: TherapeuticModality) -> str:
        """Adjust response content based on therapeutic modality."""

        if modality == TherapeuticModality.COGNITIVE_BEHAVIORAL:
            # Add CBT-specific language
            if "thoughts" not in content.lower():
                content += " What thoughts go through your mind when this happens?"

        elif modality == TherapeuticModality.PSYCHODYNAMIC:
            # Add psychodynamic exploration
            if "pattern" not in content.lower():
                content += " I wonder if this connects to patterns in your relationships."

        elif modality == TherapeuticModality.HUMANISTIC:
            # Add person-centered warmth
            if not any(word in content.lower() for word in ["feel", "experience", "sense"]):
                content += " How does this feel for you right now?"

        return content

    def _determine_response_type(self, context: ResponseContext, technique: TherapeuticTechnique) -> ResponseType:
        """Determine the type of therapeutic response."""

        if context.client_scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            return ResponseType.CRISIS_RESPONSE

        if technique == TherapeuticTechnique.PSYCHOEDUCATION:
            return ResponseType.PSYCHOEDUCATIONAL

        if technique in [TherapeuticTechnique.EMPATHIC_REFLECTION, TherapeuticTechnique.VALIDATION]:
            return ResponseType.EMPATHIC_RESPONSE

        if technique == TherapeuticTechnique.OPEN_ENDED_QUESTIONING:
            return ResponseType.ASSESSMENT_QUESTION

        if technique in [TherapeuticTechnique.COGNITIVE_RESTRUCTURING, TherapeuticTechnique.BEHAVIORAL_ACTIVATION]:
            return ResponseType.INTERVENTION_SUGGESTION

        return ResponseType.SUPPORTIVE_STATEMENT

    def _generate_clinical_rationale(
        self,
        context: ResponseContext,
        technique: TherapeuticTechnique,
        modality: TherapeuticModality
    ) -> str:
        """Generate clinical rationale for the response."""

        rationale_parts = []

        # Technique rationale
        technique_rationales = {
            TherapeuticTechnique.EMPATHIC_REFLECTION: "builds therapeutic rapport and validates client experience",
            TherapeuticTechnique.OPEN_ENDED_QUESTIONING: "encourages client exploration and self-discovery",
            TherapeuticTechnique.VALIDATION: "normalizes client experience and reduces shame",
            TherapeuticTechnique.PSYCHOEDUCATION: "provides understanding and reduces anxiety about symptoms",
            TherapeuticTechnique.ACTIVE_LISTENING: "demonstrates therapist presence and attention",
            TherapeuticTechnique.SAFETY_PLANNING: "addresses immediate safety concerns and risk management"
        }

        if technique in technique_rationales:
            rationale_parts.append(technique_rationales[technique])

        # Context-specific rationale
        if context.client_scenario.severity_level == SeverityLevel.SEVERE:
            rationale_parts.append("appropriate for high symptom severity")

        if context.client_scenario.scenario_type == ScenarioType.INITIAL_ASSESSMENT:
            rationale_parts.append("suitable for initial therapeutic contact")

        # Modality rationale
        modality_rationales = {
            TherapeuticModality.COGNITIVE_BEHAVIORAL: "addresses thought-behavior connections",
            TherapeuticModality.PSYCHODYNAMIC: "explores unconscious patterns and relationships",
            TherapeuticModality.HUMANISTIC: "emphasizes client self-determination and growth"
        }

        if modality in modality_rationales:
            rationale_parts.append(modality_rationales[modality])

        return "; ".join(rationale_parts) if rationale_parts else "evidence-based therapeutic response"

    def _extract_target_symptoms(self, context: ResponseContext) -> list[str]:
        """Extract target symptoms from client scenario."""
        return context.client_scenario.presenting_problem.symptoms[:3]  # First 3 symptoms

    def _identify_contraindications(self, context: ResponseContext, technique: TherapeuticTechnique) -> list[str]:
        """Identify contraindications for the technique."""
        contraindications = []

        # General contraindications
        if technique == TherapeuticTechnique.COGNITIVE_RESTRUCTURING:
            if context.client_scenario.severity_level == SeverityLevel.CRISIS:
                contraindications.append("Not appropriate during crisis states")

        if technique == TherapeuticTechnique.THERAPEUTIC_CHALLENGE:
            if context.client_scenario.scenario_type == ScenarioType.INITIAL_ASSESSMENT:
                contraindications.append("Too confrontational for initial sessions")

        # Personality-based contraindications
        personality = context.client_scenario.clinical_formulation.personality_profile
        if personality.get("neuroticism") == "high":
            if technique == TherapeuticTechnique.THERAPEUTIC_CHALLENGE:
                contraindications.append("High neuroticism may not tolerate challenges well")

        return contraindications

    def _suggest_follow_ups(self, context: ResponseContext, technique: TherapeuticTechnique) -> list[str]:
        """Suggest follow-up interventions."""
        follow_ups = []

        if technique == TherapeuticTechnique.EMPATHIC_REFLECTION:
            follow_ups.append("Continue with open-ended exploration")
            follow_ups.append("Validate emotional experience further")

        if technique == TherapeuticTechnique.PSYCHOEDUCATION:
            follow_ups.append("Check client understanding")
            follow_ups.append("Provide additional resources if needed")

        if technique == TherapeuticTechnique.SAFETY_PLANNING:
            follow_ups.append("Review safety plan regularly")
            follow_ups.append("Assess ongoing risk factors")

        return follow_ups

    def _define_effectiveness_indicators(self, technique: TherapeuticTechnique) -> list[str]:
        """Define indicators of technique effectiveness."""
        indicators = {
            TherapeuticTechnique.EMPATHIC_REFLECTION: [
                "Client feels heard and understood",
                "Increased emotional expression",
                "Stronger therapeutic alliance"
            ],
            TherapeuticTechnique.OPEN_ENDED_QUESTIONING: [
                "Client provides more detailed responses",
                "Increased self-exploration",
                "New insights or perspectives emerge"
            ],
            TherapeuticTechnique.VALIDATION: [
                "Reduced client defensiveness",
                "Increased emotional regulation",
                "Greater willingness to share"
            ],
            TherapeuticTechnique.PSYCHOEDUCATION: [
                "Improved understanding of symptoms",
                "Reduced anxiety about condition",
                "Increased treatment engagement"
            ]
        }

        return indicators.get(technique, ["Positive therapeutic engagement", "Symptom improvement"])

    def generate_conversation_with_responses(
        self,
        client_scenario: ClientScenario,
        num_exchanges: int = 5,
        target_modality: TherapeuticModality = None
    ) -> Conversation:
        """Generate a complete therapeutic conversation with appropriate responses."""

        messages = []
        context = ResponseContext(client_scenario=client_scenario)

        # Opening therapist message
        opening_response = self.generate_therapeutic_response(
            context,
            target_technique=TherapeuticTechnique.OPEN_ENDED_QUESTIONING,
            target_modality=target_modality
        )

        messages.append(Message(
            role="therapist",
            content=opening_response.content,
            meta={
                "technique": opening_response.technique.value,
                "response_type": opening_response.response_type.value,
                "modality": opening_response.modality.value,
                "clinical_rationale": opening_response.clinical_rationale
            }
        ))

        # Generate alternating client-therapist exchanges
        for _i in range(num_exchanges):
            # Client response (simulated based on scenario)
            client_message = self._generate_client_response(client_scenario, messages)
            messages.append(client_message)

            # Update context with conversation history
            context.conversation_history = messages

            # Therapist response
            therapist_response = self.generate_therapeutic_response(
                context,
                target_modality=target_modality
            )

            messages.append(Message(
                role="therapist",
                content=therapist_response.content,
                meta={
                    "technique": therapist_response.technique.value,
                    "response_type": therapist_response.response_type.value,
                    "modality": therapist_response.modality.value,
                    "clinical_rationale": therapist_response.clinical_rationale,
                    "target_symptoms": therapist_response.target_symptoms
                }
            ))

        # Create conversation
        conversation = Conversation(
            id=f"therapeutic_conv_{client_scenario.id}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "client_scenario_id": client_scenario.id,
                "scenario_type": client_scenario.scenario_type.value,
                "severity_level": client_scenario.severity_level.value,
                "therapeutic_modality": target_modality.value if target_modality else "mixed",
                "num_exchanges": num_exchanges
            },
            source="therapeutic_response_generator",
            meta={
                "clinical_formulation": asdict(client_scenario.clinical_formulation),
                "learning_objectives": client_scenario.learning_objectives,
                "therapeutic_considerations": client_scenario.therapeutic_considerations
            }
        )

        logger.info(f"Generated therapeutic conversation with {len(messages)} messages")
        return conversation

    def _generate_client_response(self, scenario: ClientScenario, conversation_history: list[Message]) -> Message:
        """Generate a realistic client response based on scenario and conversation context."""

        # Determine response based on conversation length and scenario
        if len(conversation_history) == 1:  # First client response
            content = self._generate_initial_client_response(scenario)
        else:
            content = self._generate_follow_up_client_response(scenario, conversation_history)

        return Message(
            role="client",
            content=content,
            meta={
                "scenario_id": scenario.id,
                "response_type": "authentic_client_response",
                "severity_level": scenario.severity_level.value
            }
        )

    def _generate_initial_client_response(self, scenario: ClientScenario) -> str:
        """Generate initial client response to opening question."""

        primary_concern = scenario.presenting_problem.primary_concern
        symptoms = scenario.presenting_problem.symptoms

        if scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            return f"I don't know what to do anymore. {primary_concern.lower()} and I feel like I can't handle it."

        if scenario.severity_level == SeverityLevel.SEVERE:
            return f"I've been really struggling with {primary_concern.lower()}. It's affecting everything in my life and I don't know how to cope."

        if symptoms:
            return f"Well, I've been dealing with {primary_concern.lower()}. I've been experiencing {symptoms[0].lower()} and it's been really difficult."

        return f"I've been having trouble with {primary_concern.lower()}. It's been going on for {scenario.presenting_problem.duration.lower()}."

    def _generate_follow_up_client_response(self, scenario: ClientScenario, history: list[Message]) -> str:
        """Generate follow-up client response based on conversation flow."""

        last_therapist_message = None
        for msg in reversed(history):
            if msg.role == "therapist":
                last_therapist_message = msg
                break

        if not last_therapist_message:
            return "I'm not sure how to answer that."

        # Response based on therapist technique
        technique = last_therapist_message.meta.get("technique", "")

        if technique == "empathic_reflection":
            return "Yes, exactly. It's really hard to deal with."

        if technique == "open_ended_questioning":
            if scenario.presenting_problem.triggers:
                trigger = scenario.presenting_problem.triggers[0]
                return f"I think it started when {trigger.lower()}. That's when I really noticed things getting worse."
            return "I'm not really sure when it started. It just gradually got worse over time."

        if technique == "validation":
            return "Thank you for saying that. Sometimes I feel like I'm going crazy."

        if technique == "psychoeducation":
            return "That's helpful to know. I was worried there was something seriously wrong with me."

        # Default response
        return "I hadn't thought about it that way before. Can you tell me more?"

    def generate_response_batch(
        self,
        scenarios: list[ClientScenario],
        responses_per_scenario: int = 3
    ) -> list[TherapeuticResponse]:
        """Generate a batch of therapeutic responses for multiple scenarios."""

        all_responses = []

        for scenario in scenarios:
            context = ResponseContext(client_scenario=scenario)

            for _ in range(responses_per_scenario):
                response = self.generate_therapeutic_response(context)
                all_responses.append(response)

        logger.info(f"Generated {len(all_responses)} therapeutic responses for {len(scenarios)} scenarios")
        return all_responses

    def export_responses_to_json(self, responses: list[TherapeuticResponse], output_path: Path) -> bool:
        """Export therapeutic responses to JSON format."""
        try:
            export_data = {
                "responses": [],
                "metadata": {
                    "total_responses": len(responses),
                    "generated_at": datetime.now().isoformat(),
                    "generator_version": "1.0"
                }
            }

            for response in responses:
                response_dict = {
                    "id": response.id,
                    "content": response.content,
                    "technique": response.technique.value,
                    "response_type": response.response_type.value,
                    "modality": response.modality.value,
                    "clinical_rationale": response.clinical_rationale,
                    "target_symptoms": response.target_symptoms,
                    "contraindications": response.contraindications,
                    "follow_up_suggestions": response.follow_up_suggestions,
                    "effectiveness_indicators": response.effectiveness_indicators
                }
                export_data["responses"].append(response_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(responses)} therapeutic responses to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export therapeutic responses: {e}")
            return False

    def get_statistics(self, responses: list[TherapeuticResponse]) -> dict[str, Any]:
        """Get statistics about generated therapeutic responses."""
        if not responses:
            return {}

        stats = {
            "total_responses": len(responses),
            "techniques_used": {},
            "response_types": {},
            "modalities_used": {},
            "average_contraindications": 0,
            "average_follow_ups": 0
        }

        total_contraindications = 0
        total_follow_ups = 0

        for response in responses:
            # Techniques
            technique = response.technique.value
            stats["techniques_used"][technique] = stats["techniques_used"].get(technique, 0) + 1

            # Response types
            response_type = response.response_type.value
            stats["response_types"][response_type] = stats["response_types"].get(response_type, 0) + 1

            # Modalities
            modality = response.modality.value
            stats["modalities_used"][modality] = stats["modalities_used"].get(modality, 0) + 1

            # Averages
            total_contraindications += len(response.contraindications)
            total_follow_ups += len(response.follow_up_suggestions)

        stats["average_contraindications"] = total_contraindications / len(responses)
        stats["average_follow_ups"] = total_follow_ups / len(responses)

        return stats
