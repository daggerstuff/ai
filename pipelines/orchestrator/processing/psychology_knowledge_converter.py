"""
Psychology knowledge converter for therapeutic conversation generation.

This module converts structured psychology knowledge from DSM-5, PDM-2, and Big Five
processors into conversational training format, creating realistic therapeutic dialogues
that incorporate clinical knowledge and assessment frameworks.
"""

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from big_five_processor import BigFiveProcessor
from conversation_schema import Conversation, Message
from dsm5_parser import DSM5Parser, DSMDisorder
from logger import get_logger
from pdm2_parser import PDM2Parser

logger = get_logger("dataset_pipeline.psychology_knowledge_converter")


class ConversationType(Enum):
    """Types of psychology-based conversations."""
    DIAGNOSTIC_ASSESSMENT = "diagnostic_assessment"
    PERSONALITY_EXPLORATION = "personality_exploration"
    PSYCHODYNAMIC_EXPLORATION = "psychodynamic_exploration"
    CLINICAL_INTERVIEW = "clinical_interview"
    THERAPEUTIC_EDUCATION = "therapeutic_education"
    SYMPTOM_ASSESSMENT = "symptom_assessment"


class ConversationStyle(Enum):
    """Therapeutic conversation styles."""
    STRUCTURED_INTERVIEW = "structured_interview"
    EXPLORATORY_DIALOGUE = "exploratory_dialogue"
    EDUCATIONAL_DISCUSSION = "educational_discussion"
    ASSESSMENT_FOCUSED = "assessment_focused"
    SUPPORTIVE_INQUIRY = "supportive_inquiry"


@dataclass
class ConversationTemplate:
    """Template for generating psychology-based conversations."""
    id: str
    conversation_type: ConversationType
    style: ConversationStyle
    knowledge_source: str  # DSM-5, PDM-2, or Big Five
    target_concept: str
    learning_objectives: list[str] = field(default_factory=list)
    clinical_focus: list[str] = field(default_factory=list)
    conversation_flow: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PsychologyKnowledgeConverter:
    """
    Comprehensive psychology knowledge converter.

    Converts structured psychology knowledge from DSM-5, PDM-2, and Big Five
    processors into realistic therapeutic conversation training data that
    incorporates clinical assessment and therapeutic dialogue patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the psychology knowledge converter."""
        self.config = config or {}

        # Initialize psychology knowledge parsers
        self.dsm5_parser = DSM5Parser()
        self.pdm2_parser = PDM2Parser()
        self.big_five_processor = BigFiveProcessor()

        # Initialize conversation templates
        self._initialize_conversation_templates()

        logger.info("Psychology Knowledge Converter initialized")

    def _initialize_conversation_templates(self) -> None:
        """Initialize conversation templates for different knowledge types."""
        self.conversation_templates = {
            "dsm5_diagnostic": self._create_dsm5_templates(),
            "pdm2_psychodynamic": self._create_pdm2_templates(),
            "big_five_personality": self._create_big_five_templates()
        }

        logger.info(f"Initialized {sum(len(templates) for templates in self.conversation_templates.values())} conversation templates")

    def _create_dsm5_templates(self) -> list[ConversationTemplate]:
        """Create conversation templates for DSM-5 knowledge."""
        templates = []

        # Diagnostic assessment template
        templates.append(ConversationTemplate(
            id="dsm5_diagnostic_assessment",
            conversation_type=ConversationType.DIAGNOSTIC_ASSESSMENT,
            style=ConversationStyle.STRUCTURED_INTERVIEW,
            knowledge_source="DSM-5",
            target_concept="diagnostic_criteria",
            learning_objectives=[
                "Apply DSM-5 diagnostic criteria systematically",
                "Conduct structured diagnostic interviews",
                "Assess symptom severity and functional impact"
            ],
            clinical_focus=[
                "Criterion-based assessment",
                "Differential diagnosis",
                "Functional impairment evaluation"
            ],
            conversation_flow=[
                {"role": "therapist", "purpose": "introduce_assessment", "knowledge_integration": "disorder_overview"},
                {"role": "client", "purpose": "present_symptoms", "knowledge_integration": "criterion_examples"},
                {"role": "therapist", "purpose": "explore_criteria", "knowledge_integration": "systematic_assessment"},
                {"role": "client", "purpose": "elaborate_symptoms", "knowledge_integration": "detailed_examples"},
                {"role": "therapist", "purpose": "assess_severity", "knowledge_integration": "functional_impact"}
            ]
        ))

        # Clinical interview template
        templates.append(ConversationTemplate(
            id="dsm5_clinical_interview",
            conversation_type=ConversationType.CLINICAL_INTERVIEW,
            style=ConversationStyle.EXPLORATORY_DIALOGUE,
            knowledge_source="DSM-5",
            target_concept="clinical_presentation",
            learning_objectives=[
                "Explore clinical presentations naturally",
                "Integrate diagnostic thinking with therapeutic rapport",
                "Assess multiple diagnostic possibilities"
            ],
            clinical_focus=[
                "Naturalistic symptom exploration",
                "Therapeutic relationship building",
                "Comprehensive assessment"
            ]
        ))

        # Therapeutic education template
        templates.append(ConversationTemplate(
            id="dsm5_therapeutic_education",
            conversation_type=ConversationType.THERAPEUTIC_EDUCATION,
            style=ConversationStyle.EDUCATIONAL_DISCUSSION,
            knowledge_source="DSM-5",
            target_concept="psychoeducation",
            learning_objectives=[
                "Provide accurate psychoeducation about disorders",
                "Help clients understand their symptoms",
                "Normalize experiences within diagnostic framework"
            ],
            clinical_focus=[
                "Disorder explanation",
                "Symptom normalization",
                "Treatment planning"
            ]
        ))

        return templates

    def _create_pdm2_templates(self) -> list[ConversationTemplate]:
        """Create conversation templates for PDM-2 knowledge."""
        templates = []

        # Psychodynamic exploration template
        templates.append(ConversationTemplate(
            id="pdm2_psychodynamic_exploration",
            conversation_type=ConversationType.PSYCHODYNAMIC_EXPLORATION,
            style=ConversationStyle.EXPLORATORY_DIALOGUE,
            knowledge_source="PDM-2",
            target_concept="psychodynamic_patterns",
            learning_objectives=[
                "Explore unconscious patterns and conflicts",
                "Understand attachment and relationship dynamics",
                "Identify defense mechanisms in action"
            ],
            clinical_focus=[
                "Attachment pattern exploration",
                "Defense mechanism identification",
                "Unconscious conflict understanding"
            ]
        ))

        # Attachment assessment template
        templates.append(ConversationTemplate(
            id="pdm2_attachment_assessment",
            conversation_type=ConversationType.CLINICAL_INTERVIEW,
            style=ConversationStyle.SUPPORTIVE_INQUIRY,
            knowledge_source="PDM-2",
            target_concept="attachment_styles",
            learning_objectives=[
                "Assess attachment patterns in relationships",
                "Understand early relationship experiences",
                "Connect attachment to current difficulties"
            ],
            clinical_focus=[
                "Relationship history exploration",
                "Attachment behavior patterns",
                "Therapeutic relationship dynamics"
            ]
        ))

        return templates

    def _create_big_five_templates(self) -> list[ConversationTemplate]:
        """Create conversation templates for Big Five knowledge."""
        templates = []

        # Personality exploration template
        templates.append(ConversationTemplate(
            id="big_five_personality_exploration",
            conversation_type=ConversationType.PERSONALITY_EXPLORATION,
            style=ConversationStyle.EXPLORATORY_DIALOGUE,
            knowledge_source="Big Five",
            target_concept="personality_factors",
            learning_objectives=[
                "Explore personality traits and patterns",
                "Understand how personality affects functioning",
                "Connect personality to therapeutic goals"
            ],
            clinical_focus=[
                "Personality trait assessment",
                "Behavioral pattern exploration",
                "Therapeutic matching"
            ]
        ))

        # Personality assessment template
        templates.append(ConversationTemplate(
            id="big_five_personality_assessment",
            conversation_type=ConversationType.SYMPTOM_ASSESSMENT,
            style=ConversationStyle.ASSESSMENT_FOCUSED,
            knowledge_source="Big Five",
            target_concept="personality_assessment",
            learning_objectives=[
                "Conduct systematic personality assessment",
                "Use personality insights for treatment planning",
                "Understand personality-psychopathology interactions"
            ],
            clinical_focus=[
                "Structured personality evaluation",
                "Clinical implications of traits",
                "Treatment customization"
            ]
        ))

        return templates

    def convert_dsm5_to_conversations(self, disorder_name: str | None = None, count: int = 5) -> list[Conversation]:
        """Convert DSM-5 knowledge into therapeutic conversations."""
        conversations = []

        # Get disorders to convert
        if disorder_name:
            disorder = self.dsm5_parser.get_disorder_by_name(disorder_name)
            disorders = [disorder] if disorder else []
        else:
            disorders = self.dsm5_parser.get_disorders()[:count]

        for disorder in disorders:
            # Generate diagnostic assessment conversation
            diagnostic_conv = self._generate_dsm5_diagnostic_conversation(disorder)
            conversations.append(diagnostic_conv)

            # Generate clinical interview conversation
            interview_conv = self._generate_dsm5_clinical_interview(disorder)
            conversations.append(interview_conv)

            # Generate psychoeducation conversation
            education_conv = self._generate_dsm5_education_conversation(disorder)
            conversations.append(education_conv)

        logger.info(f"Generated {len(conversations)} DSM-5 conversations")
        return conversations

    def _generate_dsm5_diagnostic_conversation(self, disorder: DSMDisorder) -> Conversation:
        """Generate diagnostic assessment conversation for a DSM-5 disorder."""
        messages = []

        # Opening
        messages.append(Message(
            role="therapist",
            content=f"I'd like to go through some specific questions to better understand your symptoms. This will help me determine if what you're experiencing aligns with {disorder.name}.",
            meta={"type": "diagnostic_introduction", "disorder": disorder.name, "knowledge_source": "DSM-5"}
        ))

        # Systematic criterion assessment
        for _i, criterion in enumerate(disorder.criteria[:4]):  # Limit to first 4 criteria
            # Therapist question based on criterion
            messages.append(Message(
                role="therapist",
                content=f"Have you been experiencing {criterion.description.lower()}?",
                meta={
                    "criterion_id": criterion.id,
                    "criterion_category": criterion.category,
                    "assessment_type": "systematic",
                    "knowledge_source": "DSM-5"
                }
            ))

            # Client response with examples
            if criterion.examples:
                example = random.choice(criterion.examples)
                messages.append(Message(
                    role="client",
                    content=f"Yes, I have been experiencing {example.lower()}. It's been really difficult for me.",
                    meta={
                        "criterion_id": criterion.id,
                        "symptom_example": example,
                        "knowledge_source": "DSM-5"
                    }
                ))

                # Follow-up exploration
                messages.append(Message(
                    role="therapist",
                    content="Can you tell me more about how this has been affecting your daily life?",
                    meta={
                        "technique": "functional_assessment",
                        "criterion_id": criterion.id,
                        "knowledge_source": "DSM-5"
                    }
                ))

                messages.append(Message(
                    role="client",
                    content=f"It's made it hard to {random.choice(['work', 'sleep', 'concentrate', 'enjoy things', 'maintain relationships'])}. I feel like I can't function normally.",
                    meta={
                        "functional_impact": True,
                        "criterion_id": criterion.id,
                        "knowledge_source": "DSM-5"
                    }
                ))

        # Duration and severity assessment
        messages.append(Message(
            role="therapist",
            content=f"How long have you been experiencing these symptoms? The diagnostic criteria specify {disorder.duration_requirement}.",
            meta={
                "assessment_type": "duration",
                "duration_requirement": disorder.duration_requirement,
                "knowledge_source": "DSM-5"
            }
        ))

        messages.append(Message(
            role="client",
            content=f"It's been going on for {disorder.duration_requirement.lower()}, maybe even longer. It feels like it's getting worse.",
            meta={
                "duration_confirmation": True,
                "severity_indication": "worsening",
                "knowledge_source": "DSM-5"
            }
        ))

        # Clinical summary
        messages.append(Message(
            role="therapist",
            content=f"Based on what you've shared, your symptoms align with the criteria for {disorder.name}. This means we have a clear framework for understanding what you're going through and developing an effective treatment plan.",
            meta={
                "type": "diagnostic_summary",
                "disorder": disorder.name,
                "knowledge_source": "DSM-5"
            }
        ))

        return Conversation(
            id=f"dsm5_diagnostic_{disorder.code}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "disorder": disorder.name,
                "disorder_code": disorder.code,
                "category": disorder.category.value,
                "conversation_type": "diagnostic_assessment",
                "knowledge_source": "DSM-5",
                "minimum_criteria": disorder.minimum_criteria_count,
                "duration_requirement": disorder.duration_requirement
            },
            source="psychology_knowledge_converter",
            meta={
                "learning_objectives": [
                    "Apply DSM-5 diagnostic criteria systematically",
                    "Conduct structured diagnostic interviews",
                    "Assess symptom severity and functional impact"
                ],
                "clinical_focus": [
                    "Criterion-based assessment",
                    "Differential diagnosis",
                    "Functional impairment evaluation"
                ],
                "knowledge_integration": "DSM-5 diagnostic criteria"
            }
        )


    def _generate_dsm5_clinical_interview(self, disorder) -> Conversation:
        """Generate clinical interview conversation for a DSM-5 disorder."""
        messages = []

        # Natural opening
        messages.append(Message(
            role="therapist",
            content="I'm here to listen and understand what you've been going through. What's been on your mind lately?",
            meta={"type": "open_inquiry", "style": "naturalistic", "knowledge_source": "DSM-5"}
        ))

        # Client presents naturally
        primary_symptom = disorder.criteria[0].examples[0] if disorder.criteria and disorder.criteria[0].examples else "difficult symptoms"
        messages.append(Message(
            role="client",
            content=f"I've been really struggling with {primary_symptom.lower()}. It's affecting everything in my life.",
            meta={"symptom_presentation": "natural", "primary_concern": True, "knowledge_source": "DSM-5"}
        ))

        # Empathic response with gentle exploration
        messages.append(Message(
            role="therapist",
            content="That sounds really challenging. I can hear how much this is impacting you. Can you help me understand what this experience is like for you?",
            meta={"technique": "empathic_reflection", "purpose": "rapport_building", "knowledge_source": "DSM-5"}
        ))

        return Conversation(
            id=f"dsm5_interview_{disorder.code}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "disorder": disorder.name,
                "conversation_type": "clinical_interview",
                "style": "naturalistic",
                "knowledge_source": "DSM-5"
            },
            source="psychology_knowledge_converter",
            meta={
                "learning_objectives": [
                    "Explore clinical presentations naturally",
                    "Integrate diagnostic thinking with therapeutic rapport"
                ],
                "knowledge_integration": "DSM-5 clinical presentation"
            }
        )


    def _generate_dsm5_education_conversation(self, disorder) -> Conversation:
        """Generate psychoeducation conversation for a DSM-5 disorder."""
        messages = []

        # Educational introduction
        messages.append(Message(
            role="therapist",
            content=f"Based on our assessment, it appears you're experiencing {disorder.name}. I'd like to help you understand what this means.",
            meta={"type": "psychoeducation_intro", "disorder": disorder.name, "knowledge_source": "DSM-5"}
        ))

        # Client seeks understanding
        messages.append(Message(
            role="client",
            content="I've heard of that before, but I'm not really sure what it means. Is this something serious?",
            meta={"information_seeking": True, "concern_expression": True, "knowledge_source": "DSM-5"}
        ))

        # Explanation with normalization
        prevalence_info = disorder.prevalence or "affects many people"
        messages.append(Message(
            role="therapist",
            content=f"{disorder.name} is a well-understood condition that {prevalence_info.lower()}. It's very treatable with proper support.",
            meta={
                "psychoeducation": True,
                "prevalence_info": disorder.prevalence,
                "normalization": True,
                "knowledge_source": "DSM-5"
            }
        ))

        return Conversation(
            id=f"dsm5_education_{disorder.code}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "disorder": disorder.name,
                "conversation_type": "psychoeducation",
                "knowledge_source": "DSM-5"
            },
            source="psychology_knowledge_converter",
            meta={
                "learning_objectives": [
                    "Provide accurate psychoeducation about disorders",
                    "Help clients understand their symptoms"
                ],
                "knowledge_integration": "DSM-5 disorder information"
            }
        )


    def _generate_big_five_exploration_conversation(self, profile) -> Conversation:
        """Generate personality exploration conversation for Big Five profile."""
        messages = []

        # Opening exploration
        messages.append(Message(
            role="therapist",
            content=f"I'd like to explore your {profile.name.lower()} with you. This can help us understand your natural tendencies and how they might relate to what you're experiencing.",
            meta={"type": "personality_introduction", "factor": profile.factor.value, "knowledge_source": "Big Five"}
        ))

        # Client curiosity
        messages.append(Message(
            role="client",
            content="I'm not sure what that means exactly. How does my personality relate to my problems?",
            meta={"curiosity": True, "knowledge_seeking": True, "knowledge_source": "Big Five"}
        ))

        # Explanation with examples
        high_traits = profile.score_interpretations.get("high", [])
        if high_traits:
            messages.append(Message(
                role="therapist",
                content=f"Well, people who score high on {profile.name.lower()} tend to be {high_traits[0].lower()}. This can be a strength, but sometimes it can also create challenges. How would you describe yourself in this area?",
                meta={
                    "psychoeducation": True,
                    "trait_explanation": high_traits[0],
                    "knowledge_source": "Big Five"
                }
            ))

        # Self-reflection
        messages.append(Message(
            role="client",
            content=f"I think I am pretty {high_traits[0].lower() if high_traits else 'typical'} in that way. Sometimes it helps me, but other times it makes things harder.",
            meta={"self_reflection": True, "trait_acknowledgment": True, "knowledge_source": "Big Five"}
        ))

        return Conversation(
            id=f"big_five_exploration_{profile.factor.value}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "personality_factor": profile.factor.value,
                "conversation_type": "personality_exploration",
                "knowledge_source": "Big Five"
            },
            source="psychology_knowledge_converter",
            meta={
                "learning_objectives": [
                    "Explore personality traits and patterns",
                    "Understand how personality affects functioning",
                    "Connect personality to therapeutic goals"
                ],
                "knowledge_integration": "Big Five personality factors"
            }
        )


    def _generate_big_five_assessment_conversation(self, profile) -> Conversation:
        """Generate personality assessment conversation for Big Five profile."""
        messages = []

        # Assessment introduction
        messages.append(Message(
            role="therapist",
            content=f"I'd like to ask you some questions about your {profile.name.lower()}. These questions will help me understand your personality style better.",
            meta={"type": "assessment_introduction", "factor": profile.factor.value, "knowledge_source": "Big Five"}
        ))

        # Use facets for specific questions
        if profile.facets:
            facet = profile.facets[0]
            messages.append(Message(
                role="therapist",
                content=f"How would you describe yourself when it comes to {facet.description.lower()}?",
                meta={
                    "facet_assessment": facet.name,
                    "assessment_type": "open_ended",
                    "knowledge_source": "Big Five"
                }
            ))

            # Client response based on facet characteristics
            if facet.high_score_characteristics:
                characteristic = facet.high_score_characteristics[0]
                messages.append(Message(
                    role="client",
                    content=f"I would say I'm someone who {characteristic.lower()}. It's just how I've always been.",
                    meta={
                        "facet_response": facet.name,
                        "characteristic": characteristic,
                        "knowledge_source": "Big Five"
                    }
                ))

        return Conversation(
            id=f"big_five_assessment_{profile.factor.value}_{random.randint(1000, 9999)}",
            messages=messages,
            context={
                "personality_factor": profile.factor.value,
                "conversation_type": "personality_assessment",
                "knowledge_source": "Big Five"
            },
            source="psychology_knowledge_converter",
            meta={
                "learning_objectives": [
                    "Conduct systematic personality assessment",
                    "Use personality insights for treatment planning"
                ],
                "knowledge_integration": "Big Five assessment methods"
            }
        )


    def convert_big_five_to_conversations(self, count: int = 5) -> list[Conversation]:
        """Convert Big Five knowledge into therapeutic conversations."""
        conversations = []

        # Get personality profiles
        profiles = self.big_five_processor.get_personality_profiles()[:count]

        for profile in profiles:
            # Generate personality exploration conversation
            exploration_conv = self._generate_big_five_exploration_conversation(profile)
            conversations.append(exploration_conv)

            # Generate personality assessment conversation
            assessment_conv = self._generate_big_five_assessment_conversation(profile)
            conversations.append(assessment_conv)

        logger.info(f"Generated {len(conversations)} Big Five conversations")
        return conversations

    def generate_comprehensive_dataset(
        self,
        dsm5_count: int = 3,
        pdm2_count: int = 3,
        big_five_count: int = 3
    ) -> list[Conversation]:
        """Generate comprehensive conversation dataset from all psychology knowledge."""
        all_conversations = []

        # Generate DSM-5 conversations
        dsm5_conversations = self.convert_dsm5_to_conversations(count=dsm5_count)
        all_conversations.extend(dsm5_conversations)

        # Generate PDM-2 conversations (simplified for now)
        # pdm2_conversations = self.convert_pdm2_to_conversations(count=pdm2_count)
        # all_conversations.extend(pdm2_conversations)

        # Generate Big Five conversations
        big_five_conversations = self.convert_big_five_to_conversations(count=big_five_count)
        all_conversations.extend(big_five_conversations)

        logger.info(f"Generated comprehensive dataset with {len(all_conversations)} conversations")
        return all_conversations

    def export_conversations_to_json(self, conversations: list[Conversation], output_path: Path) -> bool:
        """Export conversations to JSON format."""
        try:
            from datetime import datetime

            export_data = {
                "conversations": [],
                "metadata": {
                    "total_conversations": len(conversations),
                    "knowledge_sources": ["DSM-5", "Big Five"],
                    "generated_at": datetime.now().isoformat(),
                    "converter_version": "1.0"
                }
            }

            for conversation in conversations:
                # Convert conversation to dictionary manually to handle complex objects
                conv_dict = {
                    "id": conversation.id,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "meta": msg.meta
                        } for msg in conversation.messages
                    ],
                    "context": conversation.context,
                    "source": conversation.source,
                    "meta": conversation.meta
                }
                export_data["conversations"].append(conv_dict)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(conversations)} conversations to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export conversations: {e}")
            return False

    def get_statistics(self, conversations: list[Conversation]) -> dict[str, Any]:
        """Get statistics about generated conversations."""
        if not conversations:
            return {}

        stats = {
            "total_conversations": len(conversations),
            "knowledge_sources": {},
            "conversation_types": {},
            "average_messages_per_conversation": 0,
            "total_messages": 0
        }

        total_messages = 0

        for conversation in conversations:
            # Knowledge sources
            source = conversation.context.get("knowledge_source", "unknown")
            stats["knowledge_sources"][source] = stats["knowledge_sources"].get(source, 0) + 1

            # Conversation types
            conv_type = conversation.context.get("conversation_type", "unknown")
            stats["conversation_types"][conv_type] = stats["conversation_types"].get(conv_type, 0) + 1

            # Message count
            total_messages += len(conversation.messages)

        stats["total_messages"] = total_messages
        stats["average_messages_per_conversation"] = total_messages / len(conversations) if conversations else 0

        return stats
