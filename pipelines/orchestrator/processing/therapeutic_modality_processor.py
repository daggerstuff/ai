"""
Therapeutic modality processor for different therapy approaches.
Processes data for various therapeutic modalities (CBT, DBT, Psychodynamic, etc.).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


class TherapeuticModality(Enum):
    """Different therapeutic modalities."""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    PSYCHODYNAMIC = "psychodynamic_therapy"
    HUMANISTIC = "humanistic_therapy"
    GESTALT = "gestalt_therapy"
    EMDR = "emdr"
    ACT = "acceptance_commitment_therapy"
    MINDFULNESS = "mindfulness_based_therapy"
    FAMILY_SYSTEMS = "family_systems_therapy"
    SOLUTION_FOCUSED = "solution_focused_therapy"
    NARRATIVE = "narrative_therapy"
    EXISTENTIAL = "existential_therapy"


@dataclass
class ModalityCharacteristics:
    """Characteristics of a therapeutic modality."""
    modality: TherapeuticModality
    core_principles: list[str]
    key_techniques: list[str]
    therapeutic_focus: str
    typical_interventions: list[str]
    contraindications: list[str]
    evidence_base: str
    session_structure: dict[str, Any]


@dataclass
class ProcessedModalityData:
    """Result of modality-specific processing."""
    conversations: list[Conversation]
    modality_characteristics: ModalityCharacteristics
    technique_examples: dict[str, list[str]]
    quality_metrics: dict[str, float]
    processing_stats: dict[str, Any]


class TherapeuticModalityProcessor:
    """
    Processes data for different therapeutic modalities.

    Handles CBT, DBT, psychodynamic, humanistic, and other major
    therapeutic approaches with modality-specific processing.
    """

    def __init__(self):
        """Initialize the therapeutic modality processor."""
        self.logger = get_logger(__name__)

        # Define modality characteristics
        self.modality_definitions = self._initialize_modality_definitions()

        # Technique patterns for each modality
        self.technique_patterns = self._initialize_technique_patterns()

        # Quality thresholds
        self.quality_thresholds = {
            "modality_adherence": 0.8,
            "technique_accuracy": 0.85,
            "therapeutic_alliance": 0.8,
            "intervention_appropriateness": 0.9
        }

        self.logger.info("TherapeuticModalityProcessor initialized")

    def process_modality_data(self, data: list[dict[str, Any]], target_modality: TherapeuticModality) -> ProcessedModalityData:
        """
        Process data for a specific therapeutic modality.

        Args:
            data: Raw conversation/session data
            target_modality: The therapeutic modality to process for

        Returns:
            ProcessedModalityData with modality-specific conversations
        """
        self.logger.info(f"Processing data for modality: {target_modality.value}")

        modality_chars = self.modality_definitions[target_modality]
        conversations = []
        technique_examples = {}
        quality_scores = []

        for item in data:
            # Convert to conversation format
            conversation = self._convert_to_modality_conversation(item, target_modality)
            if conversation:
                # Assess modality adherence
                quality_score = self._assess_modality_quality(conversation, target_modality)

                if quality_score >= self.quality_thresholds["modality_adherence"]:
                    conversations.append(conversation)
                    quality_scores.append(quality_score)

                    # Extract technique examples
                    techniques = self._extract_techniques(conversation, target_modality)
                    for technique, examples in techniques.items():
                        if technique not in technique_examples:
                            technique_examples[technique] = []
                        technique_examples[technique].extend(examples)

        # Calculate quality metrics
        quality_metrics = {
            "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "modality_adherence": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "technique_coverage": len(technique_examples) / len(modality_chars.key_techniques),
            "conversation_count": len(conversations)
        }

        processing_stats = {
            "target_modality": target_modality.value,
            "processed_items": len(data),
            "accepted_conversations": len(conversations),
            "acceptance_rate": len(conversations) / len(data) if data else 0,
            "processed_at": datetime.now().isoformat()
        }

        self.logger.info(f"Processed {len(conversations)} conversations for {target_modality.value}")

        return ProcessedModalityData(
            conversations=conversations,
            modality_characteristics=modality_chars,
            technique_examples=technique_examples,
            quality_metrics=quality_metrics,
            processing_stats=processing_stats
        )

    def _initialize_modality_definitions(self) -> dict[TherapeuticModality, ModalityCharacteristics]:
        """Initialize definitions for each therapeutic modality."""
        return {
            TherapeuticModality.CBT: ModalityCharacteristics(
                modality=TherapeuticModality.CBT,
                core_principles=[
                    "Thoughts, feelings, and behaviors are interconnected",
                    "Present-focused and problem-solving oriented",
                    "Collaborative therapeutic relationship",
                    "Homework and skill practice"
                ],
                key_techniques=[
                    "cognitive restructuring", "behavioral activation", "exposure therapy",
                    "thought records", "behavioral experiments", "homework assignments"
                ],
                therapeutic_focus="Changing maladaptive thought patterns and behaviors",
                typical_interventions=[
                    "identify cognitive distortions", "challenge negative thoughts",
                    "behavioral scheduling", "graded exposure"
                ],
                contraindications=["severe psychosis", "active substance abuse"],
                evidence_base="Strong empirical support for anxiety, depression, PTSD",
                session_structure={
                    "agenda_setting": True,
                    "homework_review": True,
                    "skill_practice": True,
                    "between_session_tasks": True
                }
            ),

            TherapeuticModality.DBT: ModalityCharacteristics(
                modality=TherapeuticModality.DBT,
                core_principles=[
                    "Dialectical thinking and balance",
                    "Mindfulness and acceptance",
                    "Distress tolerance skills",
                    "Emotion regulation"
                ],
                key_techniques=[
                    "mindfulness skills", "distress tolerance", "emotion regulation",
                    "interpersonal effectiveness", "radical acceptance", "wise mind"
                ],
                therapeutic_focus="Managing intense emotions and improving relationships",
                typical_interventions=[
                    "skills training", "mindfulness practice", "crisis survival",
                    "emotion surfing"
                ],
                contraindications=["unwillingness to commit to treatment"],
                evidence_base="Strong support for borderline personality disorder",
                session_structure={
                    "skills_training": True,
                    "individual_therapy": True,
                    "phone_coaching": True,
                    "therapist_consultation": True
                }
            ),

            TherapeuticModality.PSYCHODYNAMIC: ModalityCharacteristics(
                modality=TherapeuticModality.PSYCHODYNAMIC,
                core_principles=[
                    "Unconscious processes influence behavior",
                    "Past experiences shape present relationships",
                    "Transference and countertransference",
                    "Defense mechanisms and resistance"
                ],
                key_techniques=[
                    "free association", "interpretation", "transference analysis",
                    "dream analysis", "defense analysis", "working through"
                ],
                therapeutic_focus="Understanding unconscious conflicts and patterns",
                typical_interventions=[
                    "explore childhood experiences", "analyze transference",
                    "interpret defenses", "work through resistance"
                ],
                contraindications=["severe cognitive impairment", "active psychosis"],
                evidence_base="Moderate support for personality disorders, depression",
                session_structure={
                    "free_association": True,
                    "interpretation": True,
                    "process_focus": True,
                    "long_term_treatment": True
                }
            ),

            TherapeuticModality.HUMANISTIC: ModalityCharacteristics(
                modality=TherapeuticModality.HUMANISTIC,
                core_principles=[
                    "Inherent human potential for growth",
                    "Client-centered approach",
                    "Unconditional positive regard",
                    "Empathy and genuineness"
                ],
                key_techniques=[
                    "active listening", "reflection", "empathy", "genuineness",
                    "unconditional positive regard", "here-and-now focus"
                ],
                therapeutic_focus="Facilitating self-actualization and personal growth",
                typical_interventions=[
                    "reflect feelings", "provide empathy", "facilitate self-exploration",
                    "support autonomy"
                ],
                contraindications=["need for structured intervention"],
                evidence_base="Moderate support across various conditions",
                session_structure={
                    "client_led": True,
                    "process_oriented": True,
                    "relationship_focus": True,
                    "non_directive": True
                }
            )
        }

    def _initialize_technique_patterns(self) -> dict[TherapeuticModality, dict[str, list[str]]]:
        """Initialize technique patterns for each modality."""
        return {
            TherapeuticModality.CBT: {
                "cognitive_restructuring": [
                    "what evidence supports this thought",
                    "is there another way to look at this",
                    "what would you tell a friend",
                    "thought record", "cognitive distortion"
                ],
                "behavioral_activation": [
                    "activity scheduling", "behavioral experiment",
                    "graded task assignment", "pleasant activities"
                ],
                "exposure_therapy": [
                    "gradual exposure", "hierarchy", "systematic desensitization",
                    "in vivo exposure", "imaginal exposure"
                ]
            },

            TherapeuticModality.DBT: {
                "mindfulness": [
                    "observe", "describe", "participate", "non-judgmentally",
                    "one-mindfully", "effectively", "wise mind"
                ],
                "distress_tolerance": [
                    "TIPP", "distract", "self-soothe", "improve the moment",
                    "radical acceptance", "distress tolerance skills"
                ],
                "emotion_regulation": [
                    "emotion surfing", "opposite action", "PLEASE skills",
                    "emotion regulation", "check the facts"
                ]
            },

            TherapeuticModality.PSYCHODYNAMIC: {
                "interpretation": [
                    "it seems like", "I wonder if", "perhaps",
                    "this reminds me of", "pattern", "unconscious"
                ],
                "transference": [
                    "relationship with me", "how you experience me",
                    "feelings toward me", "transference", "here and now"
                ]
            },

            TherapeuticModality.HUMANISTIC: {
                "reflection": [
                    "it sounds like", "you're feeling", "I hear",
                    "what I'm hearing is", "reflection", "empathy"
                ],
                "genuineness": [
                    "I'm wondering", "my sense is", "I notice",
                    "authentic", "genuine", "real"
                ]
            }
        }

    def _convert_to_modality_conversation(self, item: dict[str, Any], modality: TherapeuticModality) -> Conversation | None:
        """Convert data item to modality-specific conversation."""
        try:
            messages = []

            # Extract conversation content
            if "therapist" in item and "client" in item:
                messages.append(Message(
                    role="user",
                    content=str(item["client"]),
                    timestamp=datetime.now()
                ))
                messages.append(Message(
                    role="assistant",
                    content=str(item["therapist"]),
                    timestamp=datetime.now()
                ))
            elif "input" in item and "output" in item:
                messages.append(Message(
                    role="user",
                    content=str(item["input"]),
                    timestamp=datetime.now()
                ))
                messages.append(Message(
                    role="assistant",
                    content=str(item["output"]),
                    timestamp=datetime.now()
                ))
            else:
                return None

            if not messages:
                return None

            conversation_id = f"modality_{modality.value}_{hash(str(item)) % 100000}"

            return Conversation(
                id=conversation_id,
                messages=messages,
                title=f"{modality.value.replace('_', ' ').title()} Session",
                metadata={
                    "therapeutic_modality": modality.value,
                    "modality_specific": True,
                    "source": "modality_processor"
                },
                tags=["therapeutic_modality", modality.value, "processed"]
            )

        except Exception as e:
            self.logger.warning(f"Could not convert item to modality conversation: {e}")
            return None

    def _assess_modality_quality(self, conversation: Conversation, modality: TherapeuticModality) -> float:
        """Assess how well a conversation adheres to a specific modality."""
        try:
            scores = []

            # Get modality-specific patterns
            patterns = self.technique_patterns.get(modality, {})

            # Check for modality-specific language
            content = " ".join([msg.content.lower() for msg in conversation.messages])

            technique_matches = 0
            total_techniques = 0

            for _technique, keywords in patterns.items():
                total_techniques += 1
                if any(keyword.lower() in content for keyword in keywords):
                    technique_matches += 1

            if total_techniques > 0:
                technique_score = technique_matches / total_techniques
                scores.append(technique_score)

            # Check conversation structure
            if len(conversation.messages) >= 2:
                scores.append(0.8)  # Basic therapeutic structure

            # Check for therapeutic language
            therapeutic_indicators = [
                "feel", "think", "experience", "explore", "understand",
                "support", "help", "work together", "therapy", "therapeutic"
            ]

            indicator_count = sum(1 for indicator in therapeutic_indicators if indicator in content)
            if indicator_count >= 3:
                scores.append(0.9)
            elif indicator_count >= 1:
                scores.append(0.7)

            return sum(scores) / len(scores) if scores else 0.5

        except Exception as e:
            self.logger.warning(f"Could not assess modality quality: {e}")
            return 0.5

    def _extract_techniques(self, conversation: Conversation, modality: TherapeuticModality) -> dict[str, list[str]]:
        """Extract technique examples from a conversation."""
        techniques = {}
        patterns = self.technique_patterns.get(modality, {})

        " ".join([msg.content for msg in conversation.messages])

        for technique, keywords in patterns.items():
            examples = []
            for msg in conversation.messages:
                if msg.role == "assistant":  # Therapist messages
                    for keyword in keywords:
                        if keyword.lower() in msg.content.lower():
                            # Extract sentence containing the technique
                            sentences = msg.content.split(".")
                            for sentence in sentences:
                                if keyword.lower() in sentence.lower():
                                    examples.append(sentence.strip())
                                    break

            if examples:
                techniques[technique] = examples[:3]  # Limit to 3 examples

        return techniques

    def process_multiple_modalities(self, data: list[dict[str, Any]]) -> dict[TherapeuticModality, ProcessedModalityData]:
        """Process data for multiple therapeutic modalities."""
        results = {}

        for modality in TherapeuticModality:
            try:
                result = self.process_modality_data(data, modality)
                results[modality] = result
                self.logger.info(f"Successfully processed {modality.value}")
            except Exception as e:
                self.logger.error(f"Failed to process {modality.value}: {e}")

        return results

    def get_modality_comparison(self, results: dict[TherapeuticModality, ProcessedModalityData]) -> dict[str, Any]:
        """Compare results across different modalities."""
        comparison = {
            "modality_counts": {},
            "quality_comparison": {},
            "technique_coverage": {},
            "total_conversations": 0
        }

        for modality, data in results.items():
            modality_name = modality.value
            comparison["modality_counts"][modality_name] = len(data.conversations)
            comparison["quality_comparison"][modality_name] = data.quality_metrics.get("average_quality", 0)
            comparison["technique_coverage"][modality_name] = data.quality_metrics.get("technique_coverage", 0)
            comparison["total_conversations"] += len(data.conversations)

        return comparison


def validate_therapeutic_modality_processor():
    """Validate the TherapeuticModalityProcessor functionality."""
    try:
        processor = TherapeuticModalityProcessor()

        # Test basic functionality
        assert hasattr(processor, "process_modality_data")
        assert hasattr(processor, "modality_definitions")
        assert len(processor.modality_definitions) > 0

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # Run validation
    if validate_therapeutic_modality_processor():
        pass
    else:
        pass
