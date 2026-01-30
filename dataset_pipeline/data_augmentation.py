"""
Data augmentation techniques for mental health conversation datasets.
Implements paraphrasing, contextual augmentation, and noise injection
with appropriate guardrails.
"""

import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .conversation_schema import Conversation, Message
from .label_taxonomy import LabelBundle

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Types of augmentation techniques"""

    PARAPHRASE = "paraphrase"
    CONTEXTUAL = "contextual"
    NOISE_INJECTION = "noise_injection"
    SYNTHETIC_EXPANSION = "synthetic_expansion"
    DEMOGRAPHIC_VARIATION = "demographic_variation"


@dataclass
class AugmentationConfig:
    """Configuration for augmentation techniques"""

    paraphrase_enabled: bool = True
    paraphrase_probability: float = 0.3
    contextual_augmentation_enabled: bool = True
    contextual_probability: float = 0.2
    noise_injection_enabled: bool = True
    noise_probability: float = 0.1
    max_noise_level: float = 0.05  # Maximum 5% noise
    preserve_sensitive_content: bool = True
    maintain_therapeutic_context: bool = True
    demographic_variation_enabled: bool = True
    demographic_variation_probability: float = 0.15
    safety_guardrails_enabled: bool = True


class SafetyGuardrails:
    """Safety guardrails for augmentation operations"""

    def __init__(self):
        # Keywords that should never be modified or removed
        self.sensitive_keywords = {
            "crisis": [
                r"(?i)\bsuicid",
                r"(?i)\bkill\s+myself",
                r"(?i)\bend\s+life",
                r"(?i)\bcan\'t\s+go\s+on",
                r"(?i)\bharm\s+myself",
            ],
            "safety": [
                r"(?i)\bhelp.*line",
                r"(?i)\bcrisis.*text",
                r"(?i)\bemergency.*number",
                r"(?i)\bcall.*911",
                r"(?i)\bemergency.*services",
            ],
            "therapeutic": [
                r"(?i)\btherapist",
                r"(?i)\bcounselor",
                r"(?i)\btherapist",
                r"(?i)\bmental.*health",
                r"(?i)\btherapy",
            ],
        }

        # Context-preserving phrases that should be handled carefully
        self.context_phrases = [
            r"(?i)\bi.*understand",
            r"(?i)\byou.*mentioned",
            r"(?i)\bit.*sounds.*like",
            r"(?i)\bhow.*can.*i.*help",
            r"(?i)\bwhat.*brings.*you.*here",
        ]

    def validate_augmentation(
        self, original_text: str, augmented_text: str
    ) -> Tuple[bool, List[str]]:
        """Validate that augmentation preserves safety and context"""
        issues = []

        # Check for crisis keyword preservation
        for category, patterns in self.sensitive_keywords.items():
            for pattern in patterns:
                if re.search(pattern, original_text):
                    if not re.search(pattern, augmented_text) and category in [
                        "crisis",
                        "safety",
                    ]:
                        issues.append(
                            f"CRITICAL: {category} keyword lost in augmentation"
                        )

        # Check for major therapeutic context preservation
        # context_preserved = True
        for phrase_pattern in self.context_phrases:
            if re.search(phrase_pattern, original_text) and not re.search(
                phrase_pattern, augmented_text
            ):
                # This may not be an issue depending on augmentation type
                if (
                    "therapist" in original_text.lower()
                    and "therapist" not in augmented_text.lower()
                ):
                    issues.append("THERAPEUTIC_CONTEXT: Therapist reference lost")

        # Check semantic similarity (very basic check)
        original_tokens = set(re.findall(r"\b\w+\b", original_text.lower()))
        augmented_tokens = set(re.findall(r"\b\w+\b", augmented_text.lower()))

        # If too much content was changed, flag it
        if len(original_tokens) > 0:
            overlap = len(original_tokens.intersection(augmented_tokens))
            similarity = overlap / len(original_tokens)
            if similarity < 0.3:  # Less than 30% overlap
                issues.append(
                    f"CONTENT_INTEGRITY: Low semantic overlap ({similarity:.2f})"
                )

        is_valid = len(issues) == 0
        return is_valid, issues


class DataAugmenter:
    """Main class for implementing data augmentation techniques"""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.guardrails = SafetyGuardrails()
        self.paraphrase_templates = self._initialize_paraphrase_templates()
        self.contextual_variations = self._initialize_contextual_variations()
        self.demographic_variations = self._initialize_demographic_variations()
        self.noise_patterns = self._initialize_noise_patterns()

    def _initialize_paraphrase_templates(self) -> Dict[str, List[str]]:
        """Initialize paraphrasing templates for therapeutic conversations"""
        return {
            # Empathy expressions
            "empathy": [
                ("I understand how you feel", "I can see why that would be difficult"),
                ("That must be hard", "I imagine that's challenging"),
                ("I hear you", "You're making sense to me"),
                ("That sounds difficult", "That seems really tough"),
                ("I can only imagine", "I'm sure that's not easy"),
            ],
            # Reflection phrases
            "reflection": [
                ("So you're saying", "It sounds like"),
                ("What I'm hearing is", "From what you've shared"),
                ("You mentioned", "You said"),
                ("It seems like", "It appears that"),
                ("Right now you're feeling", "Currently you feel"),
            ],
            # Probing questions
            "probing": [
                (
                    "Can you tell me more about",
                    "What else comes up when you think about",
                ),
                ("How did that make you feel", "What was your experience with"),
                ("What happened next", "Then what occurred"),
                ("Can you describe", "Could you walk me through"),
                ("Tell me about", "Share with me"),
            ],
            # Validation phrases
            "validation": [
                ("That's a normal response", "It's common to feel that way"),
                ("You're not alone", "Many people experience this"),
                ("That takes courage", "It's brave of you to share this"),
                ("You have every right to feel", "Your feelings are completely valid"),
                ("Thank you for sharing", "I appreciate your openness"),
            ],
        }

    def _initialize_contextual_variations(self) -> Dict[str, List[str]]:
        """Initialize contextual variation templates"""
        return {
            "setting": [
                ("in our session", "today"),
                ("during our time together", "when we talk"),
                ("in therapy", "in our conversations"),
            ],
            "time_ref": [
                ("recently", "lately"),
                ("in the past", "before"),
                ("currently", "right now"),
                ("sometimes", "at times"),
                ("often", "frequently"),
            ],
            "emotional_intensity": [
                ("very", "really"),
                ("quite", "pretty"),
                ("extremely", "incredibly"),
                ("somewhat", "a bit"),
                ("fairly", "relatively"),
            ],
        }

    def _initialize_demographic_variations(self) -> Dict[str, List[Dict[str, str]]]:
        """Initialize demographic variation mappings"""
        return {
            "age": [
                {
                    "old": ["young", "teen", "adolescent"],
                    "new": ["adult", "mature", "experienced"],
                },
                {
                    "old": ["elderly", "senior", "older"],
                    "new": ["middle-aged", "adult", "contemporary"],
                },
            ],
            "gender": [
                {"old": ["he", "him", "his"], "new": ["she", "her", "hers"]},
                {"old": ["she", "her", "hers"], "new": ["they", "them", "theirs"]},
                {"old": ["male", "man"], "new": ["female", "woman"]},
                {"old": ["father", "dad"], "new": ["mother", "mom"]},
            ],
        }

    def _initialize_noise_patterns(self) -> List[Dict[str, Any]]:
        """Initialize noise injection patterns"""
        return [
            # Minor spelling variations (for robustness)
            {
                "type": "spelling",
                "patterns": [
                    ("therapist", "therapyst"),  # intentional misspelling
                    ("anxious", "anxius"),
                    ("depression", "deppression"),
                ],
            },
            # Filler words/phrases
            {
                "type": "filler",
                "patterns": [
                    ("", " um "),
                    ("", " uh "),
                    ("", " you know "),
                    ("", " like "),
                ],
            },
            # Punctuation variations
            {
                "type": "punctuation",
                "patterns": [(".", "?"), (".", "!"), (".", "..."), (",", ";")],
            },
        ]

    def paraphrase_text(self, text: str) -> str:
        """Apply paraphrasing to a text while preserving meaning"""
        if (
            not self.config.paraphrase_enabled
            or random.random() > self.config.paraphrase_probability
        ):
            return text

        augmented_text = text

        # Apply paraphrasing based on detected intent
        for category, templates in self.paraphrase_templates.items():
            for original, replacement in templates:
                # Only replace if the original phrase exists in the text
                if re.search(re.escape(original), augmented_text, re.IGNORECASE):
                    # Preserve capitalization
                    if original.lower().startswith(
                        augmented_text[: len(original)].lower()
                    ):
                        replacement = replacement.capitalize()
                    augmented_text = re.sub(
                        re.escape(original),
                        replacement,
                        augmented_text,
                        flags=re.IGNORECASE,
                        count=1,
                    )

        return augmented_text

    def contextual_augmentation(self, text: str) -> str:
        """Apply contextual variations to enrich the conversation"""
        if (
            not self.config.contextual_augmentation_enabled
            or random.random() > self.config.contextual_probability
        ):
            return text

        augmented_text = text

        # Apply contextual variations
        for category, variations in self.contextual_variations.items():
            for old_phrase, new_phrase in variations:
                if random.random() < 0.5:  # Randomly choose to apply or not
                    augmented_text = re.sub(
                        re.escape(old_phrase),
                        new_phrase,
                        augmented_text,
                        flags=re.IGNORECASE,
                    )

        return augmented_text

    def demographic_variation(self, text: str) -> str:
        """Apply demographic variations to increase diversity"""
        if (
            not self.config.demographic_variation_enabled
            or random.random() > self.config.demographic_variation_probability
        ):
            return text

        augmented_text = text

        # Apply demographic variations
        for category, variations_list in self.demographic_variations.items():
            for variation in variations_list:
                old_words = variation["old"]
                new_words = variation["new"]

                # Randomly select a pair of words to swap
                if len(old_words) == len(new_words):
                    for old_word, new_word in zip(old_words, new_words):
                        # Only apply if the old word exists in the text
                        if re.search(
                            r"\b" + re.escape(old_word) + r"\b",
                            augmented_text,
                            re.IGNORECASE,
                        ):
                            augmented_text = re.sub(
                                r"\b" + re.escape(old_word) + r"\b",
                                new_word,
                                augmented_text,
                                flags=re.IGNORECASE,
                            )

        return augmented_text

    def inject_noise(self, text: str) -> str:
        """Inject small amounts of noise for robustness"""
        if (
            not self.config.noise_injection_enabled
            or random.random() > self.config.noise_probability
        ):
            return text

        augmented_text = text

        # Apply noise injection based on patterns
        for noise_pattern in self.noise_patterns:
            if random.random() < 0.3:  # Random chance to apply each noise type
                for old, new in noise_pattern["patterns"]:
                    if old == "":  # Filler words
                        # Insert randomly in the text
                        words = augmented_text.split()
                        if len(words) > 1:
                            insert_pos = random.randint(1, len(words) - 1)
                            words.insert(insert_pos, new.strip())
                            augmented_text = " ".join(words)
                    else:
                        # Replace existing text
                        if old in augmented_text:
                            augmented_text = augmented_text.replace(old, new, 1)

        return augmented_text

    def augment_conversation(self, conversation: Conversation) -> Conversation:
        """Apply augmentation to an entire conversation"""
        if not self.config.safety_guardrails_enabled:
            logger.warning(
                "Safety guardrails are disabled. This is not recommended for production use."
            )

        augmented_conversation = Conversation(
            conversation_id=conversation.conversation_id + "_aug",
            source=conversation.source,
            metadata=conversation.metadata.copy() if conversation.metadata else {},
        )

        for message in conversation.messages:
            augmented_content = message.content

            # Apply augmentation techniques in sequence with safety checks
            original_content = augmented_content

            # Apply paraphrasing
            augmented_content = self.paraphrase_text(augmented_content)

            # Apply contextual augmentation
            augmented_content = self.contextual_augmentation(augmented_content)

            # Apply demographic variation
            augmented_content = self.demographic_variation(augmented_content)

            # Apply noise injection
            augmented_content = self.inject_noise(augmented_content)

            # Perform safety validation
            if self.config.safety_guardrails_enabled:
                is_valid, issues = self.guardrails.validate_augmentation(
                    original_content, augmented_content
                )
                if not is_valid:
                    logger.warning(f"Safety issues detected in augmentation: {issues}")
                    # If critical issues, revert to original
                    critical_issues = [issue for issue in issues if "CRITICAL" in issue]
                    if critical_issues:
                        logger.warning(
                            "Reverting to original content due to critical safety issues"
                        )
                        augmented_content = original_content
                    else:
                        # Log non-critical issues but keep augmented content
                        for issue in issues:
                            logger.info(f"Non-critical augmentation issue: {issue}")

            # Create new message with augmented content
            augmented_message = Message(
                role=message.role,
                content=augmented_content,
                timestamp=message.timestamp,
                metadata=message.metadata.copy() if message.metadata else {},
            )
            augmented_conversation.messages.append(augmented_message)

        # Update metadata to track augmentation
        augmented_conversation.metadata["augmented"] = True
        augmented_conversation.metadata["augmentation_types"] = (
            self._get_applied_augmentation_types()
        )
        augmented_conversation.metadata["augmentation_timestamp"] = (
            augmented_conversation.updated_at
        )

        return augmented_conversation

    def _get_applied_augmentation_types(self) -> List[str]:
        """Get list of augmentation types that were applied"""
        types = []
        if self.config.paraphrase_enabled:
            types.append(AugmentationType.PARAPHRASE.value)
        if self.config.contextual_augmentation_enabled:
            types.append(AugmentationType.CONTEXTUAL.value)
        if self.config.noise_injection_enabled:
            types.append(AugmentationType.NOISE_INJECTION.value)
        if self.config.demographic_variation_enabled:
            types.append(AugmentationType.DEMOGRAPHIC_VARIATION.value)
        return types

    def augment_label_bundle(
        self, original_bundle: LabelBundle, augmented_conversation: Conversation
    ) -> LabelBundle:
        """
        Create an augmented label bundle corresponding to an augmented conversation
        """
        # For now, we'll keep the original labels but mark them as augmented
        augmented_bundle = LabelBundle(
            conversation_id=augmented_conversation.conversation_id,
            therapeutic_response_labels=original_bundle.therapeutic_response_labels.copy(),
            crisis_label=original_bundle.crisis_label,
            therapy_modality_label=original_bundle.therapy_modality_label,
            mental_health_condition_label=original_bundle.mental_health_condition_label,
            demographic_label=original_bundle.demographic_label,
            additional_labels=original_bundle.additional_labels.copy(),
        )

        # Update metadata to indicate this is from augmented data
        for label in augmented_bundle.therapeutic_response_labels:
            label.metadata.additional_context["source_augmented"] = True
        if augmented_bundle.crisis_label:
            augmented_bundle.crisis_label.metadata.additional_context[
                "source_augmented"
            ] = True
        if augmented_bundle.therapy_modality_label:
            augmented_bundle.therapy_modality_label.metadata.additional_context[
                "source_augmented"
            ] = True
        if augmented_bundle.mental_health_condition_label:
            augmented_bundle.mental_health_condition_label.metadata.additional_context[
                "source_augmented"
            ] = True
        if augmented_bundle.demographic_label:
            augmented_bundle.demographic_label.metadata.additional_context[
                "source_augmented"
            ] = True

        return augmented_bundle

    def batch_augment(
        self,
        conversations: List[Conversation],
        label_bundles: Optional[List[LabelBundle]] = None,
    ) -> Tuple[List[Conversation], Optional[List[LabelBundle]]]:
        """Apply augmentation to a batch of conversations"""
        augmented_conversations = []
        augmented_bundles = [] if label_bundles else None

        for i, conversation in enumerate(conversations):
            augmented_conv = self.augment_conversation(conversation)
            augmented_conversations.append(augmented_conv)

            if label_bundles and i < len(label_bundles):
                augmented_bundle = self.augment_label_bundle(
                    label_bundles[i], augmented_conv
                )
                if augmented_bundles is not None:
                    augmented_bundles.append(augmented_bundle)

        return augmented_conversations, augmented_bundles


class AugmentationQualityController:
    """Quality controller for data augmentation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_augmentation_quality(
        self, original_text: str, augmented_text: str
    ) -> Dict[str, float]:
        """Calculate quality metrics for augmentation"""
        metrics = {}

        # Calculate semantic preservation (simple overlap)
        orig_tokens = set(re.findall(r"\b\w+\b", original_text.lower()))
        aug_tokens = set(re.findall(r"\b\w+\b", augmented_text.lower()))

        if orig_tokens:
            overlap = len(orig_tokens.intersection(aug_tokens))
            semantic_preservation = overlap / len(orig_tokens)
            metrics["semantic_preservation"] = semantic_preservation
        else:
            metrics["semantic_preservation"] = 1.0  # Both empty

        # Calculate length preservation
        orig_len = len(original_text)
        aug_len = len(augmented_text)
        if orig_len > 0:
            length_change = abs(aug_len - orig_len) / orig_len
            metrics["length_preservation"] = 1.0 - min(length_change, 1.0)
        else:
            metrics["length_preservation"] = 1.0

        # Calculate diversity introduced (new unique tokens)
        new_tokens = aug_tokens - orig_tokens
        if aug_tokens:
            diversity_ratio = len(new_tokens) / len(aug_tokens)
            metrics["diversity_introduced"] = diversity_ratio
        else:
            metrics["diversity_introduced"] = 0.0

        return metrics

    def validate_augmentation_batch(
        self, original_convs: List[Conversation], augmented_convs: List[Conversation]
    ) -> Dict[str, Any]:
        """Validate quality of a batch of augmented conversations"""
        if len(original_convs) != len(augmented_convs):
            raise ValueError(
                "Original and augmented conversation lists must have the same length"
            )

        all_metrics = {
            "semantic_preservation_avg": 0.0,
            "length_preservation_avg": 0.0,
            "diversity_introduced_avg": 0.0,
            "total_conversations": len(original_convs),
            "quality_issues": [],
        }

        total_semantic = 0
        total_length = 0
        total_diversity = 0

        for orig_conv, aug_conv in zip(original_convs, augmented_convs):
            # Validate each message in the conversation
            for orig_msg, aug_msg in zip(orig_conv.messages, aug_conv.messages):
                metrics = self.calculate_augmentation_quality(
                    orig_msg.content, aug_msg.content
                )

                total_semantic += metrics["semantic_preservation"]
                total_length += metrics["length_preservation"]
                total_diversity += metrics["diversity_introduced"]

                # Check for quality issues
                if metrics["semantic_preservation"] < 0.5:
                    all_metrics["quality_issues"].append(
                        f"Low semantic preservation ({metrics['semantic_preservation']:.2f}) "
                        f"in conversation {orig_conv.conversation_id}"
                    )
                if metrics["length_preservation"] < 0.3:
                    all_metrics["quality_issues"].append(
                        f"High length change ({1 - metrics['length_preservation']:.2f}) "
                        f"in conversation {orig_conv.conversation_id}"
                    )

        if len(original_convs) > 0:
            all_metrics["semantic_preservation_avg"] = total_semantic / len(
                original_convs
            )
            all_metrics["length_preservation_avg"] = total_length / len(original_convs)
            all_metrics["diversity_introduced_avg"] = total_diversity / len(
                original_convs
            )

        return all_metrics


def create_default_augmenter(
    config: Optional[AugmentationConfig] = None,
) -> DataAugmenter:
    """Create a default data augmenter with standard configuration"""
    return DataAugmenter(config)


# Example usage
def test_data_augmentation():
    """Test the data augmentation system"""
    from .conversation_schema import Conversation

    # Create a test conversation
    conversation = Conversation()
    conversation.add_message(
        "therapist",
        (
            "I understand how you're feeling. It sounds like you've been "
            "going through a really difficult time lately."
        ),
    )
    conversation.add_message(
        "client",
        "Yes, I just can't seem to get out of this rut. I feel hopeless most days.",
    )
    conversation.add_message(
        "therapist",
        (
            "That must be incredibly hard. Can you tell me more about what's "
            "been happening?"
        ),
    )

    # Create augmenter with default config
    augmenter = create_default_augmenter()

    print("Original conversation:")
    for i, msg in enumerate(conversation.messages):
        print(f"  {msg.role}: {msg.content}")

    # Apply augmentation
    augmented_conversation = augmenter.augment_conversation(conversation)

    print("\nAugmented conversation:")
    for i, msg in enumerate(augmented_conversation.messages):
        print(f"  {msg.role}: {msg.content}")

    # Test quality controller
    controller = AugmentationQualityController()
    quality_metrics = controller.validate_augmentation_batch(
        [conversation], [augmented_conversation]
    )
    print(f"\nQuality metrics: {quality_metrics}")

    # Test batch augmentation
    conversations = [conversation] * 3
    augmented_batch, _ = augmenter.batch_augment(conversations)
    print(
        f"\nBatch augmentation: {len(conversations)} conversations -> "
        f"{len(augmented_batch)} augmented conversations"
    )


if __name__ == "__main__":
    test_data_augmentation()
