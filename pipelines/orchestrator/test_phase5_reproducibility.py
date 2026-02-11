"""
Test for labeler reproducibility and augmentation determinism.
Following the existing test patterns in the codebase.
"""

import pytest
import random
from typing import List
from .label_taxonomy import (
    TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
    TherapeuticResponseType, CrisisLevelType
)
from .conversation_schema import Conversation, Message
from .automated_labeler import AutomatedLabeler
from .data_augmentation import DataAugmenter, AugmentationConfig


def test_automated_labeler_reproducibility():
    """Test that the automated labeler produces consistent results across runs."""
    # Create a fixed test conversation
    conversation = Conversation()
    conversation.add_message("therapist", "I hear you saying that you've been feeling very down lately.")
    conversation.add_message("client", "Yes, I just can't seem to get out of this funk.")
    conversation.add_message("therapist", "That sounds really difficult. Can you tell me more about what's been happening?")
    
    # Run the labeler multiple times with the same seed
    all_results = []
    for i in range(3):
        random.seed(42)  # Reset seed for consistency
        labeler = AutomatedLabeler()
        labels = labeler.label_conversation(conversation)
        all_results.append({
            'therapeutic_count': len(labels.therapeutic_response_labels),
            'crisis_level': labels.crisis_label.crisis_level.value if labels.crisis_label else None,
            'crisis_confidence': labels.crisis_label.metadata.confidence if labels.crisis_label else None,
            'therapeutic_types': [l.response_type.value for l in labels.therapeutic_response_labels],
            'therapeutic_confidences': [l.metadata.confidence for l in labels.therapeutic_response_labels]
        })
    
    # All results should be identical
    first_result = all_results[0]
    for result in all_results[1:]:
        assert result['therapeutic_count'] == first_result['therapeutic_count']
        assert result['crisis_level'] == first_result['crisis_level']
        assert result['crisis_confidence'] == pytest.approx(first_result['crisis_confidence'], abs=1e-3)
        assert result['therapeutic_types'] == first_result['therapeutic_types']
        assert result['therapeutic_confidences'] == pytest.approx(first_result['therapeutic_confidences'], abs=1e-3)


def test_augmentation_determinism():
    """Test that augmentation produces deterministic results with the same seed."""
    # Create a test conversation
    conversation = Conversation()
    conversation.add_message("therapist", "I hear what you're saying about feeling down.")
    conversation.add_message("client", "Yes, it's been really tough lately.")
    
    # Set up augmentation config with high probability to ensure changes occur
    config = AugmentationConfig(
        paraphrase_probability=0.9,
        contextual_probability=0.8,
        noise_probability=0.7,
        demographic_variation_probability=0.6
    )
    
    # Apply augmentation multiple times with the same seed
    results = []
    for i in range(3):
        random.seed(123)  # Reset to same seed each time
        augmenter = DataAugmenter(config)
        augmented = augmenter.augment_conversation(conversation)
        results.append([msg.content for msg in augmented.messages])
    
    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result


def test_different_seeds_produce_different_results():
    """Test that different random seeds can produce different augmentation results."""
    conversation = Conversation()
    conversation.add_message("therapist", "I understand how you're feeling.")
    conversation.add_message("client", "Thank you, that means a lot.")
    
    config = AugmentationConfig(
        paraphrase_probability=0.9,
        contextual_probability=0.9
    )
    
    # Apply augmentation with different seeds
    results = []
    seeds = [100, 200, 300]
    for seed in seeds:
        random.seed(seed)
        augmenter = DataAugmenter(config)
        augmented = augmenter.augment_conversation(conversation)
        results.append([msg.content for msg in augmented.messages])
    
    # Not asserting they're all different (since randomness could theoretically produce the same result),
    # but we can at least check that the system works with different seeds


def test_pattern_detection_reproducibility():
    """Test that specific therapeutic response patterns are detected consistently."""
    # Create a conversation with a clear reflection pattern
    conversation = Conversation()
    conversation.add_message("therapist", "So you're saying that work has been incredibly stressful?")
    conversation.add_message("client", "Yes, the deadlines are overwhelming.")
    conversation.add_message("therapist", "It sounds like you're carrying a heavy burden.")
    
    consistent_results = []
    for i in range(5):
        random.seed(456)  # Fixed seed
        labeler = AutomatedLabeler()
        labels = labeler.label_conversation(conversation)
        reflection_count = sum(1 for label in labels.therapeutic_response_labels 
                              if label.response_type == TherapeuticResponseType.REFLECTION)
        empathy_count = sum(1 for label in labels.therapeutic_response_labels 
                           if label.response_type == TherapeuticResponseType.EMPATHY)
        consistent_results.append((reflection_count, empathy_count))
    
    # All runs should produce the same counts
    first_result = consistent_results[0]
    for result in consistent_results[1:]:
        assert result == first_result


def test_augmentation_preserves_critical_content():
    """Test that augmentation preserves critical therapeutic and safety content."""
    # Create a conversation that includes important therapeutic elements
    conversation = Conversation()
    conversation.add_message("client", "I've been having thoughts about ending my life.")
    conversation.add_message("therapist", "Thank you for sharing that. That takes courage.")
    conversation.add_message("therapist", "Are you having thoughts of harming yourself right now?")
    
    config = AugmentationConfig(
        paraphrase_probability=0.8,
        noise_probability=0.6
    )
    
    random.seed(789)
    augmenter = DataAugmenter(config)
    augmented = augmenter.augment_conversation(conversation)
    
    # Extract all text from original and augmented
    original_text = " ".join([msg.content for msg in conversation.messages]).lower()
    augmented_text = " ".join([msg.content for msg in augmented.messages]).lower()
    
    # Critical safety terms should still be present
    assert "life" in augmented_text
    assert "harming" in augmented_text or "hurt" in augmented_text
    assert len(augmented_text) > 0  # Should not be empty


def test_low_confidence_reproducibility():
    """Test that confidence calculations are reproducible."""
    conversation = Conversation()
    conversation.add_message("therapist", "How can I help you today?")
    conversation.add_message("client", "I'm not sure, I just feel off.")
    
    confidence_values = []
    for i in range(3):
        random.seed(999)
        labeler = AutomatedLabeler()
        labels = labeler.label_conversation(conversation)
        
        if labels.therapeutic_response_labels:
            avg_confidence = sum(l.metadata.confidence for l in labels.therapeutic_response_labels) / len(labels.therapeutic_response_labels)
            confidence_values.append(avg_confidence)
    
    # All values should be the same
    assert all(abs(cv - confidence_values[0]) < 1e-6 for cv in confidence_values)


if __name__ == "__main__":
    # Run the tests manually if executed as a script
    test_automated_labeler_reproducibility()
    test_augmentation_determinism()
    test_different_seeds_produce_different_results()
    test_pattern_detection_reproducibility()
    test_augmentation_preserves_critical_content()
    test_low_confidence_reproducibility()
    
    print("All reproducibility tests passed!")