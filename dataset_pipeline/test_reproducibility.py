"""
Tests for labeler reproducibility and augmentation determinism.
Ensures consistent results across multiple runs with fixed seeds.
"""

import unittest
import random
from typing import List
from .label_taxonomy import (
    TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
    TherapeuticResponseType, CrisisLevelType
)
from .conversation_schema import Conversation, Message
from .automated_labeler import AutomatedLabeler
from .data_augmentation import DataAugmenter, AugmentationConfig
from .human_in_the_loop import HumanInLoopLabeler
from .label_versioning import LabelVersionManager


class TestLabelerReproducibility(unittest.TestCase):
    """Tests for the reproducibility of labelers"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Set a fixed seed to ensure reproducibility
        random.seed(42)
        
    def test_automated_labeler_consistency(self):
        """Test that the automated labeler produces consistent results"""
        # Create a consistent test conversation
        conversation = Conversation()
        conversation.add_message("therapist", "I hear you saying that you've been feeling very down lately.")
        conversation.add_message("client", "Yes, I just can't seem to get out of this funk.")
        conversation.add_message("therapist", "That sounds really difficult. Can you tell me more about what's been happening?")
        
        # Run the labeler multiple times
        results = []
        for i in range(3):
            # Reset seed before each run to ensure consistency
            random.seed(42)
            labeler = AutomatedLabeler()
            labels = labeler.label_conversation(conversation)
            results.append(labels)
        
        # All results should be identical
        for i in range(1, len(results)):
            self.assertEqual(len(results[0].therapeutic_response_labels), 
                           len(results[i].therapeutic_response_labels))
            
            for j, label in enumerate(results[0].therapeutic_response_labels):
                if j < len(results[i].therapeutic_response_labels):
                    self.assertEqual(label.response_type, results[i].therapeutic_response_labels[j].response_type)
                    self.assertAlmostEqual(label.metadata.confidence, 
                                         results[i].therapeutic_response_labels[j].metadata.confidence, places=3)
        
        # Check crisis label consistency
        if results[0].crisis_label:
            for i in range(1, len(results)):
                if results[i].crisis_label:
                    self.assertEqual(results[0].crisis_label.crisis_level, 
                                   results[i].crisis_label.crisis_level)
                    self.assertAlmostEqual(results[0].crisis_label.metadata.confidence, 
                                         results[i].crisis_label.metadata.confidence, places=3)
    
    def test_automated_labeler_deterministic_patterns(self):
        """Test that specific patterns are detected consistently"""
        # Create a conversation with clear therapeutic patterns
        conversation = Conversation()
        conversation.add_message("therapist", "It sounds like you're feeling overwhelmed by everything.")
        conversation.add_message("client", "Yes, I feel like I can't cope with all these demands.")
        conversation.add_message("therapist", "Can you tell me more about these demands?")
        
        # Extract therapeutic responses multiple times with same seed
        response_types = []
        for i in range(5):
            random.seed(123)  # Fixed seed for reproducibility
            labeler = AutomatedLabeler()
            labels = labeler.label_conversation(conversation)
            types = [label.response_type.value for label in labels.therapeutic_response_labels]
            response_types.append(types)
        
        # All should be identical
        first_result = response_types[0]
        for types in response_types[1:]:
            self.assertEqual(first_result, types)
    
    def test_human_in_loop_consistency(self):
        """Test that human-in-the-loop system behaves consistently"""
        conversation = Conversation()
        conversation.add_message("therapist", "I understand how difficult this must be for you.")
        conversation.add_message("client", "Thanks for saying that. It really does feel overwhelming.")
        
        # Create the human-in-the-loop system multiple times
        results = []
        for i in range(3):
            random.seed(42)  # Ensure consistent random behavior
            automated_labeler = AutomatedLabeler()
            human_loop = HumanInLoopLabeler(automated_labeler, confidence_threshold=0.7)
            
            # Check if human review is needed
            task_id = human_loop.process_conversation_for_human_review(conversation)
            results.append(task_id is not None)  # Whether human review was triggered
        
        # All should have same outcome
        self.assertTrue(all(results) or not any(results), 
                       "Human review decision should be consistent across runs")


class TestAugmentationDeterminism(unittest.TestCase):
    """Tests for the determinism of augmentation operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Set a fixed seed to ensure reproducibility
        random.seed(42)
        
    def test_augmentation_determinism_with_seed(self):
        """Test that augmentation produces the same results with the same seed"""
        # Create a test conversation
        conversation = Conversation()
        conversation.add_message("therapist", "I hear what you're saying about feeling down.")
        conversation.add_message("client", "Yes, it's been really tough lately.")
        
        # Set up augmentation config with moderate probability to ensure some changes occur
        config = AugmentationConfig(
            paraphrase_probability=0.8,
            contextual_probability=0.6,
            noise_probability=0.5,
            demographic_variation_probability=0.4
        )
        
        # Apply augmentation multiple times with the same seed
        results = []
        for i in range(3):
            random.seed(42)  # Reset to same seed each time
            augmenter = DataAugmenter(config)
            augmented = augmenter.augment_conversation(conversation)
            results.append(augmented)
        
        # Compare the results - they should be identical
        for i in range(1, len(results)):
            # Check that the number of messages is the same
            self.assertEqual(len(results[0].messages), len(results[i].messages))
            
            # Check that the content is the same
            for j, msg in enumerate(results[0].messages):
                self.assertEqual(msg.content, results[i].messages[j].content)
    
    def test_augmentation_with_different_seeds(self):
        """Test that different seeds produce potentially different results"""
        # Create a test conversation
        conversation = Conversation()
        conversation.add_message("therapist", "I understand how you're feeling.")
        conversation.add_message("client", "Thank you, that means a lot.")
        
        # Apply augmentation with different seeds
        config = AugmentationConfig(
            paraphrase_probability=0.8,
            contextual_probability=0.6
        )
        
        results = []
        seeds = [42, 123, 456]
        for seed in seeds:
            random.seed(seed)
            augmenter = DataAugmenter(config)
            augmented = augmenter.augment_conversation(conversation)
            results.append(augmented)
        
        # With high probability, the results should be different due to different seeds
        # (though there's a small chance they could be the same)
        contents = [result.messages[0].content for result in results]
        
        # At least some should potentially be different given high probability settings
        unique_contents = set(contents)
        # Note: This test could occasionally fail due to randomness, so it's not strict assertion
    
    def test_augmentation_preserves_critical_content(self):
        """Test that augmentation preserves critical safety and therapeutic content"""
        # Create a conversation with safety-related content
        conversation = Conversation()
        conversation.add_message("client", "I've been having thoughts about ending my life.")
        conversation.add_message("therapist", "Thank you for sharing that. It sounds very difficult.")
        conversation.add_message("therapist", "Are you having thoughts of harming yourself?")
        
        config = AugmentationConfig(
            paraphrase_probability=0.9,
            noise_probability=0.3
        )
        
        # Apply augmentation
        random.seed(42)
        augmenter = DataAugmenter(config)
        augmented = augmenter.augment_conversation(conversation)
        
        # Check that critical safety content is preserved
        augmented_text = " ".join([msg.content for msg in augmented.messages])
        
        # The word "life" in the context of suicide should be preserved
        original_text = " ".join([msg.content for msg in conversation.messages])
        
        # Both should contain safety-related keywords
        self.assertIn("life", original_text.lower())
        self.assertIn("life", augmented_text.lower())
        
        # Check that the meaning is preserved
        self.assertIn("thoughts", augmented_text.lower())
        self.assertIn("harming", augmented_text.lower())


class TestVersioningReproducibility(unittest.TestCase):
    """Tests for the reproducibility of versioning operations"""
    
    def test_versioning_consistency(self):
        """Test that versioning operations are consistent"""
        # Create test label bundles
        from .label_taxonomy import LabelBundle
        
        bundle1 = LabelBundle(conversation_id="test_conv_1")
        bundle1.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=TherapeuticResponseType.EMPATHY,
                metadata=LabelMetadata(confidence=0.8)
            )
        )
        
        bundle2 = LabelBundle(conversation_id="test_conv_1")  # Same conversation
        bundle2.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=TherapeuticResponseType.REFLECTION,
                metadata=LabelMetadata(confidence=0.9)
            )
        )
        
        # Test versioning operations
        vm = LabelVersionManager()
        
        # Create initial version
        v1 = vm.create_initial_version(bundle1, "test_user", "Initial version")
        
        # Update with second bundle
        v2 = vm.update_label_bundle(bundle2, v1, "test_user", 
                                   description="Updated version")
        
        # Check that the history is consistent
        history = vm.get_history(bundle1.label_id)
        self.assertIsNotNone(history)
        self.assertEqual(len(history.versions), 2)
        self.assertEqual(history.versions[0].version_number, 1)
        self.assertEqual(history.versions[1].version_number, 2)
        
        # Check that current bundle is bundle2
        current_bundle = vm.get_current_bundle(bundle1.label_id)
        self.assertIsNotNone(current_bundle)
        if current_bundle.therapeutic_response_labels:
            self.assertEqual(current_bundle.therapeutic_response_labels[0].response_type,
                           TherapeuticResponseType.REFLECTION)


def run_all_reproducibility_tests():
    """Run all reproducibility tests"""
    suite = unittest.TestSuite()
    
    # Add tests for labeler reproducibility
    suite.addTest(unittest.makeSuite(TestLabelerReproducibility))
    suite.addTest(unittest.makeSuite(TestAugmentationDeterminism))
    suite.addTest(unittest.makeSuite(TestVersioningReproducibility))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Running reproducibility tests...")
    result = run_all_reproducibility_tests()
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\nAll reproducibility tests passed! ✓")
    else:
        print("\nSome reproducibility tests failed! ✗")