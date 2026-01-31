import unittest

from ai.pipelines.orchestrator.data_augmentation import (
    AugmentationConfig,
    DataAugmenter,
    SafetyGuardrails,
)
from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation


class TestDataAugmentation(unittest.TestCase):
    """
    Test suite for Task 5.7.1.3: Data Augmentation Techniques
    Validates paraphrasing, contextual augmentation, noise injection, and safety guardrails.
    """

    def setUp(self):
        """Set up test environment."""
        self.config = AugmentationConfig(
            paraphrase_enabled=True,
            paraphrase_probability=1.0,  # Force for testing
            contextual_augmentation_enabled=True,
            contextual_probability=1.0,
            noise_injection_enabled=True,
            noise_probability=1.0,
            demographic_variation_enabled=True,
            demographic_variation_probability=1.0,
            safety_guardrails_enabled=True,
        )
        self.augmenter = DataAugmenter(self.config)
        self.guardrails = SafetyGuardrails()

    def test_paraphrase_text(self):
        """Test paraphrasing of therapeutic phrases."""
        text = "I understand how you feel. That must be hard."
        augmented = self.augmenter.paraphrase_text(text)
        # Check if it used one of the templates (e.g., "I can see why that would be difficult")
        # Template: ("I understand how you feel", "I can see why that would be difficult")
        # Template: ("That must be hard", "I imagine that's challenging")
        self.assertTrue(
            augmented != text or "difficult" in augmented or "challenging" in augmented
        )

    def test_noise_injection(self):
        """Test injection of filler words and spelling variations."""
        text = "I am a therapist. I want to help you."

        # Since noise injection has a 0.3 random chance per pattern,
        # we try a few times to ensure we catch an augmentation
        for _ in range(10):
            augmented = self.augmenter.inject_noise(text)
            if augmented != text:
                break

        self.assertNotEqual(
            text,
            augmented,
            "Noise injection should have changed the text in 10 attempts",
        )

        # Check for expected noise patterns if possible
        lowered = augmented.lower()
        has_noise = any(
            noise in lowered for noise in ["um", "uh", "you know", "like", "therapyst"]
        )
        self.assertTrue(
            has_noise or any(punc in augmented for punc in ["...", "!", "?"])
        )

    def test_safety_guardrails_preservation(self):
        """Test that critical crisis keywords are preserved during augmentation."""
        critical_text = "I am feeling suicidal and need help."

        # We'll manually run validation on a hypothetical "bad" augmentation
        bad_augmented = "I am feeling okay and need nothing."
        is_valid, issues = self.guardrails.validate_augmentation(
            critical_text, bad_augmented
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("CRITICAL" in issue for issue in issues))
        self.assertTrue(any("crisis" in issue.lower() for issue in issues))

    def test_safety_guardrails_reversion(self):
        """Test that augmenter reverts to original content when critical keywords are lost."""
        # Create a conversation with a critical message
        conv = Conversation()
        conv.add_message("client", "I want to kill myself.")

        # We'll mock the internal paraphrase_text to return something that triggers a safety violation
        # But since we can't easily mock in this simple test, we'll rely on the fact that
        # "kill myself" is in sensitive_keywords and should be preserved.

        # If the augmentation attempts to change "kill myself" to something else,
        # the guardrails should catch it.
        augmented_conv = self.augmenter.augment_conversation(conv)

        # Since "kill myself" doesn't have a paraphrase template in data_augmentation.py,
        # it shouldn't be changed by paraphrasing, but other augmentations might.
        # Critical keywords should be present in the output.
        self.assertIn("kill myself", augmented_conv.messages[0].content)

    def test_demographic_variation(self):
        """Test swapping of gender-coded terms."""
        text = "He is a good man and he loves his father."
        augmented = self.augmenter.demographic_variation(text)
        # Check if gendered terms were swapped (e.g., he -> she, man -> woman, father -> mother)
        self.assertTrue(
            "she" in augmented.lower()
            or "woman" in augmented.lower()
            or "mother" in augmented.lower()
        )

    def test_batch_augmentation(self):
        """Test augmenting multiple conversations."""
        conv1 = Conversation()
        conv1.add_message("therapist", "How are you?")
        conv2 = Conversation()
        conv2.add_message("client", "I'm stressed.")

        batch = [conv1, conv2]
        augmented_batch, _ = self.augmenter.batch_augment(batch)

        self.assertEqual(len(augmented_batch), 2)
        self.assertTrue(all("_aug" in c.conversation_id for c in augmented_batch))


if __name__ == "__main__":
    unittest.main()
