import unittest

from ai.core.pipelines.processing.emotional_cartography import (
    EmotionalCartographer,
    EmotionalState,
    PlutchikEmotion,
)


class TestEmotionalCartography(unittest.TestCase):
    def setUp(self):
        self.cartographer = EmotionalCartographer()

    def test_identify_primary_dyads(self):
        # Joy + Trust = Love
        dyad = self.cartographer.identify_dyad(PlutchikEmotion.JOY, PlutchikEmotion.TRUST)
        self.assertEqual(dyad, "Love")

        # Anger + Disgust = Contempt
        dyad = self.cartographer.identify_dyad(PlutchikEmotion.ANGER, PlutchikEmotion.DISGUST)
        self.assertEqual(dyad, "Contempt")

    def test_identify_secondary_dyads(self):
        # Joy + Fear = Guilt
        dyad = self.cartographer.identify_dyad(PlutchikEmotion.JOY, PlutchikEmotion.FEAR)
        self.assertEqual(dyad, "Guilt")

        # Anticipation + Trust = Fatalism
        dyad = self.cartographer.identify_dyad(PlutchikEmotion.ANTICIPATION, PlutchikEmotion.TRUST)
        self.assertEqual(dyad, "Fatalism")

    def test_identify_dyad_order_independence(self):
        # Trust + Joy should also be Love
        dyad = self.cartographer.identify_dyad(PlutchikEmotion.TRUST, PlutchikEmotion.JOY)
        self.assertEqual(dyad, "Love")

    def test_map_complex_state(self):
        states = [
            EmotionalState(PlutchikEmotion.JOY, 0.9),
            EmotionalState(PlutchikEmotion.TRUST, 0.8),
        ]

        result = self.cartographer.map_complex_state(states)
        self.assertEqual(result["primary_state"], "joy")
        self.assertIn("Love", result["complex_states"])

    def test_map_complex_state_empty(self):
        result = self.cartographer.map_complex_state([])
        self.assertEqual(result["primary_state"], "Neutral")


if __name__ == "__main__":
    unittest.main()
