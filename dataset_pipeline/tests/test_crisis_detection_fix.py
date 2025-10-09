#!/usr/bin/env python3
"""
Test script to validate crisis detection fixes.
"""

import logging
import sys
import unittest

# Disable logging for cleaner output
logging.getLogger("crisis_intervention_detector").setLevel(logging.CRITICAL)

from quality.crisis_intervention_detector import CrisisInterventionDetector


class TestModule(unittest.TestCase):
    def test_crisis_detection():
        """Test crisis detection with various scenarios."""
        detector = CrisisInterventionDetector()

        test_cases = [
            ("Suicide ideation", "I want to kill myself tonight"),
            ("Self harm", "I want to hurt myself"),
            ("Violence threat", "I am going to hurt him with a gun"),
            ("Substance overdose", "I took the whole bottle of pills"),
            ("Domestic violence", "He hit me and threatened to kill me"),
            ("Psychotic episode", "The voices are telling me to do things"),
            ("Mild depression", "I feel sad but I will be okay"),
            ("Normal conversation", "How are you doing today?")
        ]


        results = []
        for name, content in test_cases:
            test = {"id": f'test_{name.replace(" ", "_")}', "content": content}
            try:
                result = detector.detect_crisis(test)
                ", ".join([ct.value for ct in result.crisis_types]) if result.crisis_types else "none"
                results.append((name, result.crisis_level.value[0], result.confidence_score, len(result.crisis_types)))
            except Exception:
                results.append((name, "ERROR", 0.0, 0))

        sum(1 for _, level, _, _ in results if level == "emergency")
        sum(1 for _, level, _, _ in results if level == "critical")
        sum(1 for _, level, _, _ in results if level == "elevated")
        sum(1 for _, level, _, _ in results if level == "routine")
        sum(1 for _, level, _, _ in results if level == "ERROR")


        # Validate expected results
        expected_emergency = ["Suicide ideation", "Self harm", "Violence threat", "Substance overdose", "Domestic violence"]
        expected_routine = ["Mild depression", "Normal conversation"]

        success = True
        for name, level, _confidence, _ in results:
            if (name in expected_emergency and level != "emergency") or (name in expected_routine and level != "routine"):
                success = False

        return bool(success)

if __name__ == "__main__":
    success = test_crisis_detection()
    sys.exit(0 if success else 1)
