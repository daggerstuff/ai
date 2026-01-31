#!/usr/bin/env python3
"""
Comprehensive Crisis Test Suite - Testing all 10 crisis types with realistic scenarios.
This test suite validates the crisis detection system against real-world crisis scenarios
to ensure accurate detection, proper classification, and appropriate escalation.
"""

import logging
import sys

from .typing import Dict, List

# Configure logging for test output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from .crisis_intervention_detector import CrisisInterventionDetector, CrisisType


class ComprehensiveCrisisTestSuite:
    """Comprehensive test suite for all crisis types with realistic scenarios."""

    def __init__(self):
        self.detector = CrisisInterventionDetector()
        self.test_results = []
        self.failed_tests = []

    def run_all_tests(self):
        """Run comprehensive tests for all 10 crisis types."""

        # Test each crisis type
        self.test_suicide_ideation()
        self.test_self_harm()
        self.test_violence_threats()
        self.test_psychotic_episodes()
        self.test_substance_overdose()
        self.test_domestic_violence()
        self.test_child_abuse()
        self.test_severe_depression()
        self.test_panic_attacks()
        self.test_eating_disorder_crisis()

        # Test edge cases and combinations
        self.test_multiple_crisis_types()
        self.test_false_positives()
        self.test_borderline_cases()

        # Generate comprehensive report
        self.generate_test_report()

        return len(self.failed_tests) == 0

    def test_suicide_ideation(self):
        """Test suicide ideation detection with various scenarios."""

        test_cases = [
            # Emergency level cases
            {
                "scenario": "Explicit intent with method",
                "content": "I want to kill myself tonight. I have pills and I know how many to take.",
                "expected_level": "emergency",
                "expected_confidence": 0.9,
                "expected_indicators": ["explicit_intent", "method_mention", "timeline_specific"]
            },
            {
                "scenario": "Goodbye message",
                "content": "This is goodbye everyone. I won't be here tomorrow. I'm sorry for everything.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["goodbye_messages", "timeline_specific"]
            },
            {
                "scenario": "Giving away possessions",
                "content": "I'm giving away all my things. I don't need them anymore. Take care of my dog.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["giving_away"]
            },
            # Critical level cases
            {
                "scenario": "Suicide ideation with plan",
                "content": "I keep thinking about killing myself. I have a plan but haven't decided when.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["ideation", "plan_details"]
            },
            {
                "scenario": "Hopelessness with method access",
                "content": "There's no point in living anymore. I have access to my dad's gun.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["hopelessness", "method_mention"]
            },
            # Elevated level cases
            {
                "scenario": "Passive ideation",
                "content": "Sometimes I think about dying. I wonder what it would be like to not exist.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["ideation"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.SUICIDE_IDEATION, test_cases)

    def test_self_harm(self):
        """Test self-harm detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Escalating cutting behavior",
                "content": "I need to cut deeper this time. The shallow cuts aren't helping anymore.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["escalation", "cutting_behavior"]
            },
            {
                "scenario": "Burning with specific method",
                "content": "I want to burn myself with a cigarette. I need to feel the pain.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["burning"]
            },
            {
                "scenario": "Self-hitting behavior",
                "content": "I keep hitting myself in the head when I get angry. I can't stop.",
                "expected_level": "elevated",
                "expected_confidence": 0.6,
                "expected_indicators": ["hitting"]
            },
            {
                "scenario": "Urges to self-harm",
                "content": "I have this overwhelming urge to hurt myself. I don't know why.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["urges"]
            },
            {
                "scenario": "Tools for self-harm",
                "content": "I found a knife in the kitchen. I keep thinking about using it to cut myself.",
                "expected_level": "critical",
                "expected_confidence": 0.6,
                "expected_indicators": ["tools_mention"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.SELF_HARM, test_cases)

    def test_violence_threats(self):
        """Test violence threat detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Explicit threat with weapon",
                "content": "I'm going to kill him. I have a gun and I know where he lives.",
                "expected_level": "emergency",
                "expected_confidence": 0.9,
                "expected_indicators": ["threat_explicit", "weapon_mention", "violence_planning"]
            },
            {
                "scenario": "Stalking behavior",
                "content": "I've been following her for weeks. I know her schedule. She'll pay for what she did.",
                "expected_level": "critical",
                "expected_confidence": 0.8,
                "expected_indicators": ["violence_planning", "specific_target"]
            },
            {
                "scenario": "Violence fantasy",
                "content": "I keep imagining hurting my boss. I fantasize about beating him up.",
                "expected_level": "elevated",
                "expected_confidence": 0.6,
                "expected_indicators": ["violence_fantasy"]
            },
            {
                "scenario": "Anger escalation",
                "content": "I'm so angry I can't control myself. I feel like I'm going to explode.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["anger_escalation"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.VIOLENCE_THREAT, test_cases)

    def test_psychotic_episodes(self):
        """Test psychotic episode detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Command hallucinations",
                "content": "The voices are telling me to hurt people. They won't stop commanding me.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["command_voices", "hallucinations"]
            },
            {
                "scenario": "Auditory hallucinations",
                "content": "I keep hearing voices that aren't there. They're talking about me.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["hallucinations"]
            },
            {
                "scenario": "Paranoid delusions",
                "content": "They're watching me through the cameras. Everyone is part of the conspiracy.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["delusions", "paranoia"]
            },
            {
                "scenario": "Reality distortion",
                "content": "Nothing feels real anymore. I think I'm living in a simulation.",
                "expected_level": "critical",
                "expected_confidence": 0.6,
                "expected_indicators": ["reality_loss"]
            },
            {
                "scenario": "Disorganized thinking",
                "content": "My thoughts are all jumbled. I can't think straight. Everything is confusing.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["disorganized"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.PSYCHOTIC_EPISODE, test_cases)

    def test_substance_overdose(self):
        """Test substance overdose detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Explicit overdose",
                "content": "I took the whole bottle of pills. I think I overdosed.",
                "expected_level": "emergency",
                "expected_confidence": 0.9,
                "expected_indicators": ["overdose_explicit"]
            },
            {
                "scenario": "Medical symptoms from overdose",
                "content": "I can't breathe after taking too many pills. My chest hurts.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["overdose_explicit"]
            },
            {
                "scenario": "Excessive substance use",
                "content": "I drank the whole bottle of vodka. I can't stop drinking.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["substance_excess"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.SUBSTANCE_OVERDOSE, test_cases)

    def test_domestic_violence(self):
        """Test domestic violence detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Physical abuse with escalation",
                "content": "He hit me again last night. It's getting worse. I'm afraid he'll kill me.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["physical_abuse", "escalating_violence", "fear_for_safety"]
            },
            {
                "scenario": "Threats from partner",
                "content": "My husband threatened to kill me if I leave. He said he'll find me anywhere.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["threat_from_partner"]
            },
            {
                "scenario": "Children in danger",
                "content": "He threatened to hurt the kids if I call the police. They're not safe.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["children_danger"]
            },
            {
                "scenario": "Control and isolation",
                "content": "He won't let me see my friends. He controls all my money and monitors my phone.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["isolation_control"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.DOMESTIC_VIOLENCE, test_cases)

    def test_child_abuse(self):
        """Test child abuse detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Direct harm to child",
                "content": "I hit my child so hard I left marks. I can't control my anger.",
                "expected_level": "emergency",
                "expected_confidence": 0.9,
                "expected_indicators": ["child_harm"]
            },
            {
                "scenario": "Child in immediate danger",
                "content": "My child is in danger. I might hurt them again if I don't get help.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["child_danger"]
            },
            {
                "scenario": "Child neglect",
                "content": "I can't take care of my kids anymore. They've been left alone for days.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["child_neglect"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.CHILD_ABUSE, test_cases)

    def test_severe_depression(self):
        """Test severe depression detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Complete hopelessness",
                "content": "I'm completely hopeless. There's no reason to live. Nothing matters anymore.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["severe_hopelessness"]
            },
            {
                "scenario": "Extreme isolation",
                "content": "I'm completely alone. Nobody cares about me. I'm isolated from everyone.",
                "expected_level": "elevated",
                "expected_confidence": 0.6,
                "expected_indicators": ["isolation_extreme"]
            },
            {
                "scenario": "Loss of functioning",
                "content": "I can't get out of bed. I've stopped eating. I can't function at all.",
                "expected_level": "critical",
                "expected_confidence": 0.6,
                "expected_indicators": ["functioning_loss"]
            },
            {
                "scenario": "Severe worthlessness",
                "content": "I'm completely worthless. I hate myself. I'm a burden to everyone.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["worthlessness"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.SEVERE_DEPRESSION, test_cases)

    def test_panic_attacks(self):
        """Test panic attack detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Severe panic symptoms",
                "content": "I can't breathe, my heart is racing, I'm shaking. I think I'm dying.",
                "expected_level": "elevated",
                "expected_confidence": 0.6,
                "expected_indicators": ["panic_symptoms", "panic_fear"]
            },
            {
                "scenario": "Fear of losing control",
                "content": "I feel like I'm going crazy. I'm losing control and I can't stop it.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["panic_fear"]
            },
            {
                "scenario": "Physical panic symptoms",
                "content": "I'm sweating, nauseous, and have tingling in my hands. I feel like I'm choking.",
                "expected_level": "elevated",
                "expected_confidence": 0.4,
                "expected_indicators": ["panic_physical"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.PANIC_ATTACK, test_cases)

    def test_eating_disorder_crisis(self):
        """Test eating disorder crisis detection with various scenarios."""

        test_cases = [
            {
                "scenario": "Medical emergency from restriction",
                "content": "I haven't eaten in a week. I keep fainting and my heart feels weak.",
                "expected_level": "emergency",
                "expected_confidence": 0.8,
                "expected_indicators": ["restriction_extreme", "medical_complications"]
            },
            {
                "scenario": "Severe purging behavior",
                "content": "I can't stop throwing up after eating. I've been purging for hours.",
                "expected_level": "critical",
                "expected_confidence": 0.7,
                "expected_indicators": ["purging_behavior"]
            },
            {
                "scenario": "Extreme body dysmorphia",
                "content": "I'm so fat and disgusting. I can't look in the mirror. I hate my body.",
                "expected_level": "elevated",
                "expected_confidence": 0.4,
                "expected_indicators": ["body_dysmorphia"]
            },
            {
                "scenario": "Binge eating crisis",
                "content": "I can't stop eating. I ate everything in the house. I feel out of control.",
                "expected_level": "elevated",
                "expected_confidence": 0.5,
                "expected_indicators": ["binge_behavior"]
            }
        ]

        self._run_crisis_type_tests(CrisisType.EATING_DISORDER_CRISIS, test_cases)

    def test_multiple_crisis_types(self):
        """Test detection of multiple crisis types in single conversations."""

        test_cases = [
            {
                "scenario": "Suicide + Self-harm",
                "content": "I want to kill myself. I've been cutting myself deeper each time.",
                "expected_types": ["suicide_ideation", "self_harm"],
                "expected_level": "emergency",
                "min_confidence": 0.8
            },
            {
                "scenario": "Violence + Psychosis",
                "content": "The voices tell me to hurt people. I have a gun and I will obey them.",
                "expected_types": ["violence_threat", "psychotic_episode"],
                "expected_level": "emergency",
                "min_confidence": 0.9
            },
            {
                "scenario": "Depression + Substance",
                "content": "I'm completely hopeless. I took too many pills to numb the pain.",
                "expected_types": ["severe_depression", "substance_overdose"],
                "expected_level": "emergency",
                "min_confidence": 0.8
            }
        ]

        for test_case in test_cases:
            result = self.detector.detect_crisis({
                "id": f'multi_test_{test_case["scenario"].replace(" ", "_")}',
                "content": test_case["content"]
            })

            detected_types = [ct.value for ct in result.crisis_types]
            expected_types = test_case["expected_types"]

            # Check if all expected types are detected
            types_match = all(expected_type in detected_types for expected_type in expected_types)
            level_match = result.crisis_level.value[0] == test_case["expected_level"]
            confidence_ok = result.confidence_score >= test_case["min_confidence"]



            if not (types_match and level_match and confidence_ok):
                self.failed_tests.append({
                    "test": f'Multiple Crisis - {test_case["scenario"]}',
                    "expected": f'{test_case["expected_level"]}, {expected_types}',
                    "actual": f"{result.crisis_level.value[0]}, {detected_types}",
                    "confidence": result.confidence_score
                })

    def test_false_positives(self):
        """Test scenarios that should NOT trigger crisis detection."""

        test_cases = [
            {
                "scenario": "Normal sadness",
                "content": "I feel sad today but I know it will pass. I have good support.",
                "expected_level": "routine"
            },
            {
                "scenario": "Stress without crisis",
                "content": "Work has been really stressful but I'm managing okay with exercise.",
                "expected_level": "routine"
            },
            {
                "scenario": "Past crisis resolved",
                "content": "I used to have suicidal thoughts but I'm much better now with therapy.",
                "expected_level": "routine"
            },
            {
                "scenario": "Academic discussion",
                "content": "I'm studying suicide prevention methods for my psychology class.",
                "expected_level": "routine"
            },
            {
                "scenario": "Helping someone else",
                "content": "My friend is going through a tough time and I want to help them.",
                "expected_level": "routine"
            }
        ]

        for test_case in test_cases:
            result = self.detector.detect_crisis({
                "id": f'false_pos_test_{test_case["scenario"].replace(" ", "_")}',
                "content": test_case["content"]
            })

            level_correct = result.crisis_level.value[0] == test_case["expected_level"]


            if not level_correct:
                self.failed_tests.append({
                    "test": f'False Positive - {test_case["scenario"]}',
                    "expected": test_case["expected_level"],
                    "actual": result.crisis_level.value[0],
                    "confidence": result.confidence_score
                })

    def test_borderline_cases(self):
        """Test borderline cases that are difficult to classify."""

        test_cases = [
            {
                "scenario": "Mild suicidal ideation",
                "content": "Sometimes I wonder if life is worth living, but I would never act on it.",
                "acceptable_levels": ["routine", "elevated"]
            },
            {
                "scenario": "Emotional distress",
                "content": "I feel overwhelmed and don't know how to cope, but I'm not in danger.",
                "acceptable_levels": ["routine", "elevated"]
            },
            {
                "scenario": "Relationship conflict",
                "content": "I'm so angry at my partner I could scream, but I would never hurt them.",
                "acceptable_levels": ["routine", "elevated"]
            }
        ]

        for test_case in test_cases:
            result = self.detector.detect_crisis({
                "id": f'borderline_test_{test_case["scenario"].replace(" ", "_")}',
                "content": test_case["content"]
            })

            level_acceptable = result.crisis_level.value[0] in test_case["acceptable_levels"]


            if not level_acceptable:
                self.failed_tests.append({
                    "test": f'Borderline - {test_case["scenario"]}',
                    "expected": f'One of: {test_case["acceptable_levels"]}',
                    "actual": result.crisis_level.value[0],
                    "confidence": result.confidence_score
                })

    def _run_crisis_type_tests(self, crisis_type: CrisisType, test_cases: List[Dict]):
        """Run tests for a specific crisis type."""
        for test_case in test_cases:
            result = self.detector.detect_crisis({
                "id": f'{crisis_type.value}_test_{test_case["scenario"].replace(" ", "_")}',
                "content": test_case["content"]
            })

            # Check if crisis type is detected
            detected_types = [ct.value for ct in result.crisis_types]
            type_detected = crisis_type.value in detected_types

            # Check level and confidence
            level_match = result.crisis_level.value[0] == test_case["expected_level"]
            confidence_ok = result.confidence_score >= (test_case["expected_confidence"] - 0.2)  # Allow some tolerance

            # Check indicators (if specified)
            indicators_ok = True
            if "expected_indicators" in test_case:
                expected_indicators = test_case["expected_indicators"]
                detected_indicators = result.detected_indicators
                indicators_ok = any(indicator in detected_indicators for indicator in expected_indicators)

            overall_pass = type_detected and level_match and confidence_ok and indicators_ok


            self.test_results.append({
                "crisis_type": crisis_type.value,
                "scenario": test_case["scenario"],
                "passed": overall_pass,
                "detected_level": result.crisis_level.value[0],
                "confidence": result.confidence_score,
                "detected_types": detected_types
            })

            if not overall_pass:
                self.failed_tests.append({
                    "test": f'{crisis_type.value} - {test_case["scenario"]}',
                    "expected": f'{test_case["expected_level"]}, confidence≥{test_case["expected_confidence"]}',
                    "actual": f"{result.crisis_level.value[0]}, confidence={result.confidence_score:.3f}",
                    "type_detected": type_detected,
                    "indicators_ok": indicators_ok
                })

    def generate_test_report(self):
        """Generate comprehensive test report."""

        len(self.test_results)
        sum(1 for result in self.test_results if result["passed"])
        failed_tests = len(self.failed_tests)


        # Crisis type breakdown
        crisis_type_stats = {}
        for result in self.test_results:
            crisis_type = result["crisis_type"]
            if crisis_type not in crisis_type_stats:
                crisis_type_stats[crisis_type] = {"total": 0, "passed": 0}
            crisis_type_stats[crisis_type]["total"] += 1
            if result["passed"]:
                crisis_type_stats[crisis_type]["passed"] += 1

        for crisis_type, stats in crisis_type_stats.items():
            (stats["passed"] / stats["total"]) * 100

        # Failed tests details
        if self.failed_tests:
            for _i, failed_test in enumerate(self.failed_tests, 1):
                if "type_detected" in failed_test:
                    pass

        # Overall assessment
        if failed_tests == 0 or failed_tests <= 3:
            pass
        else:
            pass

        return failed_tests == 0

def main():
    """Run the comprehensive crisis test suite."""
    test_suite = ComprehensiveCrisisTestSuite()
    success = test_suite.run_all_tests()

    return bool(success)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
