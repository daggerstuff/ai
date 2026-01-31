#!/usr/bin/env python3
"""
Crisis Detection Integration Tests - End-to-end testing with realistic conversations.
Tests the complete crisis detection workflow from conversation input to escalation output.
"""

import logging
import sys
import time
from datetime import datetime

from quality.crisis_intervention_detector import CrisisInterventionDetector

# Configure logging for integration testing
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


class CrisisDetectionIntegrationTests:
    """Integration tests for end-to-end crisis detection workflow."""

    def __init__(self):
        self.detector = CrisisInterventionDetector()
        self.integration_results = []
        self.failed_integrations = []
        self.escalation_logs = []

        # Add callback to capture escalation events
        self.detector.add_escalation_callback(self._capture_escalation)

    def _capture_escalation(self, detection, escalation):
        """Capture escalation events for integration testing."""
        self.escalation_logs.append({
            "detection_id": detection.detection_id,
            "crisis_level": detection.crisis_level.value[0],
            "crisis_types": [ct.value for ct in detection.crisis_types],
            "confidence": detection.confidence_score,
            "actions_taken": [a.value for a in escalation.actions_taken],
            "contacts_notified": escalation.contacts_notified,
            "response_time": escalation.response_time_minutes,
            "timestamp": datetime.now()
        })

    def run_all_integration_tests(self):
        """Run all integration tests."""

        # Test realistic conversation flows
        self.test_therapeutic_conversation_integration()
        self.test_multi_turn_conversation_detection()
        self.test_escalation_workflow_integration()
        self.test_data_format_integration()
        self.test_performance_integration()
        self.test_callback_system_integration()
        self.test_history_tracking_integration()

        # Generate integration test report
        self.generate_integration_report()

        return len(self.failed_integrations) == 0

    def test_therapeutic_conversation_integration(self):
        """Test integration with realistic therapeutic conversation flows."""

        conversation_scenarios = [
            {
                "name": "Gradual Crisis Disclosure",
                "turns": [
                    {"role": "user", "content": "I've been feeling really down lately"},
                    {"role": "assistant", "content": "I'm sorry to hear you're struggling. Can you tell me more about how you've been feeling?"},
                    {"role": "user", "content": "It's getting worse. I don't see the point in anything anymore"},
                    {"role": "assistant", "content": "That sounds really difficult. Are you having thoughts of hurting yourself?"},
                    {"role": "user", "content": "Yes, I want to kill myself. I have pills and I'm going to take them tonight"},
                    {"role": "assistant", "content": "I'm very concerned about your safety. This is a crisis situation that requires immediate help."}
                ],
                "expected_final_level": "emergency",
                "expected_escalation": True,
                "expected_crisis_types": ["suicide_ideation"]
            },
            {
                "name": "Domestic Violence Revelation",
                "turns": [
                    {"role": "user", "content": "My relationship has been stressful"},
                    {"role": "assistant", "content": "Relationship stress can be very challenging. What kind of stress are you experiencing?"},
                    {"role": "user", "content": "My partner gets angry sometimes"},
                    {"role": "assistant", "content": "How does your partner express their anger?"},
                    {"role": "user", "content": "He hit me last night. He said he'll kill me if I tell anyone"},
                    {"role": "assistant", "content": "I'm very concerned for your safety. What you're describing is domestic violence."}
                ],
                "expected_final_level": "critical",
                "expected_escalation": True,
                "expected_crisis_types": ["domestic_violence"]
            },
            {
                "name": "Psychotic Episode Development",
                "turns": [
                    {"role": "user", "content": "I've been having trouble sleeping"},
                    {"role": "assistant", "content": "Sleep problems can be very disruptive. What's been keeping you awake?"},
                    {"role": "user", "content": "I keep hearing things at night"},
                    {"role": "assistant", "content": "What kind of things are you hearing?"},
                    {"role": "user", "content": "Voices. They're telling me that people are watching me and I need to hurt them"},
                    {"role": "assistant", "content": "This sounds very distressing. The voices you're hearing are concerning."}
                ],
                "expected_final_level": "emergency",
                "expected_escalation": True,
                "expected_crisis_types": ["psychotic_episode"]
            },
            {
                "name": "Non-Crisis Support Conversation",
                "turns": [
                    {"role": "user", "content": "I've been feeling stressed about work"},
                    {"role": "assistant", "content": "Work stress is very common. What aspects of work are causing you stress?"},
                    {"role": "user", "content": "My boss is demanding and I have too many deadlines"},
                    {"role": "assistant", "content": "That does sound overwhelming. How are you coping with these pressures?"},
                    {"role": "user", "content": "I've been exercising more and talking to friends. It helps"},
                    {"role": "assistant", "content": "Those are excellent coping strategies. It sounds like you're managing well."}
                ],
                "expected_final_level": "routine",
                "expected_escalation": False,
                "expected_crisis_types": []
            }
        ]

        for scenario in conversation_scenarios:
            self._test_conversation_scenario(scenario)

    def test_multi_turn_conversation_detection(self):
        """Test crisis detection across multiple conversation turns."""

        # Test conversation where crisis emerges gradually
        conversation_id = "multi_turn_test_1"
        turns = [
            "I've been having a hard time lately",
            "Everything feels overwhelming and hopeless",
            "I don't think I can keep going like this",
            "I've been thinking about ending my life",
            "I have a plan and I'm going to do it tonight"
        ]

        crisis_levels = []
        confidence_scores = []

        # Process each turn and track escalation
        full_conversation = ""
        for i, turn in enumerate(turns):
            full_conversation += f" {turn}"

            result = self.detector.detect_crisis({
                "id": f"{conversation_id}_turn_{i+1}",
                "content": full_conversation.strip(),
                "turns": [{"role": "user", "content": full_conversation.strip()}]
            })

            crisis_levels.append(result.crisis_level.value[0])
            confidence_scores.append(result.confidence_score)

        # Verify escalation pattern
        expected_escalation = ["routine", "elevated", "elevated", "critical", "emergency"]
        escalation_correct = True


        for i, (turn, level, _confidence) in enumerate(zip(turns, crisis_levels, confidence_scores, strict=False)):
            expected = expected_escalation[i] if i < len(expected_escalation) else "unknown"
            if level != expected:
                escalation_correct = False


        self.integration_results.append({
            "test": "Multi-Turn Escalation",
            "passed": escalation_correct,
            "details": f"Expected escalation pattern, got: {crisis_levels}"
        })

        if not escalation_correct:
            self.failed_integrations.append({
                "test": "Multi-Turn Escalation",
                "expected": expected_escalation,
                "actual": crisis_levels,
                "issue": "Crisis escalation pattern incorrect"
            })

    def test_escalation_workflow_integration(self):
        """Test complete escalation workflow integration."""

        # Clear previous escalation logs
        self.escalation_logs.clear()

        escalation_test_cases = [
            {
                "name": "Emergency Escalation",
                "content": "I want to kill myself right now with the gun in my drawer",
                "expected_actions": ["emergency_services", "supervisor_alert", "documentation"],
                "expected_contacts": ["911", "supervisor", "crisis_coordinator"],
                "max_response_time": 0.1  # Should be immediate
            },
            {
                "name": "Critical Escalation",
                "content": "I keep thinking about hurting myself. I have a plan but haven't decided when",
                "expected_actions": ["crisis_hotline", "immediate_referral", "supervisor_alert", "documentation"],
                "expected_contacts": ["988", "supervisor", "crisis_team"],
                "max_response_time": 0.1
            },
            {
                "name": "Elevated Escalation",
                "content": "I feel completely hopeless and alone. I can't function anymore",
                "expected_actions": ["safety_planning", "follow_up_scheduling", "documentation"],
                "expected_contacts": ["supervisor", "case_manager"],
                "max_response_time": 0.1
            }
        ]


        for test_case in escalation_test_cases:
            # Clear logs for this test
            self.escalation_logs.clear()

            # Trigger crisis detection
            self.detector.detect_crisis({
                "id": f'escalation_test_{test_case["name"].replace(" ", "_")}',
                "content": test_case["content"]
            })

            # Check if escalation was triggered
            if self.escalation_logs:
                escalation = self.escalation_logs[0]

                # Verify actions
                actions_match = len(escalation["actions_taken"]) >= len(test_case["expected_actions"])

                # Verify contacts
                contacts_match = len(escalation["contacts_notified"]) >= len(test_case["expected_contacts"])

                # Verify response time
                response_time_ok = escalation["response_time"] <= test_case["max_response_time"]

                overall_pass = actions_match and contacts_match and response_time_ok


                self.integration_results.append({
                    "test": f'Escalation - {test_case["name"]}',
                    "passed": overall_pass,
                    "details": f'Actions: {escalation["actions_taken"]}, Contacts: {escalation["contacts_notified"]}'
                })

                if not overall_pass:
                    self.failed_integrations.append({
                        "test": f'Escalation - {test_case["name"]}',
                        "expected": f'Actions≥{len(test_case["expected_actions"])}, Contacts≥{len(test_case["expected_contacts"])}, Time≤{test_case["max_response_time"]}',
                        "actual": f'Actions={len(escalation["actions_taken"])}, Contacts={len(escalation["contacts_notified"])}, Time={escalation["response_time"]:.3f}',
                        "issue": "Escalation workflow incomplete"
                    })
            else:
                self.failed_integrations.append({
                    "test": f'Escalation - {test_case["name"]}',
                    "expected": "Escalation triggered",
                    "actual": "No escalation",
                    "issue": "Escalation not triggered when expected"
                })

    def test_data_format_integration(self):
        """Test integration with different data formats."""

        crisis_content = "I want to kill myself tonight"

        format_test_cases = [
            {
                "name": "Messages Format",
                "data": {
                    "conversation_id": "format_test_1",
                    "messages": [
                        {"role": "user", "content": crisis_content},
                        {"role": "assistant", "content": "I understand you are in pain"}
                    ]
                }
            },
            {
                "name": "Content Format",
                "data": {
                    "id": "format_test_2",
                    "content": crisis_content,
                    "turns": [{"role": "user", "content": crisis_content}]
                }
            },
            {
                "name": "Text Format",
                "data": {
                    "id": "format_test_3",
                    "text": crisis_content
                }
            },
            {
                "name": "Turns Format",
                "data": {
                    "id": "format_test_4",
                    "turns": [
                        {"role": "user", "content": crisis_content},
                        {"role": "assistant", "content": "I am concerned about you"}
                    ]
                }
            }
        ]


        for test_case in format_test_cases:
            try:
                result = self.detector.detect_crisis(test_case["data"])

                crisis_detected = len(result.crisis_types) > 0
                level = result.crisis_level.value[0]
                confidence = result.confidence_score

                # Should detect emergency level crisis
                format_works = crisis_detected and level in ["emergency", "critical"]


                self.integration_results.append({
                    "test": f'Format - {test_case["name"]}',
                    "passed": format_works,
                    "details": f"Detected: {crisis_detected}, Level: {level}, Confidence: {confidence:.3f}"
                })

                if not format_works:
                    self.failed_integrations.append({
                        "test": f'Format - {test_case["name"]}',
                        "expected": "Crisis detected with emergency/critical level",
                        "actual": f"Detected: {crisis_detected}, Level: {level}",
                        "issue": "Data format not properly handled"
                    })

            except Exception as e:
                self.failed_integrations.append({
                    "test": f'Format - {test_case["name"]}',
                    "expected": "Successful processing",
                    "actual": f"Error: {e!s}",
                    "issue": "Data format caused exception"
                })

    def test_performance_integration(self):
        """Test performance integration under various loads."""

        # Test single request performance
        start_time = time.time()
        self.detector.detect_crisis({
            "id": "perf_test_single",
            "content": "I want to kill myself tonight"
        })
        single_request_time = (time.time() - start_time) * 1000

        # Test batch processing performance
        batch_requests = [
            {"id": f"perf_test_batch_{i}", "content": "I want to hurt myself"}
            for i in range(10)
        ]

        start_time = time.time()
        batch_results = []
        for request in batch_requests:
            batch_results.append(self.detector.detect_crisis(request))
        batch_time = (time.time() - start_time) * 1000
        avg_batch_time = batch_time / len(batch_requests)


        # Performance thresholds
        single_threshold = 100  # 100ms max for single request
        batch_threshold = 50   # 50ms max average for batch

        single_ok = single_request_time <= single_threshold
        batch_ok = avg_batch_time <= batch_threshold


        self.integration_results.append({
            "test": "Performance Integration",
            "passed": single_ok and batch_ok,
            "details": f"Single: {single_request_time:.1f}ms, Batch avg: {avg_batch_time:.1f}ms"
        })

        if not (single_ok and batch_ok):
            self.failed_integrations.append({
                "test": "Performance Integration",
                "expected": f"Single≤{single_threshold}ms, Batch avg≤{batch_threshold}ms",
                "actual": f"Single={single_request_time:.1f}ms, Batch avg={avg_batch_time:.1f}ms",
                "issue": "Performance thresholds exceeded"
            })

    def test_callback_system_integration(self):
        """Test callback system integration."""

        callback_triggered = False
        callback_data = None

        def test_callback(detection, escalation):
            nonlocal callback_triggered, callback_data
            callback_triggered = True
            callback_data = {
                "detection_id": detection.detection_id,
                "crisis_level": detection.crisis_level.value[0],
                "escalation_actions": len(escalation.actions_taken)
            }

        # Add test callback
        self.detector.add_escalation_callback(test_callback)

        # Trigger crisis that should cause escalation
        self.detector.detect_crisis({
            "id": "callback_test",
            "content": "I want to kill myself right now"
        })

        callback_works = callback_triggered and callback_data is not None
        if callback_data:
            pass


        self.integration_results.append({
            "test": "Callback System",
            "passed": callback_works,
            "details": f"Triggered: {callback_triggered}, Data: {callback_data}"
        })

        if not callback_works:
            self.failed_integrations.append({
                "test": "Callback System",
                "expected": "Callback triggered with data",
                "actual": f"Triggered: {callback_triggered}",
                "issue": "Callback system not working properly"
            })

    def test_history_tracking_integration(self):
        """Test history tracking integration."""

        initial_detection_count = len(self.detector.detection_history)
        initial_escalation_count = len(self.detector.escalation_history)

        # Trigger multiple crises
        test_cases = [
            "I want to kill myself",
            "I am going to hurt someone",
            "I feel completely hopeless"
        ]

        for i, content in enumerate(test_cases):
            self.detector.detect_crisis({
                "id": f"history_test_{i}",
                "content": content
            })

        final_detection_count = len(self.detector.detection_history)
        final_escalation_count = len(self.detector.escalation_history)

        detections_added = final_detection_count - initial_detection_count
        escalations_added = final_escalation_count - initial_escalation_count


        history_works = detections_added == len(test_cases) and escalations_added >= len(test_cases)

        self.integration_results.append({
            "test": "History Tracking",
            "passed": history_works,
            "details": f"Detections: +{detections_added}, Escalations: +{escalations_added}"
        })

        if not history_works:
            self.failed_integrations.append({
                "test": "History Tracking",
                "expected": f"Detections: +{len(test_cases)}, Escalations: ≥{len(test_cases)}",
                "actual": f"Detections: +{detections_added}, Escalations: +{escalations_added}",
                "issue": "History tracking not working properly"
            })

    def _test_conversation_scenario(self, scenario):
        """Test a specific conversation scenario."""
        # Build full conversation content
        conversation_content = " ".join([turn["content"] for turn in scenario["turns"]])

        # Test the final state
        result = self.detector.detect_crisis({
            "id": f'conversation_test_{scenario["name"].replace(" ", "_")}',
            "content": conversation_content,
            "turns": scenario["turns"]
        })

        # Check results
        level_match = result.crisis_level.value[0] == scenario["expected_final_level"]
        escalation_match = (len(self.escalation_logs) > 0) == scenario["expected_escalation"]

        detected_types = [ct.value for ct in result.crisis_types]
        types_match = all(expected_type in detected_types for expected_type in scenario["expected_crisis_types"])

        overall_pass = level_match and escalation_match and types_match


        self.integration_results.append({
            "test": f'Conversation - {scenario["name"]}',
            "passed": overall_pass,
            "details": f"Level: {result.crisis_level.value[0]}, Types: {detected_types}"
        })

        if not overall_pass:
            self.failed_integrations.append({
                "test": f'Conversation - {scenario["name"]}',
                "expected": f'Level: {scenario["expected_final_level"]}, Types: {scenario["expected_crisis_types"]}, Escalation: {scenario["expected_escalation"]}',
                "actual": f"Level: {result.crisis_level.value[0]}, Types: {detected_types}, Escalation: {len(self.escalation_logs) > 0}",
                "issue": "Conversation scenario not handled correctly"
            })

        # Clear escalation logs for next test
        self.escalation_logs.clear()

    def generate_integration_report(self):
        """Generate comprehensive integration test report."""

        len(self.integration_results)
        sum(1 for result in self.integration_results if result["passed"])
        failed_tests = len(self.failed_integrations)


        # Test category breakdown
        categories = {}
        for result in self.integration_results:
            category = result["test"].split(" - ")[0]
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0}
            categories[category]["total"] += 1
            if result["passed"]:
                categories[category]["passed"] += 1

        for category, stats in categories.items():
            (stats["passed"] / stats["total"]) * 100

        # Failed tests details
        if self.failed_integrations:
            for _i, _failed_test in enumerate(self.failed_integrations, 1):
                pass

        # Overall assessment
        if failed_tests == 0 or failed_tests <= 2:
            pass
        else:
            pass

        return failed_tests == 0

def main():
    """Run the crisis detection integration test suite."""
    integration_tests = CrisisDetectionIntegrationTests()
    success = integration_tests.run_all_integration_tests()

    return bool(success)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
