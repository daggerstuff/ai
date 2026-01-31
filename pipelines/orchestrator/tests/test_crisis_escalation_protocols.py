#!/usr/bin/env python3
"""
Test script for crisis escalation protocols.
"""

import logging
import sys
import time

# Configure logging to capture escalation messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from quality.crisis_intervention_detector import (
    CrisisDetection,
    CrisisInterventionDetector,
    EscalationRecord,
)


class EscalationTester:
    """Test harness for crisis escalation protocols."""

    def __init__(self):
        self.detector = CrisisInterventionDetector()
        self.escalation_logs = []
        self.callback_logs = []

        # Add test callback
        self.detector.add_escalation_callback(self._test_callback)

    def _test_callback(self, detection: CrisisDetection, escalation: EscalationRecord):
        """Test callback to capture escalation events."""
        self.callback_logs.append({
            "detection_id": detection.detection_id,
            "crisis_level": detection.crisis_level.value[0],
            "actions_taken": [a.value for a in escalation.actions_taken],
            "contacts_notified": escalation.contacts_notified,
            "response_time": escalation.response_time_minutes
        })

    def test_escalation_protocols(self):
        """Test escalation protocols for different crisis levels."""
        test_cases = [
            {
                "name": "Emergency - Suicide Intent",
                "content": "I want to kill myself tonight with pills",
                "expected_level": "emergency",
                "expected_actions": ["emergency_services", "supervisor_alert", "documentation"],
                "expected_contacts": ["911", "supervisor", "crisis_coordinator"]
            },
            {
                "name": "Emergency - Violence Threat",
                "content": "I have a gun and I will kill him",
                "expected_level": "emergency",
                "expected_actions": ["emergency_services", "supervisor_alert", "documentation"],
                "expected_contacts": ["911", "supervisor", "crisis_coordinator"]
            },
            {
                "name": "Elevated - Self Harm",
                "content": "I need to cut myself deeper this time",
                "expected_level": "elevated",
                "expected_actions": ["safety_planning", "follow_up_scheduling", "documentation"],
                "expected_contacts": ["supervisor", "case_manager"]
            },
            {
                "name": "Elevated - Severe Depression",
                "content": "I am completely hopeless and nobody cares about me",
                "expected_level": "elevated",
                "expected_actions": ["safety_planning", "follow_up_scheduling", "documentation"],
                "expected_contacts": ["supervisor", "case_manager"]
            },
            {
                "name": "Elevated - Panic Attack",
                "content": "I can't breathe and my heart is racing, I think I'm dying",
                "expected_level": "elevated",
                "expected_actions": ["safety_planning", "follow_up_scheduling", "documentation"],
                "expected_contacts": ["supervisor", "case_manager"]
            },
            {
                "name": "Routine - Normal Conversation",
                "content": "How are you doing today?",
                "expected_level": "routine",
                "expected_actions": ["documentation"],
                "expected_contacts": []
            }
        ]


        results = []
        for i, test_case in enumerate(test_cases):
            test_id = f"test_{i+1}"
            conversation = {"id": test_id, "content": test_case["content"]}

            # Clear previous logs
            self.callback_logs.clear()

            # Capture start time
            start_time = time.time()

            # Trigger crisis detection
            detection = self.detector.detect_crisis(conversation)

            # Calculate actual response time
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get callback data
            callback_data = self.callback_logs[0] if self.callback_logs else None

            # Validate results
            level_match = detection.crisis_level.value[0] == test_case["expected_level"]

            # Check if escalation was triggered for non-routine cases
            escalation_triggered = len(self.callback_logs) > 0
            should_escalate = test_case["expected_level"] != "routine"
            escalation_match = escalation_triggered == should_escalate

            # Count actions and contacts
            actions_count = len(callback_data["actions_taken"]) if callback_data else 0
            contacts_count = len(callback_data["contacts_notified"]) if callback_data else 0

            # Determine status


            results.append({
                "name": test_case["name"],
                "level_match": level_match,
                "escalation_match": escalation_match,
                "actions_count": actions_count,
                "contacts_count": contacts_count,
                "response_time_ms": response_time,
                "callback_data": callback_data
            })

        return results

    def test_response_times(self):
        """Test crisis response times meet requirements."""

        # Test multiple emergency cases for response time consistency
        emergency_cases = [
            "I want to kill myself right now",
            "I have a gun and will use it tonight",
            "I took all the pills in the bottle"
        ]

        response_times = []
        for i, content in enumerate(emergency_cases):
            start_time = time.time()
            self.detector.detect_crisis({"id": f"timing_test_{i}", "content": content})
            response_time = (time.time() - start_time) * 1000
            response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min(response_times)


        # Check if response times meet requirements (should be < 1000ms for emergency)
        if max_response_time < 1000:
            pass
        else:
            pass

        return {
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": max_response_time,
            "meets_requirements": max_response_time < 1000
        }

    def test_escalation_callbacks(self):
        """Test escalation callback system."""

        callback_count = 0

        def test_callback(detection, escalation):
            nonlocal callback_count
            callback_count += 1

        # Add additional callback
        self.detector.add_escalation_callback(test_callback)

        # Trigger crisis
        self.detector.detect_crisis({"id": "callback_test", "content": "I want to hurt myself"})

        if callback_count > 0:
            pass
        else:
            pass

        return callback_count > 0

    def test_escalation_history(self):
        """Test escalation history tracking."""

        initial_count = len(self.detector.escalation_history)

        # Trigger multiple crises
        self.detector.detect_crisis({"id": "history_test_1", "content": "I want to kill myself"})
        self.detector.detect_crisis({"id": "history_test_2", "content": "I will hurt him"})

        final_count = len(self.detector.escalation_history)
        escalations_added = final_count - initial_count


        return escalations_added >= 2

def main():
    """Run comprehensive escalation protocol tests."""
    tester = EscalationTester()

    # Test escalation protocols
    protocol_results = tester.test_escalation_protocols()

    # Test response times
    timing_results = tester.test_response_times()

    # Test callbacks
    callback_success = tester.test_escalation_callbacks()

    # Test history
    history_success = tester.test_escalation_history()

    # Summary

    protocol_passes = sum(1 for r in protocol_results if r["level_match"] and r["escalation_match"])
    protocol_total = len(protocol_results)


    overall_success = (
        protocol_passes == protocol_total and
        timing_results["meets_requirements"] and
        callback_success and
        history_success
    )

    return bool(overall_success)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
