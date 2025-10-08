#!/usr/bin/env python3
"""
Comprehensive crisis response time validation.
Tests the full escalation workflow timing to ensure compliance with emergency response requirements.
"""

import logging
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configure logging to capture timing information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

from quality.crisis_intervention_detector import (
    CrisisDetection,
    CrisisInterventionDetector,
    EscalationRecord,
)


class ResponseTimeValidator:
    """Validates crisis response times meet emergency requirements."""

    def __init__(self):
        self.detector = CrisisInterventionDetector()
        self.response_times = []
        self.escalation_logs = []

        # Add timing callback
        self.detector.add_escalation_callback(self._timing_callback)

    def _timing_callback(self, detection: CrisisDetection, escalation: EscalationRecord):
        """Callback to capture escalation timing data."""
        self.escalation_logs.append({
            "detection_id": detection.detection_id,
            "crisis_level": detection.crisis_level.value[0],
            "response_time_minutes": escalation.response_time_minutes,
            "actions_count": len(escalation.actions_taken),
            "contacts_count": len(escalation.contacts_notified),
            "timestamp": datetime.now()
        })

    def test_emergency_response_times(self):
        """Test emergency response times meet 0-5 minute requirement."""
        emergency_cases = [
            ("Suicide Intent", "I want to kill myself tonight with pills"),
            ("Violence Threat", "I have a gun and will kill him"),
            ("Child Abuse", "I hurt my child and might do it again"),
            ("Overdose", "I took the whole bottle of pills"),
            ("Immediate Suicide", "I am going to end my life right now")
        ]

        emergency_times = []

        for name, content in emergency_cases:
            start_time = time.time()

            # Clear previous logs
            self.escalation_logs.clear()

            # Trigger crisis detection
            detection = self.detector.detect_crisis({
                "id": f'emergency_test_{name.replace(" ", "_")}',
                "content": content
            })

            # Calculate detection time
            detection_time = (time.time() - start_time) * 1000  # ms

            # Get escalation time from callback
            escalation_time = self.escalation_logs[0]["response_time_minutes"] if self.escalation_logs else 0

            emergency_times.append({
                "case": name,
                "detection_time_ms": detection_time,
                "escalation_time_minutes": escalation_time,
                "crisis_level": detection.crisis_level.value[0],
                "meets_requirement": escalation_time <= 5.0
            })


        return emergency_times

    def test_critical_response_times(self):
        """Test critical response times meet 5-15 minute requirement."""
        critical_cases = [
            ("Self Harm Planning", "I need to cut myself deeper, I have the tools ready"),
            ("Severe Depression", "I am completely hopeless, nobody cares, I am alone"),
            ("Substance Crisis", "I took too many pills and feel dizzy"),
            ("Domestic Violence", "He hit me again and threatened to kill me"),
            ("Psychotic Episode", "The voices are telling me to hurt people")
        ]

        critical_times = []

        for name, content in critical_cases:
            start_time = time.time()
            self.escalation_logs.clear()

            detection = self.detector.detect_crisis({
                "id": f'critical_test_{name.replace(" ", "_")}',
                "content": content
            })

            detection_time = (time.time() - start_time) * 1000
            escalation_time = self.escalation_logs[0]["response_time_minutes"] if self.escalation_logs else 0

            critical_times.append({
                "case": name,
                "detection_time_ms": detection_time,
                "escalation_time_minutes": escalation_time,
                "crisis_level": detection.crisis_level.value[0],
                "meets_requirement": escalation_time <= 15.0
            })


        return critical_times

    def test_elevated_response_times(self):
        """Test elevated response times meet 15-60 minute requirement."""
        elevated_cases = [
            ("Panic Attack", "I can't breathe and my heart is racing"),
            ("Mild Self Harm", "I want to hurt myself but not seriously"),
            ("Depression Episode", "I feel completely hopeless today"),
            ("Eating Disorder", "I haven't eaten in days, I hate my body"),
            ("Anxiety Crisis", "I am overwhelmed and can't cope")
        ]

        elevated_times = []

        for name, content in elevated_cases:
            start_time = time.time()
            self.escalation_logs.clear()

            detection = self.detector.detect_crisis({
                "id": f'elevated_test_{name.replace(" ", "_")}',
                "content": content
            })

            detection_time = (time.time() - start_time) * 1000
            escalation_time = self.escalation_logs[0]["response_time_minutes"] if self.escalation_logs else 0

            elevated_times.append({
                "case": name,
                "detection_time_ms": detection_time,
                "escalation_time_minutes": escalation_time,
                "crisis_level": detection.crisis_level.value[0],
                "meets_requirement": escalation_time <= 60.0
            })


        return elevated_times

    def test_concurrent_response_times(self):
        """Test response times under concurrent load."""
        concurrent_cases = [
            "I want to kill myself right now",
            "I have a gun and will use it",
            "I took all the pills",
            "I am going to hurt him",
            "I want to end my life tonight"
        ] * 4  # 20 concurrent requests

        def process_crisis(content):
            start_time = time.time()
            detection = self.detector.detect_crisis({
                "id": f"concurrent_{threading.current_thread().ident}_{int(time.time() * 1000)}",
                "content": content
            })
            processing_time = (time.time() - start_time) * 1000
            return {
                "processing_time_ms": processing_time,
                "crisis_level": detection.crisis_level.value[0],
                "confidence": detection.confidence_score
            }

        # Execute concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_crisis, case) for case in concurrent_cases]
            results = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Analyze results
        processing_times = [r["processing_time_ms"] for r in results]
        emergency_count = sum(1 for r in results if r["crisis_level"] == "emergency")


        # Check if concurrent processing meets requirements
        max_acceptable_time = 1000  # 1 second max for concurrent processing
        concurrent_success = max(processing_times) <= max_acceptable_time


        return {
            "total_requests": len(concurrent_cases),
            "total_time_seconds": total_time,
            "avg_processing_time_ms": statistics.mean(processing_times),
            "max_processing_time_ms": max(processing_times),
            "emergency_detection_rate": emergency_count / len(concurrent_cases),
            "meets_concurrent_requirement": concurrent_success
        }

    def test_response_time_consistency(self):
        """Test response time consistency over multiple runs."""
        test_case = "I want to kill myself tonight"
        runs = 50
        times = []

        for i in range(runs):
            start_time = time.time()
            self.detector.detect_crisis({
                "id": f"consistency_test_{i}",
                "content": test_case
            })
            processing_time = (time.time() - start_time) * 1000
            times.append(processing_time)

        # Calculate statistics
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)


        # Check consistency (CV should be < 50% for good consistency)
        consistency_good = (std_dev/mean_time) < 0.5

        return {
            "runs": runs,
            "mean_time_ms": mean_time,
            "std_dev_ms": std_dev,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "coefficient_of_variation": (std_dev/mean_time)*100,
            "consistency_good": consistency_good
        }

def main():
    """Run comprehensive response time validation."""
    validator = ResponseTimeValidator()


    # Test emergency response times (0-5 minutes)
    emergency_results = validator.test_emergency_response_times()

    # Test critical response times (5-15 minutes)
    critical_results = validator.test_critical_response_times()

    # Test elevated response times (15-60 minutes)
    elevated_results = validator.test_elevated_response_times()

    # Test concurrent processing
    concurrent_results = validator.test_concurrent_response_times()

    # Test consistency
    consistency_results = validator.test_response_time_consistency()

    # Summary analysis

    emergency_pass = all(r["meets_requirement"] for r in emergency_results)
    critical_pass = all(r["meets_requirement"] for r in critical_results)
    elevated_pass = all(r["meets_requirement"] for r in elevated_results)


    overall_success = (
        emergency_pass and
        critical_pass and
        elevated_pass and
        concurrent_results["meets_concurrent_requirement"] and
        consistency_results["consistency_good"]
    )


    if overall_success:
        pass
    else:
        pass

    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
