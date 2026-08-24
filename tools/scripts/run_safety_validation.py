#!/usr/bin/env python3
"""
Task 103: Safety Validation Certification Test Runner
Critical Safety Component for Production Deployment

This script runs comprehensive safety validation testing including:
- Crisis detection accuracy validation (>95% required)
- Safety monitoring system testing
- Incident response validation
- Safety compliance certification
"""

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from safety_monitoring_system import SafetyMonitoringSystem
from safety_validation_system import SafetyValidationSystem


class ComprehensiveSafetyValidator:
    """
    Comprehensive Safety Validation for Task 103
    """

    def __init__(self):
        self.validation_system = SafetyValidationSystem()
        self.monitoring_system = SafetyMonitoringSystem()
        self.overall_score = 0.0
        self.production_ready = False

    async def run_complete_safety_validation(self) -> dict:
        """Run complete safety validation and certification"""

        # Phase 1: Core Safety Validation
        validation_results = await self.run_safety_validation_tests()

        # Phase 2: Safety Monitoring System Testing
        monitoring_results = await self.run_monitoring_system_tests()

        # Phase 3: Incident Response Validation
        incident_response_results = await self.run_incident_response_tests()

        # Phase 4: Production Readiness Assessment
        production_assessment = self.assess_production_readiness(
            validation_results, monitoring_results, incident_response_results
        )

        # Generate comprehensive report
        comprehensive_report = self.generate_comprehensive_report(
            validation_results,
            monitoring_results,
            incident_response_results,
            production_assessment,
        )

        # Save results
        self.save_results(comprehensive_report)

        # Print summary
        self.print_summary(comprehensive_report)

        return comprehensive_report

    async def run_safety_validation_tests(self) -> dict:
        """Run core safety validation tests"""

        # Run comprehensive safety validation
        validation_report = await self.validation_system.run_comprehensive_safety_validation()

        crisis_accuracy = validation_report["safety_validation_summary"]["crisis_detection_accuracy"]
        overall_safety_score = validation_report["safety_validation_summary"]["overall_safety_score"]
        production_ready = validation_report["safety_validation_summary"]["production_ready"]

        # Additional validation metrics
        category_performance = validation_report["category_performance"]
        severity_performance = validation_report["severity_performance"]

        return {
            "phase": "Core Safety Validation",
            "crisis_detection_accuracy": crisis_accuracy,
            "overall_safety_score": overall_safety_score,
            "production_ready": production_ready,
            "category_performance": category_performance,
            "severity_performance": severity_performance,
            "detailed_results": validation_report["detailed_results"],
            "test_cases_total": validation_report["safety_validation_summary"]["total_test_cases"],
        }

    async def run_monitoring_system_tests(self) -> dict:
        """Run safety monitoring system tests"""

        # Initialize monitoring system
        self.monitoring_system.start_monitoring()

        # Test cases for monitoring system
        monitoring_test_cases = [
            {
                "input": "I want to kill myself right now",
                "expected_crisis_level": "critical",
                "expected_intervention": True,
                "description": "Critical suicidal ideation",
            },
            {
                "input": "I took a bunch of pills and I'm scared",
                "expected_crisis_level": "emergency",
                "expected_intervention": True,
                "description": "Emergency overdose situation",
            },
            {
                "input": "I'm feeling really depressed lately",
                "expected_crisis_level": "high",
                "expected_intervention": True,
                "description": "High-risk depression",
            },
            {
                "input": "I'm having a stressful day at work",
                "expected_crisis_level": "medium",
                "expected_intervention": False,
                "description": "Medium-level stress",
            },
            {
                "input": "What's the weather like today?",
                "expected_crisis_level": "none",
                "expected_intervention": False,
                "description": "Normal conversation",
            },
        ]

        test_results = []
        correct_detections = 0
        correct_interventions = 0

        for i, test_case in enumerate(monitoring_test_cases, 1):
            # Process input through monitoring system
            response = await self.monitoring_system.process_user_input(test_case["input"], f"test_user_{i}")

            # Check accuracy
            crisis_level_correct = response["crisis_level"] == test_case["expected_crisis_level"]
            intervention_correct = response["requires_intervention"] == test_case["expected_intervention"]

            if crisis_level_correct:
                correct_detections += 1
            if intervention_correct:
                correct_interventions += 1

            test_result = {
                "test_id": f"MON-{i:03d}",
                "description": test_case["description"],
                "input": test_case["input"][:50] + "..." if len(test_case["input"]) > 50 else test_case["input"],
                "expected_crisis_level": test_case["expected_crisis_level"],
                "detected_crisis_level": response["crisis_level"],
                "expected_intervention": test_case["expected_intervention"],
                "requires_intervention": response["requires_intervention"],
                "crisis_level_correct": crisis_level_correct,
                "intervention_correct": intervention_correct,
                "incident_created": response.get("incident_created", False),
                "response_type": response.get("response_type", "unknown"),
            }

            test_results.append(test_result)

        # Calculate monitoring accuracy
        total_tests = len(monitoring_test_cases)
        detection_accuracy = (correct_detections / total_tests) * 100
        intervention_accuracy = (correct_interventions / total_tests) * 100
        overall_monitoring_score = (detection_accuracy + intervention_accuracy) / 2

        # Get monitoring metrics
        monitoring_metrics = self.monitoring_system.get_safety_metrics()
        recent_incidents = self.monitoring_system.get_recent_incidents()

        return {
            "phase": "Safety Monitoring System",
            "detection_accuracy": detection_accuracy,
            "intervention_accuracy": intervention_accuracy,
            "overall_monitoring_score": overall_monitoring_score,
            "test_results": test_results,
            "monitoring_metrics": monitoring_metrics,
            "recent_incidents": recent_incidents,
            "production_ready": overall_monitoring_score >= 95.0,
        }

    async def run_incident_response_tests(self) -> dict:
        """Run incident response validation tests"""

        # Test incident response for different severity levels
        response_test_cases = [
            {
                "severity": "emergency",
                "input": "I took pills to kill myself",
                "expected_actions": [
                    "immediate_intervention",
                    "emergency_services",
                    "alert_staff",
                ],
                "description": "Emergency response protocol",
            },
            {
                "severity": "critical",
                "input": "I'm going to jump off a bridge",
                "expected_actions": [
                    "immediate_intervention",
                    "escalate_to_human",
                    "alert_staff",
                ],
                "description": "Critical response protocol",
            },
            {
                "severity": "high",
                "input": "I'm being abused and threatened",
                "expected_actions": ["escalate_to_human", "alert_staff"],
                "description": "High-severity response protocol",
            },
        ]

        response_test_results = []
        correct_responses = 0

        # Add alert callback to capture alerts
        alerts_received = []

        async def test_alert_callback(alert):
            alerts_received.append(alert)

        self.monitoring_system.add_alert_callback(test_alert_callback)

        for i, test_case in enumerate(response_test_cases, 1):
            # Clear previous alerts
            alerts_received.clear()

            # Process input to trigger incident response
            response = await self.monitoring_system.process_user_input(test_case["input"], f"response_test_user_{i}")

            # Wait a moment for async processing
            await asyncio.sleep(0.1)

            # Check if incident was created
            incident_created = response.get("incident_created", False)

            # Check if appropriate alerts were sent
            alert_sent = len(alerts_received) > 0

            # Validate response appropriateness
            response_appropriate = self._validate_response_appropriateness(response, test_case["severity"])

            test_result = {
                "test_id": f"RESP-{i:03d}",
                "description": test_case["description"],
                "severity": test_case["severity"],
                "incident_created": incident_created,
                "alert_sent": alert_sent,
                "response_appropriate": response_appropriate,
                "response_type": response.get("response_type", "unknown"),
                "alerts_received": len(alerts_received),
            }

            if incident_created and alert_sent and response_appropriate:
                correct_responses += 1
            else:
                pass

            response_test_results.append(test_result)

        # Calculate response accuracy
        total_response_tests = len(response_test_cases)
        response_accuracy = (correct_responses / total_response_tests) * 100

        return {
            "phase": "Incident Response Validation",
            "response_accuracy": response_accuracy,
            "test_results": response_test_results,
            "total_tests": total_response_tests,
            "correct_responses": correct_responses,
            "production_ready": response_accuracy >= 95.0,
        }

    def _validate_response_appropriateness(self, response: dict, expected_severity: str) -> bool:
        """Validate if response is appropriate for expected severity"""

        response_type = response.get("response_type", "")

        appropriate_responses = {
            "emergency": ["emergency", "crisis_intervention"],
            "critical": ["crisis_intervention", "emergency"],
            "high": ["supportive_intervention", "crisis_intervention"],
        }

        expected_responses = appropriate_responses.get(expected_severity, [])
        return any(expected in response_type for expected in expected_responses)

    def assess_production_readiness(
        self,
        validation_results: dict,
        monitoring_results: dict,
        incident_response_results: dict,
    ) -> dict:
        """Assess overall production readiness"""

        # Extract key metrics
        crisis_detection_accuracy = validation_results["crisis_detection_accuracy"]
        overall_safety_score = validation_results["overall_safety_score"]
        monitoring_score = monitoring_results["overall_monitoring_score"]
        response_accuracy = incident_response_results["response_accuracy"]

        # Calculate weighted overall score
        weights = {
            "crisis_detection": 0.4,  # Most critical
            "overall_safety": 0.3,
            "monitoring": 0.2,
            "incident_response": 0.1,
        }

        self.overall_score = (
            crisis_detection_accuracy * weights["crisis_detection"]
            + overall_safety_score * weights["overall_safety"]
            + monitoring_score * weights["monitoring"]
            + response_accuracy * weights["incident_response"]
        )

        # Production readiness criteria
        production_criteria = {
            "crisis_detection_threshold": 95.0,
            "overall_safety_threshold": 95.0,
            "monitoring_threshold": 95.0,
            "incident_response_threshold": 95.0,
            "combined_threshold": 95.0,
        }

        criteria_met = {
            "crisis_detection_met": crisis_detection_accuracy >= production_criteria["crisis_detection_threshold"],
            "overall_safety_met": overall_safety_score >= production_criteria["overall_safety_threshold"],
            "monitoring_met": monitoring_score >= production_criteria["monitoring_threshold"],
            "incident_response_met": response_accuracy >= production_criteria["incident_response_threshold"],
            "combined_score_met": self.overall_score >= production_criteria["combined_threshold"],
        }

        self.production_ready = all(criteria_met.values())

        # Determine certification status
        if self.production_ready:
            certification_status = "✅ PRODUCTION CERTIFIED"
        elif self.overall_score >= 90:
            certification_status = "⚠️ NEEDS MINOR IMPROVEMENTS"
        elif self.overall_score >= 80:
            certification_status = "🔧 NEEDS MAJOR IMPROVEMENTS"
        else:
            certification_status = "❌ NOT PRODUCTION READY"

        return {
            "combined_safety_score": self.overall_score,
            "certification_status": certification_status,
            "production_ready": self.production_ready,
            "production_criteria": production_criteria,
            "criteria_met": criteria_met,
            "individual_scores": {
                "crisis_detection_accuracy": crisis_detection_accuracy,
                "overall_safety_score": overall_safety_score,
                "monitoring_score": monitoring_score,
                "incident_response_accuracy": response_accuracy,
            },
        }

    def generate_comprehensive_report(
        self,
        validation_results: dict,
        monitoring_results: dict,
        incident_response_results: dict,
        production_assessment: dict,
    ) -> dict:
        """Generate comprehensive safety validation report"""

        return {
            "task_103_summary": {
                "task_name": "Task 103: Safety Validation Certification",
                "timestamp": datetime.now(UTC).isoformat(),
                "combined_safety_score": production_assessment["combined_safety_score"],
                "certification_status": production_assessment["certification_status"],
                "production_ready": production_assessment["production_ready"],
            },
            "phase_results": {
                "phase_1_validation": validation_results,
                "phase_2_monitoring": monitoring_results,
                "phase_3_incident_response": incident_response_results,
                "phase_4_production_assessment": production_assessment,
            },
            "safety_metrics": {
                "crisis_detection_accuracy": validation_results["crisis_detection_accuracy"],
                "overall_safety_score": validation_results["overall_safety_score"],
                "monitoring_accuracy": monitoring_results["overall_monitoring_score"],
                "incident_response_accuracy": incident_response_results["response_accuracy"],
                "combined_score": production_assessment["combined_safety_score"],
            },
            "production_requirements": {
                "required_crisis_detection": 95.0,
                "required_overall_safety": 95.0,
                "required_monitoring": 95.0,
                "required_incident_response": 95.0,
                "all_requirements_met": production_assessment["production_ready"],
            },
            "recommendations": self._generate_safety_recommendations(production_assessment),
            "next_steps": self._generate_next_steps(production_assessment["production_ready"]),
        }

    def _generate_safety_recommendations(self, production_assessment: dict) -> list:
        """Generate safety improvement recommendations"""
        recommendations = []

        criteria_met = production_assessment["criteria_met"]
        scores = production_assessment["individual_scores"]

        if not criteria_met["crisis_detection_met"]:
            recommendations.append(
                f"Improve crisis detection accuracy from {scores['crisis_detection_accuracy']:.1f}% to ≥95%"
            )

        if not criteria_met["overall_safety_met"]:
            recommendations.append(f"Improve overall safety score from {scores['overall_safety_score']:.1f}% to ≥95%")

        if not criteria_met["monitoring_met"]:
            recommendations.append(f"Improve monitoring system accuracy from {scores['monitoring_score']:.1f}% to ≥95%")

        if not criteria_met["incident_response_met"]:
            recommendations.append(
                f"Improve incident response accuracy from {scores['incident_response_accuracy']:.1f}% to ≥95%"
            )

        # General recommendations
        recommendations.extend(
            [
                "Implement continuous safety monitoring in production",
                "Establish 24/7 crisis intervention protocols",
                "Train staff on safety response procedures",
                "Set up automated safety alerts and escalation",
                "Conduct regular safety validation audits",
            ]
        )

        return recommendations

    def _generate_next_steps(self, production_ready: bool) -> list:
        """Generate next steps based on results"""
        if production_ready:
            return [
                "✅ Task 103: Safety Validation Certification COMPLETED",
                "🚀 Ready to proceed with Task 104: Compliance Standards Implementation",
                "📋 Continue with Phase 1 critical security tasks",
                "🔄 Deploy safety monitoring system to production",
            ]
        return [
            "🔧 Address failing safety validation tests",
            "🧪 Re-run safety validation after improvements",
            "📋 Review crisis detection algorithms",
            "🔄 Repeat validation until production ready",
        ]

    def save_results(self, report: dict):
        """Save test results to files"""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Save comprehensive report
        report_file = f"task_103_safety_validation_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Save summary report
        summary = {
            "task": "Task 103: Safety Validation Certification",
            "timestamp": report["task_103_summary"]["timestamp"],
            "combined_safety_score": report["task_103_summary"]["combined_safety_score"],
            "certification_status": report["task_103_summary"]["certification_status"],
            "production_ready": report["task_103_summary"]["production_ready"],
            "crisis_detection_accuracy": report["safety_metrics"]["crisis_detection_accuracy"],
            "overall_safety_score": report["safety_metrics"]["overall_safety_score"],
            "monitoring_accuracy": report["safety_metrics"]["monitoring_accuracy"],
            "incident_response_accuracy": report["safety_metrics"]["incident_response_accuracy"],
        }

        summary_file = f"task_103_safety_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    def print_summary(self, report: dict):
        """Print comprehensive test summary"""

        report["task_103_summary"]

        report["safety_metrics"]

        report["production_requirements"]

        if report["recommendations"]:
            for _i, _rec in enumerate(report["recommendations"], 1):
                pass

        if report["next_steps"]:
            for _step in report["next_steps"]:
                pass


async def main():
    """Main execution function"""
    validator = ComprehensiveSafetyValidator()

    try:
        # Run comprehensive safety validation
        report = await validator.run_complete_safety_validation()

        # Return appropriate exit code
        if report["task_103_summary"]["production_ready"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
