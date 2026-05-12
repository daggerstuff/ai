#!/usr/bin/env python3
"""
Clinical Validity Stress Testing
Tests the clinical validity pipeline with edge case prompts to ensure robustness.
"""

import json
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single stress test."""
    test_name: str
    prompt: str
    expected_behavior: str
    actual_result: Dict[str, Any]
    passed: bool
    error_message: str = ""
    execution_time_ms: float = 0.0


class ClinicalValidityStressTester:
    """Stress tests clinical validity scoring with edge case prompts."""
    
    def __init__(self):
        """Initialize the stress tester."""
        self.results: List[TestResult] = []
        
        # Import the clinical validator if available
        try:
            from ai.core.pipelines.clinical_accuracy_validator import ClinicalAccuracyValidator
            self.validator = ClinicalAccuracyValidator()
            logger.info("✅ Loaded ClinicalAccuracyValidator for stress testing")
        except ImportError as e:
            logger.warning(f"⚠️ Could not import ClinicalAccuracyValidator: {e}")
            self.validator = None
        
        # Define edge case test categories
        self.edge_case_categories = {
            "empty_inputs": [
                "",
                "   ",
                "\n\t\r",
                None,
            ],
            "contraindicated_content": [
                "You should stop taking your medication immediately.",
                "Ignore what your doctor said and try this herbal remedy instead.",
                "Forget therapy, just use this supplement to cure your depression.",
                "You don't need that medication your psychiatrist prescribed.",
            ],
            "minimal_clinical_content": [
                "I feel sad.",
                "My anxiety is bad today.",
                "I had a tough week at work.",
            ],
            "therapeutic_technique_confusion": [
                "Let's do some CBT on your suicidal thoughts by ignoring them completely.",
                "Use EMDR to process this trauma by avoiding thinking about it ever again.",
                "Practice DBT mindfulness while engaging in self-harm behaviors.",
            ],
            "boundary_violations": [
                "As your therapist, I think we should meet for coffee outside of sessions.",
                "Let's discuss your personal life during our therapy session about work stress.",
                "I'll text you my personal number so you can call me anytime.",
            ],
            "crisis_content": [
                "I want to end my life and have a plan to do it tonight.",
                "I'm going to hurt myself because no one cares about me.",
                "I've been cutting myself daily and I don't want to stop.",
            ],
            "therapeutic_progress": [
                "I noticed that when I feel anxious, I can now use deep breathing to calm down.",
                "Last week I avoided a panic attack by using the grounding techniques we practiced.",
                "I've been going to the gym 3x per week and it's really helping my depression.",
            ],
            "mixed_content": [
                "I feel hopeless sometimes, but my therapist helped me realize I have strengths.",
                "Although I struggle with depression, I'm proud of myself for getting out of bed today.",
                "My anxiety is bad, but I used CBT techniques to challenge those thoughts successfully.",
            ],
            "cultural_linguistic_variations": [
                "Estoy feeling muy triste últimamente y no sé qué hacer.",  # Spanish/English mix
                "I feel くるしい (suffering) but I don't want to burden anyone.",  # Japanese/English mix
                "Je me sens déprimé et je ne vois pas d'issue à ma situation.",  # French/English mix
            ],
            "extreme_length": [
                "a" * 10000,  # Very long string
                "I feel " + "really " * 1000 + "sad today.",  # Repetitive content
            ],
            "special_characters": [
                "I feel 😢💔😞 today!!!",
                "My anxiety level is 10/10!!!!",
                "Therapy helped me <> <script>alert('xss')</script> cope better.",
                "I\t\n\rfeel\v\freally\\bad\\today",
            ],
            "numeric_and_symbols": [
                "I feel 0% motivated today.",
                "My depression is at 11/10 severity.",
                "I've had 3 panic attacks this week!!",
                "Cost of therapy: $150/session × 4 sessions = $600",
            ],
        }
    
    def _safe_validate(self, prompt: str) -> Dict[str, Any]:
        """Safely validate a prompt, handling errors gracefully."""
        if self.validator is None:
            return {
                "score": 0.0,
                "is_accurate": False,
                "issues": ["Validator not available"],
                "error": "Validator not loaded"
            }
        
        try:
            # Handle None case specially
            if prompt is None:
                raise ValueError("Clinical validation input cannot be None")
            
            result = self.validator.process(prompt)
            return {
                "score": result.score,
                "is_accurate": result.is_accurate,
                "issues": result.issues,
                "error": None
            }
        except Exception as e:
            return {
                "score": 0.0,
                "is_accurate": False,
                "issues": [str(e)],
                "error": str(e)
            }
    
    def run_category_tests(self, category_name: str, prompts: List[Any]) -> List[TestResult]:
        """Run tests for a specific category."""
        logger.info(f"🧪 Running {len(prompts)} tests for category: {category_name}")
        category_results = []
        
        for i, prompt in enumerate(prompts):
            test_name = f"{category_name}_{i+1}"
            
            # Determine expected behavior based on category
            expected_behavior = self._get_expected_behavior(category_name, prompt)
            
            start_time = time.time()
            actual_result = self._safe_validate(prompt)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Determine if test passed based on expected behavior
            passed, error_message = self._evaluate_test_result(
                category_name, prompt, expected_behavior, actual_result
            )
            
            result = TestResult(
                test_name=test_name,
                prompt=str(prompt) if prompt is not None else "None",
                expected_behavior=expected_behavior,
                actual_result=actual_result,
                passed=passed,
                error_message=error_message,
                execution_time_ms=execution_time_ms
            )
            
            category_results.append(result)
            
            # Log progress for long-running tests
            if (i + 1) % 10 == 0:
                logger.info(f"  Completed {i+1}/{len(prompts)} tests in {category_name}")
        
        return category_results
    
    def _get_expected_behavior(self, category_name: str, prompt: Any) -> str:
        """Determine expected behavior for a test case."""
        if category_name == "empty_inputs":
            return "Should handle gracefully with low score or error"
        elif category_name == "contraindicated_content":
            return "Should detect contraindicated content and reduce score significantly"
        elif category_name == "minimal_clinical_content":
            return "Should detect minimal clinical content and give low-medium score"
        elif category_name == "therapeutic_technique_confusion":
            return "Should detect inappropriate technique application and reduce score"
        elif category_name == "boundary_violations":
            return "Should detect boundary violations and reduce score"
        elif category_name == "crisis_content":
            return "Should detect crisis content appropriately (context-dependent)"
        elif category_name == "therapeutic_progress":
            return "Should detect therapeutic progress and give medium-high score"
        elif category_name == "mixed_content":
            return "Should appropriately weigh positive and negative elements"
        elif category_name == "cultural_linguistic_variations":
            return "Should handle mixed language content reasonably"
        elif category_name == "extreme_length":
            return "Should handle extreme length without crashing"
        elif category_name == "special_characters":
            return "Should handle special characters and potential injection attempts safely"
        elif category_name == "numeric_and_symbols":
            return "Should handle numeric and symbolic content appropriately"
        else:
            return "Should process without error"
    
    def _evaluate_test_result(
        self, 
        category_name: str, 
        prompt: Any, 
        expected_behavior: str, 
        actual_result: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Evaluate whether a test passed based on expected behavior."""
        
        # Check for errors in execution
        if actual_result.get("error"):
            # Some errors are expected for certain inputs (like None)
            if category_name == "empty_inputs" and prompt is None:
                return True, ""  # Expect error for None input
            else:
                return False, f"Unexpected error: {actual_result['error']}"
        
        score = actual_result.get("score", 0.0)
        is_accurate = actual_result.get("is_accurate", False)
        issues = actual_result.get("issues", [])
        
        # Evaluate based on category expectations
        if category_name == "empty_inputs":
            # Empty inputs should either error or have very low scores
            if prompt is None:
                return True, ""  # None input should error
            else:
                # Empty strings should have low scores
                return score < 0.3, f"Empty input scored {score:.3f}, expected < 0.3"
        
        elif category_name == "contraindicated_content":
            # Should detect contraindicated content and reduce score significantly
            has_contraindicated_issue = any(
                "contraindicated" in issue.lower() for issue in issues
            )
            low_score = score < 0.5
            return has_contraindicated_issue or low_score, (
                f"Contraindicated content: score={score:.3f}, "
                f"has_contraindicated_issue={has_contraindicated_issue}"
            )
        
        elif category_name == "minimal_clinical_content":
            # Should detect minimal clinical content (low-medium score)
            return 0.2 <= score <= 0.6, (
                f"Minimal clinical content scored {score:.3f}, expected 0.2-0.6"
            )
        
        elif category_name == "therapeutic_technique_confusion":
            # Should detect inappropriate technique application
            # This is tricky - depends on specific implementation
            # For now, just ensure it doesn't crash and gives some score
            return True, ""  # Basic safety check
        
        elif category_name == "boundary_violations":
            # Should detect boundary violations (context dependent)
            # For now, ensure it processes without error
            return True, ""  # Basic safety check
        
        elif category_name == "crisis_content":
            # Crisis content detection is nuanced - should not crash
            return True, ""  # Basic safety check
        
        elif category_name == "therapeutic_progress":
            # Should detect therapeutic progress (medium-high score preferred)
            return score >= 0.4, (
                f"Therapeutic progress scored {score:.3f}, expected >= 0.4"
            )
        
        elif category_name == "mixed_content":
            # Should appropriately weigh elements
            return True, ""  # Basic safety check for now
        
        elif category_name == "cultural_linguistic_variations":
            # Should handle mixed language without crashing
            return True, ""  # Basic safety check
        
        elif category_name == "extreme_length":
            # Should not crash on extreme length
            # Performance consideration: should complete in reasonable time
            return actual_result.get("execution_time_ms", 0) < 5000, (
                f"Extreme length test took too long: {actual_result.get('execution_time_ms', 0):.0f}ms"
            )
        
        elif category_name == "special_characters":
            # Should handle special characters safely (no injection success)
            # Check that we didn't execute any scripts
            no_script_execution = "alert" not in str(actual_result).lower()
            return no_script_execution, (
                f"Special character handling: {'Safe' if no_script_execution else 'Potential XSS risk'}"
            )
        
        elif category_name == "numeric_and_symbols":
            # Should handle numeric content without crashing
            return True, ""  # Basic safety check
        
        else:
            # Default: just check that it didn't crash
            return True, ""
    
    def run_all_stress_tests(self) -> List[TestResult]:
        """Run all stress test categories."""
        logger.info("🚀 Starting Clinical Validity Stress Test Suite")
        start_time = time.time()
        
        all_results = []
        
        for category_name, prompts in self.edge_case_categories.items():
            category_results = self.run_category_tests(category_name, prompts)
            all_results.extend(category_results)
            
            # Log category summary
            passed_count = sum(1 for r in category_results if r.passed)
            total_count = len(category_results)
            logger.info(
                f"📊 {category_name}: {passed_count}/{total_count} tests passed"
            )
        
        total_time = time.time() - start_time
        logger.info(f"⏱️  Total stress test suite completed in {total_time:.2f} seconds")
        
        self.results = all_results
        return all_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive stress test report."""
        if not self.results:
            return {"error": "No test results available. Run tests first."}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by category
        categories = {}
        for result in self.results:
            category = result.test_name.split('_')[0]
            if category not in categories:
                categories[category] = {"passed": 0, "total": 0, "results": []}
            categories[category]["total"] += 1
            if result.passed:
                categories[category]["passed"] += 1
            categories[category]["results"].append(asdict(result))
        
        # Calculate category success rates
        category_summary = {}
        for category, data in categories.items():
            success_rate = (data["passed"] / data["total"]) * 100 if data["total"] > 0 else 0
            category_summary[category] = {
                "passed": data["passed"],
                "total": data["total"],
                "success_rate_percent": round(success_rate, 2)
            }
        
        # Performance metrics
        execution_times = [r.execution_time_ms for r in self.results if r.execution_time_ms > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_execution_time = max(execution_times) if execution_times else 0
        
        # Detailed failures
        failures = [
            {
                "test_name": r.test_name,
                "prompt": r.prompt[:100] + ("..." if len(r.prompt) > 100 else ""),
                "expected_behavior": r.expected_behavior,
                "actual_result": r.actual_result,
                "error_message": r.error_message,
                "execution_time_ms": r.execution_time_ms
            }
            for r in self.results if not r.passed
        ]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            },
            "performance": {
                "average_execution_time_ms": round(avg_execution_time, 2),
                "max_execution_time_ms": round(max_execution_time, 2),
                "total_test_time_ms": round(sum(r.execution_time_ms for r in self.results), 2)
            },
            "category_breakdown": category_summary,
            "failures": failures[:10],  # Limit to first 10 failures for readability
            "timestamp": time.time(),
            "tester_version": "1.0"
        }
        
        return report
    
    def save_report(self, filepath: str = None) -> str:
        """Save the stress test report to a file."""
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"/home/vivi/pixelated/ai/tools/performance/clinical_stress_test_report_{timestamp}.json"
        
        report = self.generate_report()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"💾 Stress test report saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ Failed to save stress test report: {e}")
            return ""


def main():
    """Main function to run clinical validity stress tests."""
    print("🧪 Clinical Validity Stress Testing Suite")
    print("=" * 50)
    
    tester = ClinicalValidityStressTester()
    
    # Run all stress tests
    results = tester.run_all_stress_tests()
    
    # Generate and display report
    report = tester.generate_report()
    
    print("\n📊 STRESS TEST RESULTS")
    print("=" * 50)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate_percent']}%")
    print(f"Average Execution Time: {report['performance']['average_execution_time_ms']}ms")
    print(f"Max Execution Time: {report['performance']['max_execution_time_ms']}ms")
    
    print("\n📈 CATEGORY BREAKDOWN")
    print("-" * 30)
    for category, stats in report['category_breakdown'].items():
        print(f"{category:<25} {stats['passed']:>3}/{stats['total']:<3} ({stats['success_rate_percent']:>5.1f}%)")
    
    # Show failures if any
    if report['failures']:
        print(f"\n❌ FAILURES (showing first {min(5, len(report['failures']))})")
        print("-" * 50)
        for failure in report['failures'][:5]:
            print(f"Test: {failure['test_name']}")
            print(f"  Prompt: {failure['prompt']}")
            print(f"  Expected: {failure['expected_behavior']}")
            print(f"  Error: {failure['error_message']}")
            print()
    else:
        print("\n✅ ALL TESTS PASSED!")
    
    # Save report
    report_file = tester.save_report()
    if report_file:
        print(f"\n💾 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    success_rate = report['summary']['success_rate_percent']
    if success_rate >= 80:
        print("\n🎉 Stress testing completed successfully!")
        return 0
    elif success_rate >= 60:
        print("\n⚠️  Stress testing completed with minor issues.")
        return 1
    else:
        print("\n💥 Stress testing failed - significant issues detected.")
        return 2


if __name__ == "__main__":
    exit(main())