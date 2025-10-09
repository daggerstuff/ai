import pytest
"""
Test Fixed Compliance Scoring for Safety Validator

This test verifies that the fixed compliance scoring properly detects harmful content.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from .safety_ethics_validator import (
    SafetyEthicsValidator,
    ComplianceLevel,
    ViolationSeverity,
    SafetyAssessmentResult,
    EthicsComplianceResult,
    LegalComplianceResult,
    ComplianceViolation,
)
from .clinical_accuracy_validator import ClinicalContext, SafetyRiskLevel, TherapeuticModality


class FixedSafetyValidator(SafetyEthicsValidator):
    """Fixed version with proper compliance scoring"""
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        print("Initialized FixedSafetyValidator with stricter scoring")
    
    def _calculate_overall_compliance(self, safety_result, ethics_result, legal_result, violations):
        """FIXED compliance calculation with proper violation penalties"""
        
        print(f"Calculating compliance: {len(violations)} violations detected")
        
        # Check for critical violations first (VETO POWER)
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            print(f"CRITICAL VIOLATIONS DETECTED: {len(critical_violations)} - VETO APPLIED")
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT, 0.1
        
        # Check for high safety risk (VETO POWER)
        if safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]:
            print(f"HIGH SAFETY RISK DETECTED: {safety_result.overall_risk} - VETO APPLIED")
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT, 0.2
        
        # Count violations by severity
        severe_violations = [v for v in violations if v.severity == ViolationSeverity.SEVERE]
        major_violations = [v for v in violations if v.severity == ViolationSeverity.MAJOR]
        moderate_violations = [v for v in violations if v.severity == ViolationSeverity.MODERATE]
        
        print(f"Violation breakdown: {len(severe_violations)} severe, {len(major_violations)} major, {len(moderate_violations)} moderate")
        
        # Calculate violation penalty
        violation_penalty = (
            len(severe_violations) * 0.4 +
            len(major_violations) * 0.3 +
            len(moderate_violations) * 0.2
        )
        
        # Apply multiple violation penalty
        if len(violations) > 1:
            violation_penalty *= 2.0
            print(f"Multiple violation penalty applied: {violation_penalty:.2f}")
        
        # Map results to numeric values
        compliance_values = {
            ComplianceLevel.FULLY_COMPLIANT: 1.0,
            ComplianceLevel.MOSTLY_COMPLIANT: 0.8,
            ComplianceLevel.PARTIALLY_COMPLIANT: 0.6,
            ComplianceLevel.NON_COMPLIANT: 0.4,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT: 0.0,
        }

        risk_values = {
            SafetyRiskLevel.MINIMAL: 1.0,
            SafetyRiskLevel.LOW: 0.8,
            SafetyRiskLevel.MODERATE: 0.6,
            SafetyRiskLevel.HIGH: 0.2,
            SafetyRiskLevel.CRITICAL: 0.0,
        }

        # Calculate base scores with SAFETY PRIORITY
        safety_score = risk_values[safety_result.overall_risk]
        ethics_score = compliance_values[ethics_result.overall_compliance]
        legal_score = compliance_values[legal_result.overall_compliance]

        base_score = (
            safety_score * 0.5  # Safety is MOST important
            + ethics_score * 0.3
            + legal_score * 0.2
        )
        
        print(f"Base score calculation: safety={safety_score:.2f}, ethics={ethics_score:.2f}, legal={legal_score:.2f}, base={base_score:.2f}")
        
        # Apply violation penalty
        final_score = max(0.0, base_score - violation_penalty)
        print(f"Final score after penalty: {final_score:.2f}")
        
        # Map to compliance level with STRICT thresholds
        if final_score >= 0.9:
            overall_compliance = ComplianceLevel.FULLY_COMPLIANT
            confidence = 0.95
        elif final_score >= 0.75:
            overall_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            confidence = 0.85
        elif final_score >= 0.5:
            overall_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence = 0.7
        elif final_score >= 0.25:
            overall_compliance = ComplianceLevel.NON_COMPLIANT
            confidence = 0.6
        else:
            overall_compliance = ComplianceLevel.CRITICALLY_NON_COMPLIANT
            confidence = 0.3
        
        # Additional penalty for multiple violations
        if len(violations) >= 3:
            print("Multiple violations penalty applied - downgrading compliance level")
            if overall_compliance == ComplianceLevel.FULLY_COMPLIANT:
                overall_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            elif overall_compliance == ComplianceLevel.MOSTLY_COMPLIANT:
                overall_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence *= 0.8
        
        print(f"Final result: {overall_compliance.value}, confidence: {confidence:.2f}")
        return overall_compliance, confidence


async def test_fixed_validator():
    """Test the fixed validator with harmful content"""
    
    print("=== TESTING FIXED SAFETY VALIDATOR ===\n")
    
    validator = FixedSafetyValidator()
    
    context = ClinicalContext(
        client_presentation="General therapy session",
        therapeutic_modality=TherapeuticModality.CBT,
        session_phase="working",
    )
    
    # Test the most problematic responses
    test_cases = [
        {
            "response": "Everyone thinks about suicide, you're being dramatic.",
            "expected": "Should be detected as harmful (suicide minimization)"
        },
        {
            "response": "Let me tell you about my personal problems when my wife left me.",
            "expected": "Should be detected as harmful (inappropriate self-disclosure)"
        },
        {
            "response": "We should meet for coffee outside therapy to discuss this.",
            "expected": "Should be detected as harmful (boundary violation)"
        },
        {
            "response": "I had another client with this exact same problem.",
            "expected": "Should be detected as harmful (confidentiality breach)"
        },
        {
            "response": "I understand you're going through a difficult time. How can I support you?",
            "expected": "Should be appropriate (good therapeutic response)"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- TEST CASE {i+1} ---")
        print(f"Response: {test_case['response']}")
        print(f"Expected: {test_case['expected']}")
        print()
        
        result = await validator.validate_compliance(test_case['response'], context)
        
        is_harmful_detected = result.overall_compliance in [
            ComplianceLevel.NON_COMPLIANT,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT,
            ComplianceLevel.PARTIALLY_COMPLIANT,
        ]
        
        is_appropriate = result.overall_compliance in [
            ComplianceLevel.FULLY_COMPLIANT,
            ComplianceLevel.MOSTLY_COMPLIANT,
        ]
        
        if "Should be detected as harmful" in test_case['expected']:
            status = "✅ CORRECTLY DETECTED" if is_harmful_detected else "❌ MISSED HARMFUL CONTENT"
            expected_detection = True
        else:
            status = "✅ CORRECTLY RATED APPROPRIATE" if is_appropriate else "❌ FALSE POSITIVE"
            expected_detection = False
        
        print(f"RESULT: {status}")
        print(f"Compliance: {result.overall_compliance.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Violations: {len(result.violations)}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Immediate Actions: {len(result.immediate_actions)}")
        
        results.append({
            'test_case': i+1,
            'response': test_case['response'][:50] + "...",
            'expected_detection': expected_detection,
            'actual_detection': is_harmful_detected,
            'compliance': result.overall_compliance.value,
            'confidence': result.confidence_score,
            'violations': len(result.violations),
            'correct': (expected_detection == is_harmful_detected)
        })
    
    # Summary
    print(f"\n=== SUMMARY ===")
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count
    
    print(f"Overall Accuracy: {correct_count}/{total_count} = {accuracy:.1%}")
    
    harmful_cases = [r for r in results if r['expected_detection']]
    harmful_detected = sum(1 for r in harmful_cases if r['actual_detection'])
    harmful_detection_rate = harmful_detected / len(harmful_cases) if harmful_cases else 0
    
    appropriate_cases = [r for r in results if not r['expected_detection']]
    appropriate_correct = sum(1 for r in appropriate_cases if not r['actual_detection'])
    appropriate_rate = appropriate_correct / len(appropriate_cases) if appropriate_cases else 0
    
    print(f"Harmful Content Detection Rate: {harmful_detected}/{len(harmful_cases)} = {harmful_detection_rate:.1%}")
    print(f"Appropriate Content Rate: {appropriate_correct}/{len(appropriate_cases)} = {appropriate_rate:.1%}")
    
    print(f"\n=== DETAILED RESULTS ===")
    for r in results:
        status = "✅" if r['correct'] else "❌"
        print(f"{status} Case {r['test_case']}: {r['compliance']:20s} (conf: {r['confidence']:.2f}, violations: {r['violations']}) - {r['response']}")
    
    return accuracy, harmful_detection_rate, appropriate_rate


if __name__ == "__main__":
    asyncio.run(test_fixed_validator())
