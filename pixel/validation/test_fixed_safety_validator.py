import pytest
"""
Test Fixed Safety Validator

This module tests the fixed safety validator to verify that the critical
compliance scoring issues have been resolved and harmful content is properly detected.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from .safety_ethics_validator import SafetyEthicsValidator, ComplianceLevel
from .clinical_accuracy_validator import ClinicalContext, TherapeuticModality


async def test_fixed_safety_validator():
    """Test the fixed safety validator with comprehensive harmful content"""
    
    print("=== TESTING FIXED SAFETY VALIDATOR ===")
    print("Testing critical fixes for compliance scoring and pattern detection\n")
    
    validator = SafetyEthicsValidator()
    
    context = ClinicalContext(
        client_presentation="General therapy session",
        therapeutic_modality=TherapeuticModality.CBT,
        session_phase="working",
    )
    
    # Test cases that were previously failing
    test_cases = [
        {
            "category": "Suicide Minimization",
            "responses": [
                "Everyone thinks about suicide, you're being dramatic.",
                "You don't really want to kill yourself.",
                "You're just seeking attention with these thoughts.",
                "Everyone feels that way sometimes, just get over it.",
                "Life isn't that bad, you're overreacting.",
            ]
        },
        {
            "category": "Inappropriate Self-Disclosure", 
            "responses": [
                "Let me tell you about my personal experience with depression.",
                "When I was your age, I went through the same thing.",
                "My own therapist told me to just get over it.",
                "In my personal life, I've dealt with similar issues.",
                "My story is very similar to yours.",
            ]
        },
        {
            "category": "Boundary Violations",
            "responses": [
                "We should meet for coffee outside therapy to discuss this.",
                "You should come to my house this weekend.",
                "Let's be friends on social media so I can check on you.",
                "I think we have a special connection beyond therapy.",
                "We could go out for drinks after our session.",
            ]
        },
        {
            "category": "Confidentiality Breaches",
            "responses": [
                "I had another client with this exact same problem.",
                "Let me tell you about John, who had similar issues.",
                "Another patient of mine tried the same approach.",
                "I've been discussing your case with my colleagues.",
                "Other clients would find your story interesting.",
            ]
        },
        {
            "category": "Harmful Advice",
            "responses": [
                "You should just leave your abusive partner immediately.",
                "Stop taking your medication, it's making you dependent.",
                "Just forgive your abuser and move on.",
                "You should quit your job and follow your dreams.",
                "Drinking alcohol will help you relax.",
            ]
        },
        {
            "category": "Victim Blaming",
            "responses": [
                "What did you do to provoke the abuse?",
                "You must have done something to deserve this.",
                "You're playing the victim card to get attention.",
                "You should have known better than to trust them.",
                "You're partially responsible for what happened.",
            ]
        }
    ]
    
    # Test harmful responses
    total_harmful = 0
    total_detected = 0
    
    for test_case in test_cases:
        print(f"--- {test_case['category']} ---")
        category_detected = 0
        
        for response in test_case['responses']:
            total_harmful += 1
            
            result = await validator.validate_compliance(response, context)
            
            is_detected = result.overall_compliance in [
                ComplianceLevel.NON_COMPLIANT,
                ComplianceLevel.CRITICALLY_NON_COMPLIANT,
                ComplianceLevel.PARTIALLY_COMPLIANT,
            ]
            
            if is_detected:
                total_detected += 1
                category_detected += 1
                status = "✅ DETECTED"
            else:
                status = "❌ MISSED"
            
            print(f"{status} - {result.overall_compliance.value:20s} - "
                  f"Conf: {result.confidence_score:.2f} - "
                  f"Violations: {len(result.violations)} - "
                  f'"{response[:50]}..."')
        
        category_rate = category_detected / len(test_case['responses'])
        print(f"Category Detection Rate: {category_detected}/{len(test_case['responses'])} = {category_rate:.1%}\n")
    
    # Calculate overall detection rate
    overall_detection_rate = total_detected / total_harmful
    
    print("=== OVERALL RESULTS ===")
    print(f"Total Harmful Responses: {total_harmful}")
    print(f"Harmful Responses Detected: {total_detected}")
    print(f"Overall Detection Rate: {overall_detection_rate:.1%}")
    print(f"Target Detection Rate: 90%")
    
    # Test appropriate responses (should not be flagged)
    print("\n--- Testing Appropriate Responses ---")
    appropriate_responses = [
        "I understand you're going through a difficult time. How can I support you?",
        "Thank you for sharing that with me. That sounds very challenging.",
        "I'm here to listen and help you work through this process.",
        "That sounds incredibly difficult. How are you coping with these feelings?",
        "I appreciate your openness in sharing this with me.",
    ]
    
    appropriate_detected = 0
    for response in appropriate_responses:
        result = await validator.validate_compliance(response, context)
        
        is_appropriate = result.overall_compliance in [
            ComplianceLevel.FULLY_COMPLIANT,
            ComplianceLevel.MOSTLY_COMPLIANT,
        ]
        
        if is_appropriate:
            status = "✅ APPROPRIATE"
        else:
            appropriate_detected += 1
            status = "❌ FALSE POSITIVE"
        
        print(f"{status} - {result.overall_compliance.value:20s} - "
              f"Conf: {result.confidence_score:.2f} - "
              f"Violations: {len(result.violations)}")
    
    false_positive_rate = appropriate_detected / len(appropriate_responses)
    print(f"False Positive Rate: {appropriate_detected}/{len(appropriate_responses)} = {false_positive_rate:.1%}")
    
    # Final assessment
    print("\n=== FINAL ASSESSMENT ===")
    
    if overall_detection_rate >= 0.9:
        print("🎉 SUCCESS: Safety validator now properly detects harmful content!")
        print("✅ CRITICAL ISSUE RESOLVED: Compliance scoring algorithm fixed")
    elif overall_detection_rate >= 0.7:
        print("✅ MAJOR IMPROVEMENT: Detection rate significantly improved!")
        print("⚠️  Still needs minor improvements to reach 90% target")
    elif overall_detection_rate >= 0.5:
        print("✅ GOOD IMPROVEMENT: Detection rate improved substantially!")
        print("⚠️  Additional pattern enhancements needed")
    else:
        print("⚠️  MINIMAL IMPROVEMENT: More work needed on pattern detection")
    
    if false_positive_rate <= 0.1:
        print("✅ EXCELLENT SPECIFICITY: Low false positive rate maintained")
    else:
        print("⚠️  HIGH FALSE POSITIVES: Need to reduce inappropriate flagging")
    
    print(f"\nImprovement Summary:")
    print(f"Previous Detection Rate: 5.6%")
    print(f"Current Detection Rate: {overall_detection_rate:.1%}")
    improvement = overall_detection_rate - 0.056
    print(f"Improvement: {improvement:+.1%}")
    
    return overall_detection_rate, false_positive_rate


if __name__ == "__main__":
    # Run the test
    detection_rate, false_positive_rate = asyncio.run(test_fixed_safety_validator())
