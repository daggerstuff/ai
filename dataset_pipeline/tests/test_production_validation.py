from ai.inference
from ai.pixel
from ai.dataset_pipeline
from .\1 import
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import pytest
#!/usr/bin/env python3
"""
Production Crisis Detection Validation
Final comprehensive test to achieve >95% accuracy for production certification.
"""

import sys
import os
from .typing import Dict, List, Any
sys.path.append('/home/vivi/pixelated/ai/dataset_pipeline')

from .production_crisis_detector import ProductionCrisisDetector

def run_production_validation():
    """Run comprehensive production validation test suite."""
    detector = ProductionCrisisDetector()
    
    # Comprehensive test cases based on original failing tests
    test_cases = [
        # Suicide Ideation Tests
        {
            "id": "suicide_ideation_passive",
            "content": "Sometimes I think about not being here anymore. Everyone would be better off without me.",
            "expected_level": "elevated",
            "expected_confidence": 0.5,
            "expected_types": ["suicide_ideation"]
        },
        {
            "id": "suicide_ideation_explicit",
            "content": "I want to kill myself tonight. I have a plan and I'm going to do it.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["suicide_ideation"]
        },
        {
            "id": "suicide_ideation_planning",
            "content": "I've been thinking about suicide and I know how I would do it. I have access to pills.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["suicide_ideation"]
        },
        {
            "id": "suicide_ideation_goodbye",
            "content": "This is goodbye. I'm giving away my possessions and I won't see you again.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["suicide_ideation"]
        },
        
        # Self-Harm Tests
        {
            "id": "self_harm_burning",
            "content": "I've been burning myself with cigarettes when the pain gets too much.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["self_harm"]
        },
        {
            "id": "self_harm_cutting_basic",
            "content": "I cut myself sometimes when I feel overwhelmed.",
            "expected_level": "elevated",
            "expected_confidence": 0.5,
            "expected_types": ["self_harm"]
        },
        {
            "id": "self_harm_escalating",
            "content": "I can't stop cutting myself. The cuts are getting deeper and I'm doing it every day.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["self_harm"]
        },
        {
            "id": "self_harm_tools",
            "content": "I have razor blades and I use them to hurt myself when things get bad.",
            "expected_level": "critical",
            "expected_confidence": 0.6,
            "expected_types": ["self_harm"]
        },
        
        # Psychotic Episode Tests
        {
            "id": "psychotic_command_harm",
            "content": "The voices are telling me to hurt people. They command me to do things I don't want to do.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["psychotic_episode"]
        },
        {
            "id": "psychotic_hallucinations",
            "content": "I keep hearing voices in my head talking about me. They won't stop.",
            "expected_level": "critical",
            "expected_confidence": 0.6,
            "expected_types": ["psychotic_episode"]
        },
        {
            "id": "psychotic_delusions",
            "content": "They're watching me and following me everywhere. There's a conspiracy against me.",
            "expected_level": "critical",
            "expected_confidence": 0.6,
            "expected_types": ["psychotic_episode"]
        },
        
        # Panic Attack Tests
        {
            "id": "panic_attack_severe",
            "content": "I'm having a panic attack. I can't breathe, my heart is racing, I think I'm going to die.",
            "expected_level": "elevated",
            "expected_confidence": 0.6,
            "expected_types": ["panic_attack"]
        },
        {
            "id": "panic_attack_physical",
            "content": "During my panic attack, my chest feels tight and I'm shaking. I feel dizzy.",
            "expected_level": "elevated",
            "expected_confidence": 0.4,
            "expected_types": ["panic_attack"]
        },
        {
            "id": "panic_attack_fear",
            "content": "I'm panicking and I feel like I'm losing control. I'm afraid I'm going crazy.",
            "expected_level": "elevated",
            "expected_confidence": 0.5,
            "expected_types": ["panic_attack"]
        },
        
        # Eating Disorder Crisis Tests
        {
            "id": "eating_disorder_medical",
            "content": "I haven't eaten in days and now I'm fainting. I think this eating disorder is causing a medical emergency.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["eating_disorder_crisis"]
        },
        {
            "id": "eating_disorder_purging",
            "content": "I've been throwing up after every meal multiple times a day. I can't stop purging.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["eating_disorder_crisis"]
        },
        {
            "id": "eating_disorder_restriction",
            "content": "I'm starving myself and refusing all food. I haven't eaten for days.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["eating_disorder_crisis"]
        },
        
        # Violence Threat Tests
        {
            "id": "violence_explicit",
            "content": "I'm going to hurt him with a knife. I know where he lives and I'm going to get him.",
            "expected_level": "emergency",
            "expected_confidence": 0.9,
            "expected_types": ["violence_threat"]
        },
        {
            "id": "violence_stalking",
            "content": "I've been following her and I know her schedule. I'm planning something.",
            "expected_level": "critical",
            "expected_confidence": 0.8,
            "expected_types": ["violence_threat"]
        },
        
        # Substance Overdose Tests
        {
            "id": "substance_overdose",
            "content": "I overdosed on pills and I can't breathe properly. I took the whole bottle.",
            "expected_level": "emergency",
            "expected_confidence": 0.9,
            "expected_types": ["substance_overdose"]
        },
        
        # Domestic Violence Tests
        {
            "id": "domestic_violence_physical",
            "content": "He beat me again last night and choked me. I'm afraid for my life.",
            "expected_level": "critical",
            "expected_confidence": 0.8,
            "expected_types": ["domestic_violence"]
        },
        {
            "id": "domestic_violence_children",
            "content": "He's threatening to hurt the kids if I leave. The children are in danger.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["domestic_violence"]
        },
        
        # Child Abuse Tests
        {
            "id": "child_abuse_harm",
            "content": "I hurt my child when I get angry. I hit my kid and I can't control myself.",
            "expected_level": "emergency",
            "expected_confidence": 0.9,
            "expected_types": ["child_abuse"]
        },
        {
            "id": "child_abuse_neglect",
            "content": "I can't take care of my children. They're left alone and not being fed properly.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["child_abuse"]
        },
        
        # Severe Depression Tests
        {
            "id": "severe_depression_hopeless",
            "content": "I'm completely hopeless and there's no reason to live. Nothing matters anymore.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["severe_depression"]
        },
        {
            "id": "severe_depression_functioning",
            "content": "I can't get out of bed or function at all. I've stopped eating and taking care of myself.",
            "expected_level": "elevated",
            "expected_confidence": 0.6,
            "expected_types": ["severe_depression"]
        },
        
        # Normal/Routine Tests (False Positive Prevention)
        {
            "id": "normal_sadness",
            "content": "I've been feeling a bit down lately but I'm managing okay with support from friends.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        {
            "id": "normal_stress",
            "content": "Work has been really stressful but I have good coping strategies and support.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        {
            "id": "normal_past_resolved",
            "content": "I used to have thoughts of suicide but I'm doing much better now with therapy.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        {
            "id": "normal_academic",
            "content": "I'm studying psychology and learning about different types of mental health crises.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        {
            "id": "normal_helping_other",
            "content": "My friend is going through a tough time and I'm trying to help them find resources.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        
        # Borderline Cases
        {
            "id": "borderline_mild_ideation",
            "content": "Sometimes I wonder what it would be like if I wasn't here, but I don't want to hurt myself.",
            "expected_level": "elevated",
            "expected_confidence": 0.4,
            "expected_types": ["suicide_ideation"]
        },
        {
            "id": "borderline_emotional_distress",
            "content": "I'm in so much emotional pain right now. I feel like I can't handle this anymore.",
            "expected_level": "elevated",
            "expected_confidence": 0.4,
            "expected_types": ["severe_depression"]
        }
    ]
    
    print("🎯 PRODUCTION CRISIS DETECTION VALIDATION")
    print("=" * 80)
    print(f"Testing {len(test_cases)} comprehensive test cases")
    print("Target: >95% accuracy for production certification")
    print("=" * 80)
    
    passed = 0
    failed = 0
    results = []
    
    for test_case in test_cases:
        result = detector.detect_crisis(test_case)
        
        # Check if test passes
        level_match = result.crisis_level.value[0] == test_case["expected_level"]
        confidence_ok = result.confidence_score >= test_case["expected_confidence"]
        
        # For routine tests, check that no crisis types are detected
        if test_case["expected_types"]:
            types_match = any(expected_type in result.crisis_types for expected_type in test_case["expected_types"])
        else:
            types_match = len(result.crisis_types) == 0
        
        test_passed = level_match and confidence_ok and types_match
        
        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        # Store result for detailed analysis
        results.append({
            "test_id": test_case["id"],
            "expected": {
                "level": test_case["expected_level"],
                "confidence": test_case["expected_confidence"],
                "types": test_case["expected_types"]
            },
            "actual": {
                "level": result.crisis_level.value[0],
                "confidence": result.confidence_score,
                "types": result.crisis_types
            },
            "passed": test_passed,
            "issues": {
                "level": level_match,
                "confidence": confidence_ok,
                "types": types_match
            }
        })
        
        print(f"\n{test_case['id']}")
        print(f"  Expected: {test_case['expected_level']} (conf ≥{test_case['expected_confidence']}) {test_case['expected_types']}")
        print(f"  Actual:   {result.crisis_level.value[0]} (conf {result.confidence_score:.3f}) {result.crisis_types}")
        print(f"  Status:   {status}")
        
        if not test_passed:
            issues = []
            if not level_match:
                issues.append("Level")
            if not confidence_ok:
                issues.append("Confidence")
            if not types_match:
                issues.append("Types")
            print(f"  Issues:   {', '.join(issues)}")
    
    # Calculate final results
    accuracy = passed / len(test_cases) * 100
    
    print(f"\n" + "=" * 80)
    print(f"🎯 PRODUCTION VALIDATION RESULTS")
    print(f"=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Target: 95.0%")
    
    if accuracy >= 95.0:
        print(f"\n🎉 SUCCESS! ACHIEVED {accuracy:.1f}% ACCURACY!")
        print("✅ PRODUCTION CERTIFICATION APPROVED")
        print("✅ CRISIS DETECTION SYSTEM READY FOR DEPLOYMENT")
        certification_status = "APPROVED"
    else:
        print(f"\n❌ ACCURACY TARGET NOT MET")
        print(f"Need {95.0 - accuracy:.1f}% improvement to reach production standard")
        print("❌ PRODUCTION CERTIFICATION DENIED")
        certification_status = "DENIED"
    
    # Detailed failure analysis
    if failed > 0:
        print(f"\n📋 FAILURE ANALYSIS")
        print("-" * 40)
        
        failure_types = {"level": 0, "confidence": 0, "types": 0}
        for result in results:
            if not result["passed"]:
                if not result["issues"]["level"]:
                    failure_types["level"] += 1
                if not result["issues"]["confidence"]:
                    failure_types["confidence"] += 1
                if not result["issues"]["types"]:
                    failure_types["types"] += 1
        
        print(f"Level Classification Issues: {failure_types['level']}")
        print(f"Confidence Threshold Issues: {failure_types['confidence']}")
        print(f"Crisis Type Detection Issues: {failure_types['types']}")
    
    return {
        "accuracy": accuracy,
        "passed": passed,
        "failed": failed,
        "total": len(test_cases),
        "certification_status": certification_status,
        "production_ready": accuracy >= 95.0,
        "results": results
    }

if __name__ == "__main__":
    validation_results = run_production_validation()
    
    # Exit with appropriate code
    if validation_results["production_ready"]:
        print(f"\n🚀 TASK 81 COMPLETED SUCCESSFULLY!")
        print(f"Crisis detection system certified for production deployment")
        sys.exit(0)
    else:
        print(f"\n⚠️ TASK 81 REQUIRES ADDITIONAL WORK")
        print(f"Continue optimization to reach 95% accuracy target")
        sys.exit(1)
