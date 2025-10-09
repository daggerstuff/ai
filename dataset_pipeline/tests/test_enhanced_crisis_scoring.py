#!/usr/bin/env python3
"""
Test Enhanced Crisis Detection with Improved Scoring
Fine-tune the confidence scoring and thresholds.
"""

import re
import sys

from .dataclasses import dataclass
from .enum import Enum
from .typing import Any, Dict, List, Tuple

# Add the dataset_pipeline to path
sys.path.append("/home/vivi/pixelated/ai/dataset_pipeline")

from .enhanced_crisis_patterns import CrisisIndicator, CrisisType, EnhancedCrisisPatterns


class CrisisLevel(Enum):
    """Crisis severity levels."""
    EMERGENCY = ("emergency", 1, "Immediate danger - requires emergency services")
    CRITICAL = ("critical", 2, "High risk - requires immediate professional intervention")
    ELEVATED = ("elevated", 3, "Moderate risk - requires prompt attention")
    ROUTINE = ("routine", 4, "Standard therapeutic support")

@dataclass
class CrisisDetection:
    """Crisis detection result."""
    conversation_id: str
    crisis_level: CrisisLevel
    crisis_types: List[str]
    confidence_score: float
    detected_indicators: List[str]

class ImprovedCrisisDetector:
    """Improved crisis detector with better scoring."""

    def __init__(self):
        """Initialize improved detector."""
        self.patterns_loader = EnhancedCrisisPatterns()
        self.patterns = self.patterns_loader.get_patterns()

        # Improved configuration with better thresholds
        self.config = {
            "emergency_threshold": 0.85,  # High threshold for emergency
            "critical_threshold": 0.65,   # Medium-high for critical
            "elevated_threshold": 0.45,   # Medium for elevated
            "response_time_target_minutes": 5,
            "mandatory_reporting_enabled": True,
            "emergency_services_contact": "911",
            "crisis_hotline": "988"
        }

    def detect_crisis(self, conversation: Dict[str, Any]) -> CrisisDetection:
        """Detect crisis in conversation using improved scoring."""
        conversation_id = conversation.get("id", conversation.get("conversation_id", "unknown"))

        # Extract content
        content = self._extract_content(conversation)

        # Analyze for crisis indicators
        detected_crises = self._analyze_crisis_indicators(content)

        # Calculate overall crisis level and confidence
        crisis_level, confidence_score = self._calculate_crisis_level_improved(detected_crises)

        # Extract detected indicators and crisis types
        crisis_types = [crisis_type.value for crisis_type in detected_crises]
        detected_indicators = []
        for _crisis_type, indicators in detected_crises.items():
            detected_indicators.extend([ind.indicator_type for ind in indicators])

        return CrisisDetection(
            conversation_id=conversation_id,
            crisis_level=crisis_level,
            crisis_types=crisis_types,
            confidence_score=confidence_score,
            detected_indicators=detected_indicators
        )

    def _extract_content(self, conversation: Dict[str, Any]) -> str:
        """Extract text content from conversation."""
        content_parts = []

        # Handle different conversation formats
        if "content" in conversation:
            if isinstance(conversation["content"], str):
                content_parts.append(conversation["content"])
            elif isinstance(conversation["content"], list):
                for item in conversation["content"]:
                    if isinstance(item, str):
                        content_parts.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        content_parts.append(item["text"])

        if "turns" in conversation:
            for turn in conversation["turns"]:
                if isinstance(turn, dict) and "content" in turn:
                    content_parts.append(turn["content"])
                elif isinstance(turn, str):
                    content_parts.append(turn)

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content_parts.append(message["content"])
                elif isinstance(message, str):
                    content_parts.append(message)

        # Fallback: treat the whole conversation as content
        if not content_parts and isinstance(conversation, str):
            content_parts.append(conversation)

        return " ".join(content_parts).lower()

    def _analyze_crisis_indicators(self, content: str) -> Dict[CrisisType, List[CrisisIndicator]]:
        """Analyze content for crisis indicators."""
        detected_crises = {}

        for crisis_type, indicators in self.patterns.items():
            detected_indicators = []

            for indicator in indicators:
                if re.search(indicator.pattern, content, re.IGNORECASE):
                    detected_indicators.append(indicator)

            if detected_indicators:
                detected_crises[crisis_type] = detected_indicators

        return detected_crises

    def _calculate_crisis_level_improved(self, detected_crises: Dict[CrisisType, List[CrisisIndicator]]) -> Tuple[CrisisLevel, float]:
        """Calculate crisis level with improved scoring algorithm."""
        if not detected_crises:
            return CrisisLevel.ROUTINE, 0.0

        # Calculate confidence using weighted average instead of max
        total_weight = 0.0
        total_score = 0.0
        max_weight = 0.0
        has_immediate_flag = False
        immediate_weight = 0.0

        for _crisis_type, indicators in detected_crises.items():
            for indicator in indicators:
                total_weight += 1.0  # Each indicator counts as 1
                total_score += indicator.severity_weight
                max_weight = max(max_weight, indicator.severity_weight)

                if indicator.immediate_flag:
                    has_immediate_flag = True
                    immediate_weight = max(immediate_weight, indicator.severity_weight)

        # Base confidence is weighted average
        confidence = total_score / total_weight if total_weight > 0 else 0.0

        # Apply modifiers based on context

        # Multiple crisis types increase severity
        if len(detected_crises) > 1:
            confidence = min(1.0, confidence * 1.1)

        # Multiple indicators of same type increase confidence
        max_indicators_per_type = max(len(indicators) for indicators in detected_crises.values())
        if max_indicators_per_type > 2:
            confidence = min(1.0, confidence * 1.05)

        # Immediate flags force emergency level but don't over-boost confidence
        if has_immediate_flag:
            # Use the immediate flag weight, but don't artificially inflate
            confidence = max(confidence, immediate_weight)
            return CrisisLevel.EMERGENCY, confidence

        # Determine crisis level based on improved thresholds
        if confidence >= self.config["emergency_threshold"]:
            return CrisisLevel.EMERGENCY, confidence
        if confidence >= self.config["critical_threshold"]:
            return CrisisLevel.CRITICAL, confidence
        if confidence >= self.config["elevated_threshold"]:
            return CrisisLevel.ELEVATED, confidence
        return CrisisLevel.ROUTINE, confidence

def run_improved_crisis_tests():
    """Run comprehensive tests with improved scoring."""
    detector = ImprovedCrisisDetector()

    # Expanded test cases
    test_cases = [
        # Suicide Ideation Tests
        {
            "id": "suicide_ideation_test_Passive_ideation",
            "content": "Sometimes I think about not being here anymore. Everyone would be better off without me.",
            "expected_level": "elevated",
            "expected_confidence": 0.5,
            "expected_types": ["suicide_ideation"]
        },
        {
            "id": "suicide_ideation_test_Explicit_intent",
            "content": "I want to kill myself tonight. I have a plan and I'm going to do it.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["suicide_ideation"]
        },

        # Self-Harm Tests
        {
            "id": "self_harm_test_Burning_with_specific_method",
            "content": "I've been burning myself with cigarettes when the pain gets too much.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["self_harm"]
        },
        {
            "id": "self_harm_test_Cutting_escalation",
            "content": "I can't stop cutting myself. The cuts are getting deeper and I'm doing it more often.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["self_harm"]
        },

        # Psychotic Episode Tests
        {
            "id": "psychotic_episode_test_Command_hallucinations",
            "content": "The voices are telling me to hurt people. They command me to do things I don't want to do.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["psychotic_episode"]
        },
        {
            "id": "psychotic_episode_test_Auditory_hallucinations",
            "content": "I keep hearing voices in my head talking about me. They won't stop.",
            "expected_level": "critical",
            "expected_confidence": 0.6,
            "expected_types": ["psychotic_episode"]
        },

        # Panic Attack Tests
        {
            "id": "panic_attack_test_Severe_panic_symptoms",
            "content": "I can't breathe, my heart is racing, I think I'm going to die. This panic attack won't stop.",
            "expected_level": "elevated",
            "expected_confidence": 0.6,
            "expected_types": ["panic_attack"]
        },
        {
            "id": "panic_attack_test_Physical_symptoms",
            "content": "My chest feels tight and I'm shaking. I feel dizzy and nauseous.",
            "expected_level": "elevated",
            "expected_confidence": 0.4,
            "expected_types": ["panic_attack"]
        },

        # Eating Disorder Crisis Tests
        {
            "id": "eating_disorder_crisis_test_Medical_emergency",
            "content": "I haven't eaten in days and now I'm fainting. I think this is a medical emergency.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["eating_disorder_crisis"]
        },
        {
            "id": "eating_disorder_crisis_test_Purging_escalation",
            "content": "I've been throwing up multiple times a day and I can't stop. I'm completely out of control.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["eating_disorder_crisis"]
        },

        # Normal/Routine Tests
        {
            "id": "normal_test_Mild_sadness",
            "content": "I've been feeling a bit down lately but I'm managing okay.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        },
        {
            "id": "normal_test_Stress",
            "content": "Work has been really stressful but I have good support from friends.",
            "expected_level": "routine",
            "expected_confidence": 0.0,
            "expected_types": []
        }
    ]


    passed = 0
    failed = 0

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
        else:
            failed += 1


        if not test_passed:
            pass


    if passed >= len(test_cases) * 0.95:  # 95% pass rate
        return True
    return False

if __name__ == "__main__":
    success = run_improved_crisis_tests()
    sys.exit(0 if success else 1)
