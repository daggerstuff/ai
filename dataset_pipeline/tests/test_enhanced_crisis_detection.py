#!/usr/bin/env python3
"""
Test Enhanced Crisis Detection Patterns
Test the improved patterns against the comprehensive test suite.
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

class EnhancedCrisisDetector:
    """Enhanced crisis detector with improved patterns."""

    def __init__(self):
        """Initialize enhanced detector."""
        self.patterns_loader = EnhancedCrisisPatterns()
        self.patterns = self.patterns_loader.get_patterns()
        self.config = self.patterns_loader.get_config()

    def detect_crisis(self, conversation: Dict[str, Any]) -> CrisisDetection:
        """Detect crisis in conversation using enhanced patterns."""
        conversation_id = conversation.get("id", conversation.get("conversation_id", "unknown"))

        # Extract content
        content = self._extract_content(conversation)

        # Analyze for crisis indicators
        detected_crises = self._analyze_crisis_indicators(content)

        # Calculate overall crisis level and confidence
        crisis_level, confidence_score = self._calculate_crisis_level(detected_crises)

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

    def _calculate_crisis_level(self, detected_crises: Dict[CrisisType, List[CrisisIndicator]]) -> Tuple[CrisisLevel, float]:
        """Calculate overall crisis level and confidence."""
        if not detected_crises:
            return CrisisLevel.ROUTINE, 0.0

        # Calculate weighted confidence score
        total_weight = 0.0
        max_weight = 0.0
        has_immediate_flag = False

        for _crisis_type, indicators in detected_crises.items():
            for indicator in indicators:
                total_weight += indicator.severity_weight
                max_weight = max(max_weight, indicator.severity_weight)
                if indicator.immediate_flag:
                    has_immediate_flag = True

        # Base confidence is the maximum weight found
        confidence = max_weight

        # Boost confidence for multiple indicators
        if len(detected_crises) > 1:
            confidence = min(1.0, confidence * 1.1)

        # Boost confidence for immediate flags
        if has_immediate_flag:
            confidence = min(1.0, confidence * 1.05)

        # Determine crisis level based on enhanced thresholds
        if confidence >= self.config["emergency_threshold"] or has_immediate_flag:
            return CrisisLevel.EMERGENCY, confidence
        if confidence >= self.config["critical_threshold"]:
            return CrisisLevel.CRITICAL, confidence
        if confidence >= self.config["elevated_threshold"]:
            return CrisisLevel.ELEVATED, confidence
        return CrisisLevel.ROUTINE, confidence

def run_enhanced_crisis_tests():
    """Run the comprehensive crisis test suite with enhanced patterns."""
    detector = EnhancedCrisisDetector()

    # Test cases from the comprehensive test suite
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
            "id": "self_harm_test_Burning_with_specific_method",
            "content": "I've been burning myself with cigarettes when the pain gets too much.",
            "expected_level": "critical",
            "expected_confidence": 0.7,
            "expected_types": ["self_harm"]
        },
        {
            "id": "psychotic_episode_test_Command_hallucinations",
            "content": "The voices are telling me to hurt people. They command me to do things I don't want to do.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["psychotic_episode"]
        },
        {
            "id": "panic_attack_test_Severe_panic_symptoms",
            "content": "I can't breathe, my heart is racing, I think I'm going to die. This panic attack won't stop.",
            "expected_level": "elevated",
            "expected_confidence": 0.6,
            "expected_types": ["panic_attack"]
        },
        {
            "id": "eating_disorder_crisis_test_Medical_emergency_from_restriction",
            "content": "I haven't eaten in days and now I'm fainting. I think this is a medical emergency.",
            "expected_level": "emergency",
            "expected_confidence": 0.8,
            "expected_types": ["eating_disorder_crisis"]
        }
    ]


    passed = 0
    failed = 0

    for test_case in test_cases:
        result = detector.detect_crisis(test_case)

        # Check if test passes
        level_match = result.crisis_level.value[0] == test_case["expected_level"]
        confidence_ok = result.confidence_score >= test_case["expected_confidence"]
        types_match = any(expected_type in result.crisis_types for expected_type in test_case["expected_types"])

        test_passed = level_match and confidence_ok and types_match

        if test_passed:
            passed += 1
        else:
            failed += 1


        if not test_passed:
            pass


    return passed == len(test_cases)

if __name__ == "__main__":
    success = run_enhanced_crisis_tests()
    sys.exit(0 if success else 1)
