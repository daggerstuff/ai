"""
Reasoning Output Parser - Format Conversion Agent

This module parses GLM4.7's natural language reasoning output into structured
classification results. GLM4.7 is a reasoning model that provides detailed
analysis but doesn't follow JSON format instructions - this agent extracts
the classification decision from that reasoning.

Strategy:
1. Parse the reasoning text to identify category mentions
2. Extract confidence indicators from the analysis
3. Identify key indicators/keywords mentioned
4. Construct structured CategoryClassification result
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ai.core.pipelines.design.taxonomy_classifier import (
    CategoryClassification,
    TherapeuticCategory,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningParseResult:
    """Result of parsing reasoning text."""

    category: Optional[TherapeuticCategory]
    confidence: float
    reasoning_text: str
    key_indicators: List[str]
    parse_method: str  # How we extracted the category


class ReasoningOutputParser:
    """
    Dedicated agent for converting GLM4.7 reasoning output to structured format.

    GLM4.7 provides detailed analysis like:
    "The user mentions sexual abuse (trauma indicator), depression (mental health),
    and anxiety. However, the primary focus should be trauma_processing because..."

    This parser extracts: category=trauma_processing, confidence=0.85, indicators=[...]
    """

    def __init__(self):
        """Initialize the reasoning parser."""
        # Category detection patterns in reasoning text
        self.category_patterns = {
            TherapeuticCategory.CRISIS_SUPPORT: [
                r"crisis[_\s]support",
                r"active crisis",
                r"immediate danger",
                r"suicide risk",
                r"self-harm risk",
                r"crisis intervention",
            ],
            TherapeuticCategory.TRAUMA_PROCESSING: [
                r"trauma[_\s]processing",
                r"trauma-focused",
                r"PTSD",
                r"processing trauma",
                r"abuse history",
                r"traumatic event",
            ],
            TherapeuticCategory.RELATIONSHIP_THERAPY: [
                r"relationship[_\s]therapy",
                r"couples therapy",
                r"family therapy",
                r"relationship issues",
                r"interpersonal",
            ],
            TherapeuticCategory.CLINICAL_ASSESSMENT: [
                r"clinical[_\s]assessment",
                r"diagnostic",
                r"evaluation",
                r"screening",
                r"assessment",
            ],
            TherapeuticCategory.MENTAL_HEALTH_SUPPORT: [
                r"mental[_\s]health[_\s]support",
                r"mental health guidance",
                r"coping strategies",
                r"anxiety management",
                r"depression management",
            ],
            TherapeuticCategory.THERAPEUTIC_CONVERSATION: [
                r"therapeutic[_\s]conversation",
                r"general therapy",
                r"standard therapy",
                r"general session",
            ],
        }

        # Confidence indicator phrases
        self.high_confidence_phrases = [
            "clearly",
            "obviously",
            "definitely",
            "strong indicator",
            "primary focus",
            "main category",
            "unambiguous",
        ]
        self.medium_confidence_phrases = [
            "likely",
            "probably",
            "suggests",
            "indicates",
            "appears to be",
            "seems to be",
        ]
        self.low_confidence_phrases = [
            "unclear",
            "ambiguous",
            "could be",
            "might be",
            "difficult to classify",
            "multiple categories",
        ]

    def parse_reasoning_output(self, reasoning_text: str) -> CategoryClassification:
        """
        Parse GLM4.7's reasoning output into structured classification.

        Args:
            reasoning_text: The full reasoning output from GLM4.7

        Returns:
            CategoryClassification with extracted category, confidence, etc.
        """
        logger.debug(f"Parsing reasoning text (length: {len(reasoning_text)})")

        # Step 1: Try to find explicit category mentions
        category, parse_method = self._extract_category(reasoning_text)

        # Step 2: Estimate confidence from language
        confidence = self._estimate_confidence(reasoning_text)

        # Step 3: Extract key indicators mentioned in reasoning
        indicators = self._extract_indicators(reasoning_text)

        # Step 4: Extract the summary/conclusion if present
        summary = self._extract_summary(reasoning_text)

        return CategoryClassification(
            category=category,
            confidence=confidence,
            reasoning=f"LLM (parsed from reasoning): {summary}",
            keywords_detected=indicators[:5],
        )

    def _extract_category(self, text: str) -> Tuple[TherapeuticCategory, str]:
        """
        Extract the category from reasoning text.

        Returns: (category, parse_method)
        """
        text_lower = text.lower()

        # Method 1: Look for explicit category name mentions
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            if score > 0:
                category_scores[category] = score

        if category_scores:
            # Get category with most mentions
            best_category = max(category_scores.items(), key=lambda x: x[1])
            logger.debug(f"Found category mentions: {category_scores}")
            return best_category[0], "pattern_matching"

        # Method 2: Look for conclusion statements
        conclusion_patterns = [
            r"(?:therefore|thus|conclude|classification|category).*?:\s*([a-z_]+)",
            r"best classified as\s+([a-z_]+)",
            r"this is\s+([a-z_]+)",
            r"categorize.*?as\s+([a-z_]+)",
        ]

        for pattern in conclusion_patterns:
            match = re.search(pattern, text_lower)
            if match:
                category_str = match.group(1).replace(" ", "_")
                try:
                    category = TherapeuticCategory(category_str)
                    logger.debug(f"Found category in conclusion: {category_str}")
                    return category, "conclusion_extraction"
                except ValueError:
                    continue

        # Method 3: Fallback - check for keywords in the original conversation
        # (this should be rare if GLM4.7 is working properly)
        logger.warning("Could not extract category from reasoning, using fallback")
        return TherapeuticCategory.THERAPEUTIC_CONVERSATION, "fallback"

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence from language used in reasoning.

        Returns: confidence score 0.0-1.0
        """
        text_lower = text.lower()

        # Count confidence indicators
        high_count = sum(
            1 for phrase in self.high_confidence_phrases if phrase in text_lower
        )
        medium_count = sum(
            1 for phrase in self.medium_confidence_phrases if phrase in text_lower
        )
        low_count = sum(
            1 for phrase in self.low_confidence_phrases if phrase in text_lower
        )

        # Calculate base confidence
        if high_count > 0:
            base = 0.85
        elif low_count > 0:
            base = 0.60
        else:
            base = 0.75  # Medium/unclear

        # Adjust based on reasoning length (longer = more thorough = higher confidence)
        if len(text) > 500:
            base += 0.05
        elif len(text) < 200:
            base -= 0.05

        # Cap at reasonable bounds
        return max(0.50, min(0.95, base))

    def _extract_indicators(self, text: str) -> List[str]:
        """
        Extract key indicators/keywords mentioned in the reasoning.

        Returns: List of indicator phrases
        """
        indicators = []

        # Look for indicator patterns
        indicator_patterns = [
            r"indicator[s]?:\s*([^.]+)",
            r"keyword[s]?:\s*([^.]+)",
            r"mentions?\s+([^.]+?)\s+\(",
            r"evidence of\s+([^.]+)",
            r"includes?\s+([^.]+?)\s+(?:which|that)",
        ]

        for pattern in indicator_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean = match.strip(" \"'(),")
                if clean and len(clean) < 50:  # Reasonable length
                    indicators.append(clean)

        # Also extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', text)
        indicators.extend(quoted[:3])  # Add up to 3 quoted phrases

        return indicators[:10]  # Return top 10

    def _extract_summary(self, text: str) -> str:
        """
        Extract a summary/conclusion from the reasoning.

        Returns: Summary string
        """
        # Look for conclusion sentences
        conclusion_markers = [
            r"(?:therefore|thus|in conclusion|conclude)[:,]?\s*([^.]+\.[^.]*\.?)",
            r"(?:this is|classified as|category:)\s*([^.]+\.)",
            r"(?:best|most appropriate|primary)\s+(?:category|classification)(?:\s+is)?:?\s*([^.]+\.)",
        ]

        for pattern in conclusion_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: Use last sentence
        sentences = text.split(".")
        if sentences:
            last = sentences[-2] if len(sentences) > 1 else sentences[-1]
            return last.strip()[:200]  # Max 200 chars

        return "Category extracted from reasoning analysis"


def main():
    """Test the reasoning parser with sample GLM4.7 outputs."""
    parser = ReasoningOutputParser()

    # Test case 1: Typical GLM4.7 reasoning output
    test_reasoning_1 = """
    The user presents a comprehensive history including sexual abuse (trauma indicator),
    breast cancer survivor (medical trauma), chronic insomnia, depression, and anxiety.
    They ask if they have "too many issues" for counseling, indicating this may be an
    intake/assessment context.
    
    However, the primary focus should be trauma_processing because:
    1. Sexual abuse is explicitly mentioned and represents significant trauma
    2. This trauma history likely underlies other symptoms (depression, anxiety)
    3. The therapeutic priority would be addressing the trauma
    
    While multiple categories apply, trauma_processing is the most appropriate
    classification given the trauma history and its central role.
    """

    result1 = parser.parse_reasoning_output(test_reasoning_1)
    print("\n" + "=" * 80)
    print("Test 1: Complex trauma case")
    print("=" * 80)
    print(f"Category: {result1.category.value}")
    print(f"Confidence: {result1.confidence:.2%}")
    print(f"Indicators: {result1.keywords_detected[:3]}")
    print(f"Reasoning: {result1.reasoning[:100]}...")

    # Test case 2: Crisis case
    test_reasoning_2 = """
    Clear crisis_support case. The user explicitly states suicidal ideation
    ("I want to kill myself"). This requires immediate crisis intervention
    and safety planning. The assistant appropriately responds with safety
    focused dialogue. This is unambiguous and high priority.
    """

    result2 = parser.parse_reasoning_output(test_reasoning_2)
    print("\n" + "=" * 80)
    print("Test 2: Clear crisis case")
    print("=" * 80)
    print(f"Category: {result2.category.value}")
    print(f"Confidence: {result2.confidence:.2%}")
    print(f"Indicators: {result2.keywords_detected[:3]}")


if __name__ == "__main__":
    main()
