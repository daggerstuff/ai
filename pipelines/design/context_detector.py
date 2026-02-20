"""
Context Awareness Detector

Detects educational, theoretical, or meta-discussion contexts to prevent
misclassification of conversations ABOUT therapy vs actual therapy.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ContextSignals:
    """Signals indicating conversation context."""
    
    is_educational: bool = False
    is_theoretical: bool = False
    is_meta_discussion: bool = False
    is_therapeutic: bool = True
    confidence: float = 0.0
    indicators: List[str] = None
    
    def __post_init__(self):
        if self.indicators is None:
            self.indicators = []


class ContextDetector:
    """
    Detects conversation context to distinguish:
    - Educational discussions ABOUT therapy
    - Theoretical/academic discussions
    - Training/supervision scenarios
    - Actual therapeutic conversations
    """
    
    # Educational/academic indicators
    EDUCATIONAL_PATTERNS = [
        r'\b(textbook|academic|research|study|literature|article)\b',
        r'\b(learning|teaching|education|training|course)\b',
        r'\b(example of|for instance|such as)\b',
        r'\b(theory|theoretical|hypothesis|model)\b',
        r'\b(defined as|refers to|means that)\b',
        r'\b(according to|based on research)\b',
        r'\b(studying|student|learning about)\b',
        r'\b(explain|describe|what is|what are)\b',
        r'\bcan you (explain|tell me about|describe)\b',
    ]
    
    # Meta-discussion indicators
    META_PATTERNS = [
        r'\b(discussing|talking about|conversation about)\b',
        r'\b(supervision|case review|consultation)\b',
        r'\b(this approach|this technique|this method)\b',
        r'\b(therapist would|counselor might|psychologist could)\b',
        r'\b(in therapy|in counseling|in treatment)\b',
    ]
    
    # Third-person / hypothetical indicators
    HYPOTHETICAL_PATTERNS = [
        r'\b(someone|people|clients|patients) (who|that)\b',
        r'\b(if (a|the) (patient|client))\b',
        r'\b(they might|they could|they would)\b',
        r'\b(one might|one could|one would)\b',
    ]
    
    # Strong therapeutic indicators (override educational)
    THERAPEUTIC_PATTERNS = [
        r'\b(I feel|I\'m feeling|I have been feeling)\b',
        r'\b(my (depression|anxiety|trauma|relationship))\b',
        r'\b(tell me (more|about)|let\'s explore)\b',
        r'\b(how does that make you feel)\b',
        r'\b(your (feelings|thoughts|experience))\b',
    ]
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize context detector.
        
        Args:
            threshold: Confidence threshold for context classification
        """
        self.threshold = threshold
    
    def detect_context(self, conversation_text: str) -> ContextSignals:
        """
        Detect the context of a conversation.
        
        Args:
            conversation_text: The conversation to analyze
            
        Returns:
            ContextSignals with detected context
        """
        text_lower = conversation_text.lower()
        
        # Count pattern matches
        educational_count = self._count_patterns(
            text_lower, self.EDUCATIONAL_PATTERNS
        )
        meta_count = self._count_patterns(text_lower, self.META_PATTERNS)
        hypothetical_count = self._count_patterns(
            text_lower, self.HYPOTHETICAL_PATTERNS
        )
        therapeutic_count = self._count_patterns(text_lower, self.THERAPEUTIC_PATTERNS)
        
        # Calculate total non-therapeutic signals
        non_therapeutic = educational_count + meta_count + hypothetical_count
        total_signals = non_therapeutic + therapeutic_count
        
        if total_signals == 0:
            # No clear signals, assume therapeutic
            return ContextSignals(
                is_therapeutic=True,
                confidence=0.5,
                indicators=["No clear context indicators"]
            )
        
        # Calculate confidence scores
        if total_signals > 0:
            non_therapeutic_ratio = non_therapeutic / total_signals
            therapeutic_ratio = therapeutic_count / total_signals
        else:
            non_therapeutic_ratio = 0
            therapeutic_ratio = 0
        
        # Collect indicators
        indicators = []
        if educational_count > 0:
            indicators.append(f"Educational language ({educational_count} matches)")
        if meta_count > 0:
            indicators.append(f"Meta-discussion ({meta_count} matches)")
        if hypothetical_count > 0:
            indicators.append(f"Hypothetical language ({hypothetical_count} matches)")
        if therapeutic_count > 0:
            indicators.append(f"Therapeutic language ({therapeutic_count} matches)")
        
        # Determine context
        if therapeutic_count > non_therapeutic:
            # Strong therapeutic signals override
            return ContextSignals(
                is_therapeutic=True,
                confidence=therapeutic_ratio,
                indicators=indicators
            )
        elif non_therapeutic_ratio >= self.threshold:
            # Non-therapeutic context detected
            return ContextSignals(
                is_educational=(educational_count > 0),
                is_theoretical=(hypothetical_count > 0),
                is_meta_discussion=(meta_count > 0),
                is_therapeutic=False,
                confidence=non_therapeutic_ratio,
                indicators=indicators
            )
        else:
            # Mixed signals, lean therapeutic
            return ContextSignals(
                is_therapeutic=True,
                confidence=0.5,
                indicators=indicators + ["Mixed signals - defaulting to therapeutic"]
            )
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        count = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count


# Test the detector
if __name__ == "__main__":
    detector = ContextDetector()
    
    test_cases = [
        # Educational/theoretical
        (
            "Discussing trauma processing techniques in therapy training. "
            "For example, EMDR is an approach therapists might use with "
            "clients who experienced assault."
        ),
        
        # Actual therapy
        ("I experienced sexual abuse as a child and it's affecting my relationships. "
         "Can you help me process this trauma?"),
        
        # Meta-discussion
        (
            "This technique is useful when talking about relationship "
            "issues in counseling sessions."
        ),
        
        # Real therapeutic conversation
        (
            "I feel depressed and anxious. My therapist suggested "
            "cognitive behavioral therapy."
        ),
    ]
    
    print("Testing Context Detector:\n")
    for i, text in enumerate(test_cases, 1):
        result = detector.detect_context(text)
        print(f"Test {i}:")
        print(f"  Text: {text[:80]}...")
        print(f"  Therapeutic: {result.is_therapeutic}")
        print(f"  Educational: {result.is_educational}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Indicators: {', '.join(result.indicators)}")
        print()
