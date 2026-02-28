"""
Temporal Context Analyzer

Distinguishes between:
- Current/active problems (present tense, ongoing)
- Past/resolved issues (past tense, processed, overcome)
- Future concerns (planning, prevention)

This helps prevent misclassification of "I used to have trauma" as trauma_processing
when it should be therapeutic_conversation about past experiences.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class TemporalContext:
    """Temporal context analysis result."""
    
    temporal_focus: str  # 'current', 'past', 'future', 'mixed'
    is_active_issue: bool  # Is this an ongoing problem?
    is_resolved: bool  # Has this been processed/overcome?
    confidence: float
    indicators: List[str]


class TemporalContextAnalyzer:
    """Analyzes temporal context of conversations."""
    
    # Present tense / current problem indicators
    PRESENT_PATTERNS = [
        r'\b(am|is|are|feel|feeling|experiencing)\b',
        r'\b(currently|right now|these days|lately)\b',
        r'\b(still|continue to|keep)\b',
        r'\b(struggle|struggling|dealing with)\b',
        r'\b(have been|has been)\b.*\b(for|since)\b',  # "have been depressed for months"
    ]
    
    # Past tense / resolved indicators
    PAST_PATTERNS = [
        r'\b(was|were|had|used to|would)\b',
        r'\b(overcame|overcome|processed|healed|recovered)\b',
        r'\b(no longer|not anymore|don\'t anymore)\b',
        r'\b(in the past|previously|before|ago)\b',
        r'\b(fully processed|worked through|dealt with)\b',
    ]
    
    # Future / planning indicators
    FUTURE_PATTERNS = [
        r'\b(will|going to|plan to|want to)\b',
        r'\b(future|upcoming|next|soon)\b',
        r'\b(hope to|trying to|working on)\b',
    ]
    
    # Active suffering indicators (overrides past tense)
    ACTIVE_SUFFERING_PATTERNS = [
        r'\b(still (feel|experience|suffer|struggle))\b',
        r'\b(can\'t (stop|get over|forget))\b',
        r'\b(haunts|haunted|triggers|flashback)\b',
        r'\b(every day|constantly|always)\b',
    ]
    
    # Resolution indicators (overrides present tense)
    RESOLUTION_PATTERNS = [
        r'\b(better now|doing well|much better)\b',
        r'\b(learned to|found peace|made peace)\b',
        r'\b(doesn\'t (bother|affect|hurt) me)\b',
        r'\b(moved on|moved past|let go)\b',
    ]
    
    def analyze(self, text: str) -> TemporalContext:
        """
        Analyze the temporal context of a conversation.
        
        Args:
            text: Conversation text to analyze
            
        Returns:
            TemporalContext with temporal focus and indicators
        """
        text_lower = text.lower()
        
        # Count pattern matches
        present_count = self._count_patterns(text_lower, self.PRESENT_PATTERNS)
        past_count = self._count_patterns(text_lower, self.PAST_PATTERNS)
        future_count = self._count_patterns(text_lower, self.FUTURE_PATTERNS)
        active_suffering = self._count_patterns(
            text_lower, self.ACTIVE_SUFFERING_PATTERNS
        )
        resolution = self._count_patterns(text_lower, self.RESOLUTION_PATTERNS)
        
        # Determine temporal focus
        indicators = []
        
        # Active suffering overrides past tense
        if active_suffering > 0:
            temporal_focus = 'current'
            is_active_issue = True
            is_resolved = False
            indicators.append(f'Active suffering indicators ({active_suffering})')
            confidence = min(0.9, 0.6 + (active_suffering * 0.15))
            
        # Resolution overrides present tense
        elif resolution > 0:
            temporal_focus = 'past'
            is_active_issue = False
            is_resolved = True
            indicators.append(f'Resolution indicators ({resolution})')
            confidence = min(0.9, 0.6 + (resolution * 0.15))
            
        # Otherwise, go by majority
        else:
            total = present_count + past_count + future_count
            
            if total == 0:
                # No clear temporal markers
                return TemporalContext(
                    temporal_focus='current',  # Default to current
                    is_active_issue=True,
                    is_resolved=False,
                    confidence=0.3,
                    indicators=['No clear temporal markers - assuming current'],
                )
            
            # Determine dominant temporal focus
            if present_count > past_count and present_count > future_count:
                temporal_focus = 'current'
                is_active_issue = True
                is_resolved = False
                indicators.append(f'Present tense dominant ({present_count})')
                confidence = present_count / total
                
            elif past_count > present_count and past_count > future_count:
                temporal_focus = 'past'
                is_active_issue = False
                is_resolved = True
                indicators.append(f'Past tense dominant ({past_count})')
                confidence = past_count / total
                
            elif future_count > present_count and future_count > past_count:
                temporal_focus = 'future'
                is_active_issue = False  # Planning, not current problem
                is_resolved = False
                indicators.append(f'Future tense dominant ({future_count})')
                confidence = future_count / total
                
            else:
                # Mixed temporal focus
                temporal_focus = 'mixed'
                is_active_issue = present_count >= past_count
                is_resolved = past_count > present_count and resolution > 0
                indicators.append(
                    f'Mixed temporal focus (P:{present_count}, '
                    f'Pa:{past_count}, F:{future_count})'
                )
                confidence = 0.5
        
        return TemporalContext(
            temporal_focus=temporal_focus,
            is_active_issue=is_active_issue,
            is_resolved=is_resolved,
            confidence=confidence,
            indicators=indicators,
        )
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        return count


def main():
    """Test the temporal analyzer."""
    analyzer = TemporalContextAnalyzer()
    
    test_cases = [
        # Current/active issues
        "I am feeling very depressed and anxious right now.",
        "I struggle with PTSD every day.",
        "I have been dealing with trauma for months.",
        
        # Past/resolved issues
        "I used to have depression but overcame it.",
        "I had trauma in the past but fully processed it.",
        "I was anxious before, but I'm better now.",
        
        # Mixed - active despite past tense
        "I was abused and still can't get over it.",
        "The trauma happened years ago but haunts me daily.",
        
        # Mixed - resolved despite present tense
        "I have trauma but it doesn't bother me anymore.",
        "I feel much better now and moved past it.",
    ]
    
    print("Temporal Context Analysis Test:\n")
    for text in test_cases:
        result = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"  Focus: {result.temporal_focus}")
        print(f"  Active Issue: {result.is_active_issue}")
        print(f"  Resolved: {result.is_resolved}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Indicators: {result.indicators}")
        print()


if __name__ == "__main__":
    main()
