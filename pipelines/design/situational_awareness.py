"""
Situational Awareness Agent

Provides contextual analysis to help the LLM classifier make better decisions
by identifying key situational markers that distinguish between categories.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional

from ai.pipelines.design.temporal_analyzer import TemporalContextAnalyzer


@dataclass
class SituationalContext:
    """Contextual analysis of a therapeutic conversation."""
    
    # Primary indicators
    is_crisis: bool
    is_growth_focused: bool
    is_assessment: bool
    is_relationship_focused: bool
    is_trauma_active: bool
    
    # Temporal markers (enhanced with full temporal analysis)
    is_present_tense: bool  # Happening now
    is_past_tense: bool  # Already processed/historical
    is_future_oriented: bool  # Goals, planning
    is_active_issue: bool  # Current ongoing problem
    is_resolved: bool  # Past and fully processed
    
    # Severity markers
    has_urgent_language: bool
    has_processing_language: bool  # "working through", "learning to"
    has_metaphorical_language: bool
    
    # Confidence and reasoning
    confidence: float
    key_indicators: List[str]
    reasoning: str
    temporal_guidance: str  # Temporal-specific guidance


class SituationalAwarenessAgent:
    """
    Analyzes conversational context to provide situational awareness.
    
    Helps distinguish between:
    - Active crisis vs. coping with mental health
    - Self-growth vs. mental health support
    - Metaphorical vs. literal crisis language
    - Current trauma vs. past trauma processing
    - Relationship abuse vs. general relationship issues
    """
    
    def __init__(self):
        """Initialize with temporal analyzer."""
        self.temporal_analyzer = TemporalContextAnalyzer()
    
    # Crisis indicators (URGENT, IMMEDIATE)
    CRISIS_URGENT = [
        r'\b(right now|currently|at this moment)\b.*\b(suicidal|kill myself|end it|harm)\b',
        r'\b(planning to|going to|about to)\b.*\b(suicide|kill myself|harm)\b',
        r'\b(can\'t go on|can\'t take it|want to die)\b',
        r'\b(immediate danger|emergency|crisis hotline)\b',
    ]
    
    # Growth/wellness (NOT crisis or mental health disorder)
    GROWTH_LANGUAGE = [
        r'\b(become|becoming|growth|evolve|evolving|transform)\b',
        r'\b(better version|improve myself|self-improvement|personal development)\b',
        r'\b(life coaching|life goals|self-actualization)\b',
        r'\b(potential|possibility|opportunity)\b.*\b(growth|change)\b',
    ]
    
    # Metaphorical/figurative language
    METAPHORICAL = [
        r'\b(feel like|feels like|it\'s like)\b.*\b(dying|death|killing)\b',
        r'\b(old self|former self|past self)\b.*\b(dying|fading|disappearing)\b',
        r'\b(rebirth|phoenix|metamorphosis|cocoon)\b',
        r'\b(metaphor|figuratively|symbolically)\b',
    ]
    
    # Active processing (working through issues)
    PROCESSING_LANGUAGE = [
        r'\b(working through|processing|learning to cope|developing skills)\b',
        r'\b(therapy helps|therapist taught|learning from)\b',
        r'\b(making progress|getting better|improving)\b',
        r'\b(coping strategies|tools|techniques)\b',
    ]
    
    # Clinical assessment markers
    ASSESSMENT_MARKERS = [
        r'\b(diagnosis|diagnostic|screening|assessment|evaluation)\b',
        r'\b(symptoms of|criteria for|meets criteria)\b',
        r'\b(psychiatric evaluation|psychological testing)\b',
        r'\b(medication adjustment|dosage|prescription review)\b',
    ]
    
    # Relationship-specific (NOT just abuse mentions)
    RELATIONSHIP_CONTEXT = [
        r'\b(my partner|my spouse|my husband|my wife|my boyfriend|my girlfriend)\b',
        r'\b(our relationship|our marriage|couples therapy)\b',
        r'\b(communication problems|trust issues|intimacy)\b',
        r'\b(relationship patterns|attachment|boundaries in relationship)\b',
    ]
    
    # Domestic violence (specific patterns)
    DOMESTIC_VIOLENCE_MARKERS = [
        r'\b(he hits|she hits|he beats|she beats|beats me|hits me|physically abusive)\b',
        r'\b(controlling|isolating me|won\'t let me)\b',
        r'\b(scared|afraid|terrified)\b.*\b(of|to leave)\b',
        r'\b(domestic violence|domestic abuse|intimate partner violence)\b',
    ]
    
    # Temporal tense markers
    PRESENT_TENSE = [
        r'\b(am|is|are)\b.*\b(feeling|experiencing|going through)\b',
        r'\b(right now|currently|at the moment|these days)\b',
        r'\b(I feel|I\'m|I am)\b',
    ]
    
    PAST_TENSE = [
        r'\b(was|were|had|used to|previously|in the past)\b',
        r'\b(overcame|processed|worked through|dealt with)\b',
        r'\b(no longer|not anymore|moved past)\b',
    ]
    
    FUTURE_ORIENTED = [
        r'\b(want to|going to|will|hope to|plan to|goal is)\b',
        r'\b(working on|focusing on|trying to)\b',
        r'\b(future|next steps|moving forward)\b',
    ]
    
    def analyze(self, text: str) -> SituationalContext:
        """
        Analyze situational context of conversation.
        
        Args:
            text: Conversation text
            
        Returns:
            SituationalContext with detailed analysis
        """
        text_lower = text.lower()
        indicators = []
        
        # Check crisis markers
        crisis_urgent = self._match_patterns(text_lower, self.CRISIS_URGENT)
        is_crisis = len(crisis_urgent) > 0
        if crisis_urgent:
            indicators.extend([f"Urgent crisis: {m}" for m in crisis_urgent[:2]])
        
        # Check growth/wellness
        growth_matches = self._match_patterns(text_lower, self.GROWTH_LANGUAGE)
        is_growth = len(growth_matches) > 0
        if growth_matches:
            indicators.extend([f"Growth-focused: {m}" for m in growth_matches[:2]])
        
        # Check metaphorical
        metaphor_matches = self._match_patterns(text_lower, self.METAPHORICAL)
        has_metaphor = len(metaphor_matches) > 0
        if metaphor_matches:
            indicators.extend([f"Metaphorical: {m}" for m in metaphor_matches[:2]])
        
        # Check processing
        processing_matches = self._match_patterns(text_lower, self.PROCESSING_LANGUAGE)
        has_processing = len(processing_matches) > 0
        if processing_matches:
            indicators.extend([f"Processing: {m}" for m in processing_matches[:2]])
        
        # Check assessment
        assessment_matches = self._match_patterns(text_lower, self.ASSESSMENT_MARKERS)
        is_assessment = len(assessment_matches) > 0
        if assessment_matches:
            indicators.extend([f"Assessment: {m}" for m in assessment_matches[:2]])
        
        # Check relationship context
        relationship_matches = self._match_patterns(text_lower, self.RELATIONSHIP_CONTEXT)
        dv_matches = self._match_patterns(text_lower, self.DOMESTIC_VIOLENCE_MARKERS)
        is_relationship = len(relationship_matches) > 0 or len(dv_matches) > 0
        if relationship_matches:
            indicators.extend([f"Relationship: {m}" for m in relationship_matches[:2]])
        if dv_matches:
            indicators.extend([f"DV markers: {m}" for m in dv_matches[:2]])
        
        # Get full temporal analysis using TemporalContextAnalyzer
        temporal = self.temporal_analyzer.analyze(text)
        
        # Use temporal analyzer results
        present = temporal.temporal_focus == 'current'
        past = temporal.temporal_focus == 'past'
        future = temporal.temporal_focus == 'future'
        is_active_issue = temporal.is_active_issue
        is_resolved = temporal.is_resolved
        
        # Trauma processing check with temporal awareness
        trauma_keywords = ['trauma', 'ptsd', 'abuse', 'assault']
        has_trauma_words = any(word in text_lower for word in trauma_keywords)
        is_trauma_active = has_trauma_words and is_active_issue and not is_resolved
        
        # Build reasoning
        reasoning_parts = []
        if is_crisis:
            reasoning_parts.append("URGENT CRISIS LANGUAGE detected")
        if is_growth and has_metaphor:
            reasoning_parts.append("Growth-focused with metaphorical language (NOT literal crisis)")
        if has_processing and not is_crisis:
            reasoning_parts.append("Active processing/coping (NOT acute crisis)")
        if is_assessment:
            reasoning_parts.append("Clinical assessment/evaluation context")
        if dv_matches and relationship_matches:
            reasoning_parts.append("Domestic violence in relationship context (NOT general crisis)")
        if is_resolved and has_trauma_words:
            reasoning_parts.append("Past trauma (FULLY PROCESSED/RESOLVED, NOT active treatment)")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard therapeutic conversation"
        
        # Build temporal guidance
        temporal_guidance = ""
        if is_resolved:
            temporal_guidance = "⏱️ RESOLVED/PAST → therapeutic_conversation (NOT active treatment category)"
        elif is_active_issue and has_trauma_words:
            temporal_guidance = "⏱️ ACTIVE TRAUMA → trauma_processing (NOT therapeutic_conversation)"
        elif is_assessment:
            temporal_guidance = "🏥 CLINICAL MARKERS → clinical_assessment (NOT mental_health_support)"
        elif dv_matches and relationship_matches:
            temporal_guidance = "💑 ABUSE IN RELATIONSHIP → relationship_therapy (prioritize over crisis_support)"
        
        # Calculate confidence
        total_indicators = len(indicators)
        confidence = min(1.0, total_indicators * 0.15 + 0.3)
        
        return SituationalContext(
            is_crisis=is_crisis,
            is_growth_focused=is_growth,
            is_assessment=is_assessment,
            is_relationship_focused=is_relationship,
            is_trauma_active=is_trauma_active,
            is_present_tense=present,
            is_past_tense=past,
            is_future_oriented=future,
            is_active_issue=is_active_issue,
            is_resolved=is_resolved,
            has_urgent_language=is_crisis,
            has_processing_language=has_processing,
            has_metaphorical_language=has_metaphor,
            confidence=confidence,
            key_indicators=indicators[:6],  # Top 6
            reasoning=reasoning,
            temporal_guidance=temporal_guidance,
        )
    
    def _match_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Find all matching patterns in text."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Extract the matched portion for reporting
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    matches.append(match.group(0)[:50])
        return matches


# Test cases
if __name__ == "__main__":
    agent = SituationalAwarenessAgent()
    
    test_cases = [
        ("I feel like my old self is dying as I become someone new.", "Growth/metaphorical"),
        ("I am planning to kill myself tonight.", "Urgent crisis"),
        ("I worked through my trauma in therapy last year.", "Past processing"),
        ("My husband beats me and I'm scared.", "Domestic violence + relationship"),
        ("Need evaluation for medication adjustment.", "Clinical assessment"),
        ("Learning to cope with depression through CBT.", "Processing/mental health support"),
    ]
    
    print("Situational Awareness Agent Test\n" + "="*60)
    for text, label in test_cases:
        context = agent.analyze(text)
        print(f"\nText: {text}")
        print(f"Label: {label}")
        print(f"Analysis: {context.reasoning}")
        print(f"Indicators: {context.key_indicators}")
        print(f"Crisis: {context.is_crisis}, Growth: {context.is_growth_focused}, "
              f"Assessment: {context.is_assessment}")
