#!/usr/bin/env python3
"""
Improved Crisis Detection Model
Addresses critical 84.4% false negative rate identified in validation

Target: Reduce false negative rate from 84.4% to <1%
Focus: Comprehensive crisis pattern recognition with context awareness
"""

import re
import asyncio
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CrisisDetectionResult:
    """Crisis detection result with confidence and reasoning"""
    is_crisis: bool
    confidence_score: float
    crisis_level: int  # 1-5 scale
    detected_patterns: List[str]
    risk_factors: List[str]
    context_analysis: Dict[str, any]
    reasoning: str

class ImprovedCrisisDetector:
    """
    Improved crisis detection model addressing 84.4% false negative rate
    Uses multi-layered pattern recognition with context awareness
    """
    
    def __init__(self):
        self.crisis_patterns = self._initialize_crisis_patterns()
        self.context_filters = self._initialize_context_filters()
        self.risk_scoring = self._initialize_risk_scoring()
        
    def _initialize_crisis_patterns(self) -> Dict[str, Dict]:
        """Initialize comprehensive crisis detection patterns"""
        
        return {
            # Direct suicidal ideation (Level 5 - Immediate)
            "direct_suicidal": {
                "patterns": [
                    r"(?:want to|going to|plan to|will) (?:kill myself|commit suicide|end (?:my )?life)",
                    r"(?:i'm|i am) (?:going to|planning to) (?:kill myself|die|commit suicide)",
                    r"(?:suicide|kill myself|end it all|end my life)",
                    r"(?:i (?:want to|will|am going to) die)",
                    r"(?:ready to (?:die|end it|kill myself))",
                    r"(?:have a plan (?:to|and))",
                    r"(?:tonight (?:i|i'm|i am) (?:going to|will))",
                ],
                "level": 5,
                "weight": 1.0
            },
            
            # Self-harm indicators (Level 4-5)
            "self_harm": {
                "patterns": [
                    r"(?:cutting|cut) (?:myself|my)",
                    r"(?:hurt|hurting|harm|harming) myself",
                    r"(?:burning|burn) myself",
                    r"(?:hitting|hit) myself",
                    r"(?:scratching|scratch) (?:myself|until)",
                    r"(?:razor|blade|knife|sharp)",
                    r"(?:scars|cuts|wounds) (?:are|getting)",
                    r"(?:bleeding|blood) (?:was|is)",
                ],
                "level": 4,
                "weight": 0.9
            },
            
            # Hopelessness and despair (Level 3-4)
            "hopelessness": {
                "patterns": [
                    r"(?:no (?:point|hope|way out|reason))",
                    r"(?:hopeless|worthless|pointless)",
                    r"(?:can't (?:go on|take (?:it|this)|handle))",
                    r"(?:better off (?:dead|without me))",
                    r"(?:nothing (?:matters|left|to live for))",
                    r"(?:give up|giving up)",
                    r"(?:end (?:the|this) pain)",
                    r"(?:tired of (?:living|life|everything))",
                ],
                "level": 3,
                "weight": 0.7
            },
            
            # Depression severity indicators (Level 2-4)
            "severe_depression": {
                "patterns": [
                    r"(?:can't (?:eat|sleep|function|get out of bed))",
                    r"(?:haven't (?:left|eaten|slept) (?:in|for) (?:days|weeks))",
                    r"(?:empty (?:inside|feeling))",
                    r"(?:numb|nothing|void)",
                    r"(?:drowning in (?:sadness|depression))",
                    r"(?:completely (?:broken|destroyed|lost))",
                    r"(?:barely (?:surviving|existing|breathing))",
                ],
                "level": 3,
                "weight": 0.6
            },
            
            # Crisis escalation indicators (Level 3-5)
            "escalation": {
                "patterns": [
                    r"(?:getting worse|can't stop|more (?:frequent|often))",
                    r"(?:every (?:day|night|time))",
                    r"(?:all i (?:think about|can think))",
                    r"(?:obsessed|consumed|overwhelmed)",
                    r"(?:spiraling|falling apart|breaking down)",
                    r"(?:losing (?:control|it|my mind))",
                    r"(?:scared (?:of myself|i might))",
                ],
                "level": 4,
                "weight": 0.8
            },
            
            # Means and methods (Level 5)
            "means_methods": {
                "patterns": [
                    r"(?:pills|medication|overdose)",
                    r"(?:gun|weapon|firearm)",
                    r"(?:rope|hanging|noose)",
                    r"(?:bridge|jump|jumping)",
                    r"(?:car|traffic|train)",
                    r"(?:gas|carbon monoxide)",
                    r"(?:wrote (?:a )?(?:note|letter))",
                ],
                "level": 5,
                "weight": 1.0
            },
            
            # Isolation and withdrawal (Level 2-3)
            "isolation": {
                "patterns": [
                    r"(?:no one (?:cares|understands|would miss))",
                    r"(?:all (?:alone|by myself))",
                    r"(?:isolated|disconnected|withdrawn)",
                    r"(?:pushed (?:everyone|people) away)",
                    r"(?:no (?:friends|family|support))",
                    r"(?:nobody (?:loves|cares about) me)",
                ],
                "level": 2,
                "weight": 0.4
            },
            
            # Substance abuse crisis (Level 3-4)
            "substance_crisis": {
                "patterns": [
                    r"(?:drinking (?:to|until|myself))",
                    r"(?:drugs (?:to|until|numb))",
                    r"(?:high (?:to|until|all the time))",
                    r"(?:overdose|od|too many)",
                    r"(?:blackout|passed out|woke up)",
                    r"(?:addiction|addicted|dependent)",
                ],
                "level": 3,
                "weight": 0.6
            }
        }
    
    def _initialize_context_filters(self) -> Dict[str, List[str]]:
        """Initialize context filters to prevent false positives"""
        
        return {
            "negation_patterns": [
                r"(?:not|never|don't|won't|can't|wouldn't) .{0,20}",
                r"(?:no longer|no more|stopped|quit) .{0,20}",
                r"(?:used to|in the past|before) .{0,20}",
            ],
            
            "hypothetical_patterns": [
                r"(?:if i|what if|imagine|suppose) .{0,30}",
                r"(?:sometimes i (?:think|wonder|feel like)) .{0,30}",
                r"(?:would|could|might) .{0,20}",
            ],
            
            "metaphorical_patterns": [
                r"(?:kill time|dying to (?:see|know|hear))",
                r"(?:dead (?:tired|beat|serious))",
                r"(?:suicide (?:mission|squad|doors))",
                r"(?:cut (?:class|corners|loose))",
                r"(?:hurt (?:feelings|reputation))",
            ],
            
            "media_references": [
                r"(?:movie|film|book|song|game|tv|show)",
                r"(?:character|story|plot|scene)",
                r"(?:watched|read|played|heard)",
                r"(?:netflix|youtube|spotify)",
            ],
            
            "positive_context": [
                r"(?:getting (?:better|help|treatment))",
                r"(?:therapy|therapist|counselor|doctor)",
                r"(?:medication|treatment|support)",
                r"(?:feeling (?:better|hopeful|optimistic))",
                r"(?:improving|progress|recovery)",
            ]
        }
    
    def _initialize_risk_scoring(self) -> Dict[str, float]:
        """Initialize risk scoring weights"""
        
        return {
            "direct_threat": 1.0,
            "immediate_plan": 0.9,
            "means_available": 0.8,
            "escalation": 0.7,
            "severe_symptoms": 0.6,
            "isolation": 0.4,
            "substance_use": 0.5,
            "context_mitigation": -0.3,
            "positive_factors": -0.2
        }
    
    async def detect_crisis(self, text: str) -> CrisisDetectionResult:
        """
        Main crisis detection method with improved accuracy
        
        Args:
            text: Conversation text to analyze
            
        Returns:
            CrisisDetectionResult with detailed analysis
        """
        
        # Normalize text
        normalized_text = text.lower().strip()
        
        # Initialize detection variables
        detected_patterns = []
        risk_factors = []
        confidence_score = 0.0
        crisis_level = 0
        
        # Pattern matching with scoring
        for category, pattern_data in self.crisis_patterns.items():
            patterns = pattern_data["patterns"]
            level = pattern_data["level"]
            weight = pattern_data["weight"]
            
            for pattern in patterns:
                matches = re.findall(pattern, normalized_text, re.IGNORECASE)
                if matches:
                    detected_patterns.append(f"{category}: {pattern}")
                    risk_factors.append(category)
                    confidence_score += weight * 0.2  # Base score per pattern
                    crisis_level = max(crisis_level, level)
        
        # Context analysis
        context_analysis = self._analyze_context(normalized_text)
        
        # Apply context filters
        if context_analysis["has_negation"]:
            confidence_score *= 0.7  # Reduce confidence for negated statements
        
        if context_analysis["is_hypothetical"]:
            confidence_score *= 0.5  # Reduce confidence for hypothetical statements
        
        if context_analysis["is_metaphorical"]:
            confidence_score *= 0.1  # Heavily reduce for metaphorical usage
        
        if context_analysis["has_positive_context"]:
            confidence_score *= 0.8  # Slight reduction for positive context
        
        # Boost confidence for multiple risk factors
        if len(set(risk_factors)) > 1:
            confidence_score *= 1.3  # Multiple risk factors increase confidence
        
        # Boost confidence for high-level patterns
        if crisis_level >= 4:
            confidence_score *= 1.2
        
        # Normalize confidence score
        confidence_score = min(1.0, confidence_score)
        
        # Determine if crisis based on confidence and level
        is_crisis = (confidence_score >= 0.3 and crisis_level >= 3) or confidence_score >= 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            detected_patterns, risk_factors, context_analysis, 
            confidence_score, crisis_level, is_crisis
        )
        
        return CrisisDetectionResult(
            is_crisis=is_crisis,
            confidence_score=confidence_score,
            crisis_level=crisis_level,
            detected_patterns=detected_patterns,
            risk_factors=list(set(risk_factors)),
            context_analysis=context_analysis,
            reasoning=reasoning
        )
    
    def _analyze_context(self, text: str) -> Dict[str, any]:
        """Analyze context to prevent false positives"""
        
        context = {
            "has_negation": False,
            "is_hypothetical": False,
            "is_metaphorical": False,
            "has_media_reference": False,
            "has_positive_context": False,
            "text_length": len(text),
            "sentence_count": len(text.split('.')),
        }
        
        # Check for negation patterns
        for pattern in self.context_filters["negation_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                context["has_negation"] = True
                break
        
        # Check for hypothetical patterns
        for pattern in self.context_filters["hypothetical_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                context["is_hypothetical"] = True
                break
        
        # Check for metaphorical patterns
        for pattern in self.context_filters["metaphorical_patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                context["is_metaphorical"] = True
                break
        
        # Check for media references
        for pattern in self.context_filters["media_references"]:
            if re.search(pattern, text, re.IGNORECASE):
                context["has_media_reference"] = True
                break
        
        # Check for positive context
        for pattern in self.context_filters["positive_context"]:
            if re.search(pattern, text, re.IGNORECASE):
                context["has_positive_context"] = True
                break
        
        return context
    
    def _generate_reasoning(self, detected_patterns: List[str], 
                          risk_factors: List[str], 
                          context_analysis: Dict[str, any],
                          confidence_score: float,
                          crisis_level: int,
                          is_crisis: bool) -> str:
        """Generate human-readable reasoning for the decision"""
        
        reasoning_parts = []
        
        if is_crisis:
            reasoning_parts.append(f"CRISIS DETECTED (Level {crisis_level}, Confidence: {confidence_score:.2f})")
        else:
            reasoning_parts.append(f"NO CRISIS (Level {crisis_level}, Confidence: {confidence_score:.2f})")
        
        if detected_patterns:
            reasoning_parts.append(f"Detected patterns: {len(detected_patterns)}")
            
        if risk_factors:
            reasoning_parts.append(f"Risk factors: {', '.join(set(risk_factors))}")
        
        # Context modifiers
        context_notes = []
        if context_analysis["has_negation"]:
            context_notes.append("negation detected")
        if context_analysis["is_hypothetical"]:
            context_notes.append("hypothetical language")
        if context_analysis["is_metaphorical"]:
            context_notes.append("metaphorical usage")
        if context_analysis["has_positive_context"]:
            context_notes.append("positive context present")
        
        if context_notes:
            reasoning_parts.append(f"Context modifiers: {', '.join(context_notes)}")
        
        return " | ".join(reasoning_parts)

# Enhanced model predictor using improved crisis detector
async def enhanced_model_predictor(conversation_text: str) -> bool:
    """
    Enhanced model predictor using improved crisis detection
    Target: Reduce false negative rate from 84.4% to <1%
    """
    
    detector = ImprovedCrisisDetector()
    result = await detector.detect_crisis(conversation_text)
    
    # Log detailed analysis for debugging
    if result.confidence_score > 0.1:  # Log any significant detection
        logger.info(f"Crisis Analysis: {result.reasoning}")
    
    return result.is_crisis

# Test the improved detector
async def test_improved_detector():
    """Test the improved crisis detector on sample cases"""
    
    test_cases = [
        # Should detect (true positives)
        ("I can't take this anymore. I want to kill myself.", True),
        ("I've been cutting myself every night this week.", True),
        ("I have a plan to end my life tonight.", True),
        ("Life is hopeless and I can't go on.", True),
        ("I'm going to overdose on these pills.", True),
        
        # Should NOT detect (avoid false positives)
        ("I'm fine, just having a bad day.", False),
        ("I want to kill time until my appointment.", False),
        ("I'm dying to see that new movie.", False),
        ("I have a gun in my video game.", False),
        ("I'm going to end this conversation now.", False),
        
        # Edge cases
        ("Sometimes I think about not being here anymore.", True),  # Should detect
        ("I used to think about suicide but not anymore.", False),  # Should NOT detect
        ("What if I killed myself? Just wondering.", False),  # Hypothetical - should NOT detect
        ("I'm getting help for my suicidal thoughts.", False),  # Positive context - should NOT detect
    ]
    
    detector = ImprovedCrisisDetector()
    
    correct = 0
    total = len(test_cases)
    
    print("\n" + "="*80)
    print("TESTING IMPROVED CRISIS DETECTOR")
    print("="*80)
    
    for text, expected in test_cases:
        result = await detector.detect_crisis(text)
        is_correct = result.is_crisis == expected
        correct += is_correct
        
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"\n{status}")
        print(f"Text: {text}")
        print(f"Expected: {expected}, Got: {result.is_crisis}")
        print(f"Reasoning: {result.reasoning}")
    
    accuracy = (correct / total) * 100
    print(f"\n{'='*80}")
    print(f"TEST RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"{'='*80}")
    
    return accuracy

if __name__ == "__main__":
    asyncio.run(test_improved_detector())
