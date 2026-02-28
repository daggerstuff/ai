#!/usr/bin/env python3
"""
AI-Powered Content Router
Intelligently analyzes segments and determines optimal Q/A generation strategy
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path

class AIContentRouter:
    def __init__(self):
        self.content_patterns = {
            "interview_qa": {
                "indicators": ["interviewer", "host", "question", "asked", "you mentioned"],
                "structure_markers": ["Q:", "A:", "HOST:", "GUEST:", "INTERVIEWER:"],
                "conversational_flow": ["follow up", "building on that", "you said earlier"]
            },
            "monologue_teaching": {
                "indicators": ["let me explain", "what I want you to understand", "here's what happens"],
                "teaching_markers": ["first", "second", "the key point", "remember this"],
                "direct_address": ["you need to", "I want you to", "listen carefully"]
            },
            "story_narrative": {
                "indicators": ["I remember", "there was a time", "I had a client", "for example"],
                "narrative_markers": ["once", "then", "after that", "eventually"],
                "personal_experience": ["in my practice", "I've seen", "what I learned"]
            },
            "explanation_educational": {
                "indicators": ["the reason", "because", "this is why", "what happens is"],
                "concept_markers": ["definition", "means", "refers to", "is when"],
                "process_description": ["step by step", "the process", "how it works"]
            },
            "advice_practical": {
                "indicators": ["you should", "try this", "I recommend", "what works"],
                "action_markers": ["do this", "start by", "the first step", "practice"],
                "guidance_language": ["helpful", "effective", "strategy", "approach"]
            }
        }
    
    def analyze_content_structure(self, text: str, metadata: Dict) -> Dict:
        """AI-like analysis of content structure and context"""
        analysis = {
            "primary_format": None,
            "confidence": 0.0,
            "content_characteristics": [],
            "optimal_qa_strategy": None,
            "reasoning": ""
        }
        
        text_lower = text.lower()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        
        # Analyze conversational structure
        has_dialogue_markers = any(marker in text for marker in [":", "Q:", "A:", "HOST:", "GUEST:"])
        has_question_response = "?" in text and len(text.split("?")) > 1
        
        # Analyze teaching/explanation patterns
        has_teaching_flow = any(word in text_lower for word in ["first", "second", "next", "then", "finally"])
        has_explanation_language = any(phrase in text_lower for phrase in ["because", "the reason", "what happens", "this is why"])
        
        # Analyze narrative/story elements
        has_narrative_elements = any(phrase in text_lower for phrase in ["i remember", "there was", "i had a client", "for example"])
        has_personal_experience = any(phrase in text_lower for phrase in ["in my experience", "i've seen", "what i learned"])
        
        # Analyze direct advice/guidance
        has_direct_advice = any(phrase in text_lower for phrase in ["you should", "try this", "i recommend", "what works"])
        has_action_language = any(phrase in text_lower for phrase in ["do this", "start by", "practice", "the key is"])
        
        # Score each format type
        format_scores = {}
        
        # Interview/Q&A format scoring
        interview_score = 0
        if has_dialogue_markers: interview_score += 0.4
        if has_question_response: interview_score += 0.3
        if any(word in text_lower for word in ["asked", "question", "interviewer", "host"]): interview_score += 0.3
        format_scores["interview_qa"] = interview_score
        
        # Teaching/Educational format scoring
        teaching_score = 0
        if has_teaching_flow: teaching_score += 0.3
        if has_explanation_language: teaching_score += 0.4
        if any(phrase in text_lower for phrase in ["let me explain", "what i want you to understand"]): teaching_score += 0.3
        format_scores["monologue_teaching"] = teaching_score
        
        # Story/Narrative format scoring
        story_score = 0
        if has_narrative_elements: story_score += 0.4
        if has_personal_experience: story_score += 0.3
        if any(word in text_lower for word in ["story", "example", "case", "client"]): story_score += 0.3
        format_scores["story_narrative"] = story_score
        
        # Explanation/Educational format scoring
        explanation_score = 0
        if has_explanation_language: explanation_score += 0.4
        if any(phrase in text_lower for phrase in ["definition", "means", "refers to", "is when"]): explanation_score += 0.3
        if len([s for s in sentences if "because" in s.lower()]) >= 2: explanation_score += 0.3
        format_scores["explanation_educational"] = explanation_score
        
        # Advice/Practical format scoring
        advice_score = 0
        if has_direct_advice: advice_score += 0.4
        if has_action_language: advice_score += 0.3
        if any(phrase in text_lower for phrase in ["strategy", "approach", "technique", "method"]): advice_score += 0.3
        format_scores["advice_practical"] = advice_score
        
        # Determine primary format
        if format_scores:
            primary_format = max(format_scores, key=format_scores.get)
            confidence = format_scores[primary_format]
            
            analysis["primary_format"] = primary_format
            analysis["confidence"] = confidence
            analysis["optimal_qa_strategy"] = self.determine_qa_strategy(primary_format, text, metadata)
            analysis["reasoning"] = self.generate_reasoning(primary_format, confidence, text)
        
        return analysis
    
    def determine_qa_strategy(self, format_type: str, text: str, metadata: Dict) -> str:
        """Determine optimal Q/A generation strategy based on format analysis"""
        
        if format_type == "interview_qa":
            if "?" in text and len(text.split("?")) > 1:
                return "extract_embedded_qa"
            else:
                return "reconstruct_interview_question"
                
        elif format_type == "monologue_teaching":
            return "create_learning_question"
            
        elif format_type == "story_narrative":
            return "create_context_question"
            
        elif format_type == "explanation_educational":
            return "create_understanding_question"
            
        elif format_type == "advice_practical":
            return "create_guidance_question"
            
        else:
            return "create_general_therapeutic_question"
    
    def generate_reasoning(self, format_type: str, confidence: float, text: str) -> str:
        """Generate human-readable reasoning for the classification"""
        
        reasoning_map = {
            "interview_qa": f"Detected conversational Q&A structure (confidence: {confidence:.2f})",
            "monologue_teaching": f"Identified teaching/explanatory monologue (confidence: {confidence:.2f})",
            "story_narrative": f"Recognized narrative/story format (confidence: {confidence:.2f})",
            "explanation_educational": f"Found educational explanation pattern (confidence: {confidence:.2f})",
            "advice_practical": f"Detected practical advice/guidance (confidence: {confidence:.2f})"
        }
        
        return reasoning_map.get(format_type, f"General therapeutic content (confidence: {confidence:.2f})")
    
    def generate_contextual_question(self, text: str, strategy: str, style: str) -> str:
        """Generate appropriate question based on strategy and content"""
        
        if strategy == "extract_embedded_qa":
            # Extract actual question from interview format
            question_match = re.search(r'([^.!?]*\?)', text)
            if question_match:
                return question_match.group(1).strip()
        
        elif strategy == "reconstruct_interview_question":
            # Create question that would lead to this response in interview
            if "trauma" in text.lower():
                return "Can you tell us about trauma and its impact on mental health?"
            elif "narcissist" in text.lower():
                return "What should people understand about narcissistic behavior?"
            elif "therapy" in text.lower():
                return "What's your perspective on finding effective therapy?"
        
        elif strategy == "create_learning_question":
            # Create educational question for teaching content
            return f"Can you teach me about this concept from a {style} perspective?"
        
        elif strategy == "create_context_question":
            # Create question that would prompt this story/example
            return "Can you share an example that illustrates this?"
        
        elif strategy == "create_understanding_question":
            # Create question seeking explanation
            return "Help me understand how this works."
        
        elif strategy == "create_guidance_question":
            # Create question seeking practical advice
            return "What would you recommend in this situation?"
        
        # Default fallback
        return f"Can you share your insights on this from a {style} approach?"
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment with AI-powered routing"""
        
        # Analyze content structure
        analysis = self.analyze_content_structure(segment['text'], segment)
        
        # Generate appropriate question
        question = self.generate_contextual_question(
            segment['text'], 
            analysis['optimal_qa_strategy'], 
            segment['style']
        )
        
        return {
            "input": question,
            "output": segment['text'],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "ai_analysis": {
                "format_detected": analysis['primary_format'],
                "confidence": analysis['confidence'],
                "strategy_used": analysis['optimal_qa_strategy'],
                "reasoning": analysis['reasoning']
            }
        }

def test_ai_router():
    """Test the AI content router with sample segments"""
    router = AIContentRouter()
    
    # Test segments of different types
    test_segments = [
        {
            "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles. How can somebody begin to take that path and make sure that they're finding somebody that is trained in trauma? That's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about.",
            "style": "therapeutic",
            "confidence": 1.0,
            "quality": 0.8,
            "source": "interview",
            "file": "test.txt"
        },
        {
            "text": "Let me explain what happens when someone has complex trauma. First, their nervous system becomes dysregulated. Second, they develop coping mechanisms that may have helped them survive but now cause problems. The key thing to understand is that this is a normal response to abnormal circumstances.",
            "style": "educational", 
            "confidence": 1.0,
            "quality": 0.8,
            "source": "teaching",
            "file": "test.txt"
        }
    ]
    
    print("=== AI CONTENT ROUTER TEST ===\n")
    
    for i, segment in enumerate(test_segments, 1):
        result = router.process_segment(segment)
        
        print(f"**Test {i}**")
        print(f"**Format Detected**: {result['ai_analysis']['format_detected']}")
        print(f"**Strategy**: {result['ai_analysis']['strategy_used']}")
        print(f"**Reasoning**: {result['ai_analysis']['reasoning']}")
        print(f"**Generated Q**: {result['input']}")
        print(f"**Original A**: {result['output'][:200]}...")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_ai_router()
