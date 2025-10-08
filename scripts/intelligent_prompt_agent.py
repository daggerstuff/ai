#!/usr/bin/env python3
"""
Intelligent Multi-Pattern Prompt Generation Agent
Replaces simple template system with sophisticated content analysis for therapeutic training data.

Based on session savepoint findings:
- Initial system created 100% generic questions unrelated to segment content
- Need multi-pattern analysis for diverse therapeutic content formats
- Must handle interviews, podcasts, speeches, monologues with confidence weighting
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    INTERVIEW = "interview"
    PODCAST = "podcast" 
    SPEECH = "speech"
    MONOLOGUE = "monologue"
    UNKNOWN = "unknown"

@dataclass
class PatternMatch:
    """Represents a pattern match with confidence scoring"""
    pattern_type: str
    confidence: float
    extracted_question: Optional[str] = None
    response_start: Optional[str] = None
    transition_markers: List[str] = None

class MultiPatternAgent:
    """Intelligent agent for analyzing diverse therapeutic content formats"""
    
    def __init__(self):
        # Transition markers for response boundary detection
        self.transition_markers = [
            "that's a huge question",
            "unfortunately", 
            "look",
            "well",
            "you know",
            "the thing is",
            "what happens is",
            "here's what",
            "let me tell you",
            "i think",
            "the reality is"
        ]
        
        # Interview question detection patterns
        self.question_patterns = [
            # Direct questions (may end with period in natural speech)
            r"how can somebody (?:begin to )?(.{10,100}?)[\.\?]",
            r"what (?:should|would|can) (?:someone|people|you) (.{10,100}?)[\.\?]", 
            r"can you (?:help|explain|tell) (.{10,100}?)[\.\?]",
            r"why (?:do|does|is) (.{10,100}?)[\.\?]",
            r"where (?:do|does) (.{10,100}?)[\.\?]",
            r"when (?:do|does|should) (.{10,100}?)[\.\?]",
            
            # Embedded questions in interview format
            r"(?:interviewer|host|question):\s*(.{10,200}?)[\.\?]",
            r"q:\s*(.{10,200}?)[\.\?]",
            r"question:\s*(.{10,200}?)[\.\?]",
            
            # Natural speech question patterns
            r"so (.{5,100}?) right[\.\?]",
            r"i'm wondering (.{10,100}?)[\.\?]",
            r"could you (?:talk about|discuss) (.{10,100}?)[\.\?]"
        ]
        
        # Response boundary detection patterns
        self.response_boundary_patterns = [
            r"(?:that's a huge question|unfortunately|look|well|you know|the thing is|what happens is|here's what|let me tell you|i think|the reality is)",
            r"(?:now|so|first|basically|essentially|fundamentally)",
            r"(?:when a person|if someone|people who|those who)"
        ]

    def classify_content_type(self, text: str) -> Tuple[ContentType, float]:
        """Classify content type with confidence scoring"""
        text_lower = text.lower()
        indicators = {
            ContentType.INTERVIEW: 0.0,
            ContentType.PODCAST: 0.0,
            ContentType.SPEECH: 0.0,
            ContentType.MONOLOGUE: 0.0
        }
        
        # Interview indicators
        if any(word in text_lower for word in ['interviewer', 'host', 'question:', 'q:', 'asks']):
            indicators[ContentType.INTERVIEW] += 0.4
        if re.search(r'(?:how can|what should|can you|why do)', text_lower):
            indicators[ContentType.INTERVIEW] += 0.3
        if len(re.findall(r'\?', text)) > 0:
            indicators[ContentType.INTERVIEW] += 0.2
            
        # Podcast indicators  
        if any(word in text_lower for word in ['podcast', 'episode', 'today we', 'our guest']):
            indicators[ContentType.PODCAST] += 0.4
        if re.search(r'(?:welcome|today|discussion)', text_lower):
            indicators[ContentType.PODCAST] += 0.2
            
        # Speech indicators
        if any(word in text_lower for word in ['audience', 'thank you all', 'gathered here', 'presentation']):
            indicators[ContentType.SPEECH] += 0.4
        if text.count('\n') < 3 and len(text) > 500:  # Long continuous text
            indicators[ContentType.SPEECH] += 0.2
            
        # Monologue indicators (single speaker, continuous)
        if not any(word in text_lower for word in ['interviewer', 'host', 'question', 'asks', 'you said']):
            indicators[ContentType.MONOLOGUE] += 0.3
        if len(text.split()) > 100 and text.count('?') == 0:
            indicators[ContentType.MONOLOGUE] += 0.2
            
        # Determine best match
        best_type = max(indicators.items(), key=lambda x: x[1])
        return best_type[0], best_type[1]

    def extract_interview_question(self, text: str) -> PatternMatch:
        """Extract actual questions from interview-style content"""
        best_match = PatternMatch("interview_question", 0.0)
        
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                question_part = match.group(1) if match.groups() else match.group(0)
                
                # Clean up the question
                question = question_part.strip()
                if len(question) < 10 or len(question) > 200:
                    continue
                    
                # Calculate confidence based on context
                confidence = 0.5
                
                # Higher confidence for clear question markers
                if '?' in match.group(0):
                    confidence += 0.2
                if any(word in text.lower() for word in ['interviewer', 'host', 'q:']):
                    confidence += 0.2
                if re.search(r'(?:how|what|why|when|where|can|could|would|should)', question, re.IGNORECASE):
                    confidence += 0.1
                    
                if confidence > best_match.confidence:
                    best_match = PatternMatch(
                        "interview_question",
                        confidence,
                        question,
                        None,
                        []
                    )
        
        return best_match

    def detect_response_boundaries(self, text: str, question: str = None) -> PatternMatch:
        """Detect where response begins using transition markers"""
        response_match = PatternMatch("response_boundary", 0.0)
        
        text_lower = text.lower()
        found_markers = []
        
        # Look for transition markers
        for marker in self.transition_markers:
            if marker in text_lower:
                found_markers.append(marker)
                response_match.confidence += 0.1
                
                # Find response start after marker
                marker_pos = text_lower.find(marker)
                response_start = text[marker_pos:marker_pos + 100].strip()
                if not response_match.response_start or len(response_start) > len(response_match.response_start):
                    response_match.response_start = response_start
        
        # Look for response boundary patterns
        for pattern in self.response_boundary_patterns:
            if re.search(pattern, text_lower):
                response_match.confidence += 0.05
                
        response_match.transition_markers = found_markers
        return response_match

    def validate_semantic_coherence(self, question: str, response: str) -> float:
        """Validate that question and response are semantically coherent"""
        if not question or not response:
            return 0.0
            
        question_lower = question.lower()
        response_lower = response.lower()
        
        coherence_score = 0.0
        
        # Check for keyword overlap
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        response_words = set(re.findall(r'\b\w+\b', response_lower))
        
        # Remove common stop words for better matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        question_content = question_words - stop_words
        response_content = response_words - stop_words
        
        if question_content and response_content:
            overlap = len(question_content.intersection(response_content))
            coherence_score += min(0.4, overlap * 0.1)
        
        # Check for thematic coherence
        therapeutic_themes = {
            'trauma': ['trauma', 'ptsd', 'abuse', 'hurt', 'pain', 'healing'],
            'relationships': ['relationship', 'family', 'parent', 'child', 'partner', 'marriage'],
            'mental_health': ['depression', 'anxiety', 'mental', 'emotional', 'therapy', 'therapeutic'],
            'growth': ['growth', 'development', 'change', 'progress', 'journey', 'path'],
            'narcissism': ['narcissist', 'narcissism', 'manipulation', 'gaslighting', 'boundaries']
        }
        
        for theme, keywords in therapeutic_themes.items():
            question_has_theme = any(word in question_lower for word in keywords)
            response_has_theme = any(word in response_lower for word in keywords)
            if question_has_theme and response_has_theme:
                coherence_score += 0.2
                break
                
        # Penalty for obvious mismatches
        if 'how' in question_lower and 'because' not in response_lower and 'by' not in response_lower:
            coherence_score -= 0.1
        if 'what' in question_lower and 'is' not in response_lower and 'are' not in response_lower:
            coherence_score -= 0.1
            
        return max(0.0, min(1.0, coherence_score))

    def analyze_segment(self, segment_text: str) -> Dict:
        """Comprehensive analysis of a therapeutic segment"""
        analysis = {
            "content_type": None,
            "content_confidence": 0.0,
            "extracted_question": None,
            "question_confidence": 0.0,
            "response_boundary": None,
            "boundary_confidence": 0.0,
            "semantic_coherence": 0.0,
            "overall_confidence": 0.0,
            "transition_markers": [],
            "processing_notes": []
        }
        
        # Step 1: Classify content type
        content_type, content_confidence = self.classify_content_type(segment_text)
        analysis["content_type"] = content_type.value
        analysis["content_confidence"] = content_confidence
        
        # Step 2: Extract questions based on content type
        if content_type in [ContentType.INTERVIEW, ContentType.PODCAST]:
            question_match = self.extract_interview_question(segment_text)
            analysis["extracted_question"] = question_match.extracted_question
            analysis["question_confidence"] = question_match.confidence
            
            if question_match.extracted_question:
                # Step 3: Detect response boundaries
                boundary_match = self.detect_response_boundaries(segment_text, question_match.extracted_question)
                analysis["response_boundary"] = boundary_match.response_start
                analysis["boundary_confidence"] = boundary_match.confidence
                analysis["transition_markers"] = boundary_match.transition_markers
                
                # Step 4: Validate semantic coherence
                coherence = self.validate_semantic_coherence(
                    question_match.extracted_question, 
                    segment_text
                )
                analysis["semantic_coherence"] = coherence
        
        # Calculate overall confidence
        confidence_factors = [
            analysis["content_confidence"],
            analysis["question_confidence"], 
            analysis["boundary_confidence"],
            analysis["semantic_coherence"]
        ]
        analysis["overall_confidence"] = sum(confidence_factors) / len(confidence_factors)
        
        # Add processing notes
        if analysis["overall_confidence"] < 0.3:
            analysis["processing_notes"].append("Low confidence - may need manual review")
        if not analysis["extracted_question"]:
            analysis["processing_notes"].append("No clear question found - may be monologue/speech format")
        if analysis["semantic_coherence"] < 0.2:
            analysis["processing_notes"].append("Potential semantic mismatch between question and response")
            
        return analysis

    def generate_contextual_prompt(self, segment: Dict, analysis: Dict) -> str:
        """Generate contextually appropriate prompt based on analysis"""
        
        # If we found a good extracted question, use it
        if (analysis["extracted_question"] and 
            analysis["question_confidence"] > 0.4 and 
            analysis["semantic_coherence"] > 0.3):
            return analysis["extracted_question"]
        
        # Fallback: Generate style-appropriate question based on content analysis
        style = segment.get('style', 'therapeutic')
        text = segment['text']
        
        # Extract actual topics from the text
        topics = self._extract_actual_topics(text)
        selected_topic = topics[0] if topics else "this situation"
        
        # Generate contextual question based on what the text actually discusses
        if 'trauma' in text.lower() or 'hurt' in text.lower():
            if style == 'therapeutic':
                return f"I'm struggling with trauma and healing. Can you help me understand what's happening?"
            elif style == 'practical':
                return f"What specific steps can I take to heal from trauma?"
            elif style == 'empathetic':
                return f"I'm really struggling with past trauma and feeling alone. Can you help?"
            else:  # educational
                return f"Can you explain how trauma affects someone and the healing process?"
        
        elif 'narcissist' in text.lower() or 'manipulation' in text.lower():
            if style == 'therapeutic':
                return f"I think I'm dealing with a narcissist. Can you help me understand this?"
            elif style == 'practical':
                return f"What should I do if I'm in a relationship with a narcissist?"
            elif style == 'empathetic':
                return f"I feel so confused and hurt by narcissistic abuse. Can you help me?"
            else:  # educational
                return f"Can you explain narcissistic behavior and how it affects relationships?"
        
        elif any(word in text.lower() for word in ['relationship', 'family', 'parent']):
            if style == 'therapeutic':
                return f"I'm having relationship difficulties. Can you help me work through this?"
            elif style == 'practical':
                return f"What can I do to improve my relationships?"
            elif style == 'empathetic':
                return f"I'm struggling in my relationships and need support."
            else:  # educational
                return f"Can you explain healthy relationship dynamics?"
        
        # Generic fallback based on style
        style_defaults = {
            'therapeutic': "I'm working through some emotional challenges. Can you offer guidance?",
            'practical': "What practical steps can I take to improve my mental health?", 
            'empathetic': "I'm struggling emotionally and need someone to understand.",
            'educational': "Can you help me understand mental health and healing better?"
        }
        
        return style_defaults.get(style, style_defaults['therapeutic'])

    def _extract_actual_topics(self, text: str) -> List[str]:
        """Extract topics that the text actually discusses (not generic keywords)"""
        topics = []
        text_lower = text.lower()
        
        # Look for specific therapeutic concepts being discussed
        if 'trauma' in text_lower:
            topics.append('trauma')
        if any(word in text_lower for word in ['narcissist', 'manipulation', 'gaslighting']):
            topics.append('narcissistic abuse')
        if any(word in text_lower for word in ['anxiety', 'depression', 'panic']):
            topics.append('anxiety and depression')
        if any(word in text_lower for word in ['relationship', 'family', 'parent']):
            topics.append('relationships')
        if any(word in text_lower for word in ['shame', 'guilt', 'self-worth']):
            topics.append('self-worth and shame')
        if any(word in text_lower for word in ['healing', 'recovery', 'growth']):
            topics.append('healing and recovery')
            
        return topics if topics else ['emotional healing']

# Example usage and testing
if __name__ == "__main__":
    agent = MultiPatternAgent()
    
    # Test with sample interview content
    sample_interview = """
    Interviewer: How can somebody begin to take that path toward healing from complex trauma.
    
    Tim Fletcher: Well, that's a huge question because unfortunately, most people with complex trauma don't realize they have complex trauma. They think they're just anxious or depressed or they have relationship problems, but they don't understand the underlying root cause.
    """
    
    analysis = agent.analyze_segment(sample_interview)
    print("Analysis Results:")
    print(json.dumps(analysis, indent=2))