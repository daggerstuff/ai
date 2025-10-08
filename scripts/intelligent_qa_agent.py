#!/usr/bin/env python3
"""
Intelligent Q/A Agent - Multi-pattern analysis with nuanced decision making
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ContentAnalysis:
    dialogue_confidence: float = 0.0
    question_embedded: Optional[str] = None
    response_portion: Optional[str] = None
    content_type: str = "unknown"
    setup_detected: bool = False
    speaker_transitions: List[int] = None
    semantic_coherence: float = 0.0

class IntelligentQAAgent:
    def __init__(self):
        pass
    
    def multi_pattern_analysis(self, text: str) -> ContentAnalysis:
        """Analyze text using multiple patterns simultaneously"""
        
        analysis = ContentAnalysis()
        analysis.speaker_transitions = []
        
        # Pattern 1: Dialogue structure detection
        dialogue_signals = self.detect_dialogue_patterns(text)
        analysis.dialogue_confidence = dialogue_signals['confidence']
        
        # Pattern 2: Setup/context detection
        setup_signals = self.detect_setup_patterns(text)
        analysis.setup_detected = setup_signals['has_setup']
        
        # Pattern 3: Question extraction
        question_signals = self.extract_question_patterns(text)
        analysis.question_embedded = question_signals['question']
        
        # Pattern 4: Response boundary detection
        response_signals = self.detect_response_boundaries(text, question_signals)
        analysis.response_portion = response_signals['response']
        
        # Pattern 5: Content type classification
        content_signals = self.classify_content_type(text)
        analysis.content_type = content_signals['type']
        
        # Pattern 6: Semantic coherence assessment
        if analysis.question_embedded and analysis.response_portion:
            analysis.semantic_coherence = self.assess_semantic_coherence(
                analysis.question_embedded, analysis.response_portion
            )
        
        return analysis
    
    def detect_dialogue_patterns(self, text: str) -> Dict:
        """Detect various dialogue structure patterns"""
        
        confidence = 0.0
        indicators = []
        
        # Explicit dialogue markers
        if re.search(r'(HOST|INTERVIEWER|GUEST):', text, re.IGNORECASE):
            confidence += 0.4
            indicators.append("explicit_markers")
        
        # Question-response flow
        questions = re.findall(r'[^.!?]*\?', text)
        if questions:
            confidence += 0.3 * min(len(questions), 2)
            indicators.append("question_markers")
        
        # Conversational transitions
        transitions = ['you mentioned', 'you said', 'you talked about', 'building on that']
        if any(trans in text.lower() for trans in transitions):
            confidence += 0.2
            indicators.append("conversational_flow")
        
        # Response indicators after questions
        response_starters = ['that\'s', 'well', 'so', 'unfortunately', 'look', 'the thing is']
        if any(starter in text.lower() for starter in response_starters):
            confidence += 0.1
            indicators.append("response_markers")
        
        return {
            'confidence': min(confidence, 1.0),
            'indicators': indicators
        }
    
    def detect_setup_patterns(self, text: str) -> Dict:
        """Detect contextual setup that references previous conversation"""
        
        setup_patterns = [
            r'and i guess to take this.*?further',
            r'i\'ve heard you.*?about',
            r'you mentioned.*?earlier',
            r'building on.*?that',
            r'following up on'
        ]
        
        has_setup = any(re.search(pattern, text, re.IGNORECASE) for pattern in setup_patterns)
        
        setup_end = 0
        if has_setup:
            for pattern in setup_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    setup_end = max(setup_end, match.end())
        
        return {
            'has_setup': has_setup,
            'setup_end': setup_end
        }
    
    def extract_question_patterns(self, text: str) -> Dict:
        """Extract embedded questions using multiple approaches"""
        
        # Direct question extraction
        question_patterns = [
            r'(How can [^?]+\?)',
            r'(What should [^?]+\?)',
            r'(Why do [^?]+\?)',
            r'(Can you [^?]+\?)',
            r'(Would you [^?]+\?)',
            r'([A-Z][^.!?]*\?)'  # Any capitalized sentence ending with ?
        ]
        
        questions = []
        for pattern in question_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                question = match.group(1).strip()
                # Validate it's a substantial question
                if len(question.split()) >= 4:
                    questions.append({
                        'text': question,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self.assess_question_quality(question)
                    })
        
        # Return best question
        if questions:
            best_question = max(questions, key=lambda q: q['confidence'])
            return {
                'question': best_question['text'],
                'position': (best_question['start'], best_question['end']),
                'confidence': best_question['confidence']
            }
        
        return {'question': None, 'position': None, 'confidence': 0.0}
    
    def detect_response_boundaries(self, text: str, question_signals: Dict) -> Dict:
        """Detect where the actual response begins"""
        
        if not question_signals['question'] or not question_signals['position']:
            return {'response': None, 'confidence': 0.0}
        
        question_end = question_signals['position'][1]
        remaining_text = text[question_end:].strip()
        
        # Look for response transition markers
        response_markers = [
            r'[,\s]*that\'s a (huge|big|great) question',
            r'[,\s]*well[,\s]',
            r'[,\s]*so[,\s]',
            r'[,\s]*unfortunately[,\s]',
            r'[,\s]*look[,\s]',
            r'[,\s]*the thing is[,\s]',
            r'[,\s]*what happens is[,\s]'
        ]
        
        for marker_pattern in response_markers:
            match = re.search(marker_pattern, remaining_text, re.IGNORECASE)
            if match:
                response_start = match.start()
                response = remaining_text[response_start:].strip()
                # Clean leading punctuation
                response = re.sub(r'^[,\s]*', '', response)
                
                if len(response.split()) >= 10:  # Substantial response
                    return {
                        'response': response,
                        'confidence': 0.8,
                        'marker_used': marker_pattern
                    }
        
        # Fallback: take everything after question if substantial
        if len(remaining_text.split()) >= 15:
            cleaned_response = re.sub(r'^[,\s.]*', '', remaining_text).strip()
            return {
                'response': cleaned_response,
                'confidence': 0.5,
                'marker_used': 'fallback'
            }
        
        return {'response': None, 'confidence': 0.0}
    
    def classify_content_type(self, text: str) -> Dict:
        """Classify the type of content"""
        
        text_lower = text.lower()
        
        # Interview indicators
        interview_score = 0
        if any(word in text_lower for word in ['interviewer', 'host', 'guest', 'asked']):
            interview_score += 0.3
        if '?' in text and 'that\'s' in text_lower:
            interview_score += 0.4
        
        # Teaching indicators  
        teaching_score = 0
        if any(phrase in text_lower for phrase in ['let me explain', 'what happens is', 'the key thing']):
            teaching_score += 0.4
        if any(word in text_lower for word in ['first', 'second', 'then', 'finally']):
            teaching_score += 0.2
        
        # Story indicators
        story_score = 0
        if any(phrase in text_lower for phrase in ['i remember', 'there was', 'for example', 'i had a client']):
            story_score += 0.4
        
        scores = {
            'interview': interview_score,
            'teaching': teaching_score, 
            'story': story_score
        }
        
        content_type = max(scores, key=scores.get) if max(scores.values()) > 0.3 else 'general'
        
        return {
            'type': content_type,
            'confidence': scores[content_type] if content_type in scores else 0.3
        }
    
    def assess_question_quality(self, question: str) -> float:
        """Assess the quality of an extracted question"""
        
        score = 0.5  # Base score
        
        # Length check
        word_count = len(question.split())
        if 4 <= word_count <= 20:
            score += 0.2
        
        # Question word presence
        question_words = ['how', 'what', 'why', 'when', 'where', 'can', 'would', 'should']
        if any(word in question.lower() for word in question_words):
            score += 0.2
        
        # Therapeutic relevance
        therapeutic_terms = ['trauma', 'therapy', 'heal', 'help', 'understand', 'feel']
        if any(term in question.lower() for term in therapeutic_terms):
            score += 0.1
        
        return min(score, 1.0)
    
    def assess_semantic_coherence(self, question: str, response: str) -> float:
        """Assess if question and response are semantically coherent"""
        
        # Extract key terms from question and response
        q_words = set(re.findall(r'\b\w+\b', question.lower()))
        r_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        q_keywords = q_words - common_words
        r_keywords = r_words - common_words
        
        # Calculate overlap
        if not q_keywords:
            return 0.3
        
        overlap = len(q_keywords & r_keywords)
        overlap_ratio = overlap / len(q_keywords)
        
        # Boost score if response is substantial
        if len(response.split()) > 50:
            overlap_ratio += 0.2
        
        return min(overlap_ratio, 1.0)
    
    def make_intelligent_decision(self, analysis: ContentAnalysis, text: str, style: str) -> Dict:
        """Make nuanced decision based on multi-pattern analysis"""
        
        # High confidence dialogue with good Q/A extraction
        if (analysis.dialogue_confidence > 0.6 and 
            analysis.question_embedded and 
            analysis.response_portion and
            analysis.semantic_coherence > 0.4):
            
            return {
                "input": analysis.question_embedded,
                "output": analysis.response_portion,
                "method": "dialogue_extraction",
                "confidence": analysis.semantic_coherence
            }
        
        # Medium confidence dialogue - try to improve extraction
        elif (analysis.dialogue_confidence > 0.4 and analysis.question_embedded):
            
            # If no clean response boundary, use contextual approach
            if not analysis.response_portion:
                contextual_question = self.generate_contextual_question(text, style)
                cleaned_text = self.clean_setup_references(text)
                
                return {
                    "input": contextual_question,
                    "output": cleaned_text,
                    "method": "hybrid_contextual",
                    "confidence": 0.6
                }
            else:
                return {
                    "input": analysis.question_embedded,
                    "output": analysis.response_portion,
                    "method": "partial_extraction",
                    "confidence": 0.7
                }
        
        # No clear dialogue structure - use contextual generation
        else:
            contextual_question = self.generate_contextual_question(text, style)
            cleaned_text = self.clean_setup_references(text)
            
            return {
                "input": contextual_question,
                "output": cleaned_text,
                "method": "contextual_generation",
                "confidence": 0.5
            }
    
    def clean_setup_references(self, text: str) -> str:
        """Clean contextual setup references"""
        
        # Remove specific setup patterns
        setup_patterns = [
            r'^And I guess to take this one step further,?\s*',
            r'^Building on that,?\s*',
            r'^Following up on.*?,\s*'
        ]
        
        cleaned = text
        for pattern in setup_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        return cleaned
    
    def generate_contextual_question(self, text: str, style: str) -> str:
        """Generate contextually appropriate question"""
        
        text_lower = text.lower()
        
        # Topic-specific questions
        if "trauma" in text_lower and "therap" in text_lower:
            return "How can someone find proper trauma therapy and what should they look for?"
        elif "narcissist" in text_lower and "manipulat" in text_lower:
            return "Why do narcissists become manipulative and how does this affect relationships?"
        elif "shame" in text_lower:
            return "How does shame affect people and what can be done about it?"
        elif "heal" in text_lower and "core" in text_lower:
            return "What's the difference between treating symptoms and healing at the core?"
        
        # Style-based fallbacks
        style_questions = {
            "therapeutic": "Can you share your therapeutic perspective on this?",
            "educational": "Can you explain this concept?", 
            "empathetic": "Can you help me understand this?",
            "practical": "What should I know about this situation?"
        }
        
        return style_questions.get(style, "Can you help me understand this?")
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment with intelligent multi-pattern analysis"""
        
        # Perform comprehensive analysis
        analysis = self.multi_pattern_analysis(segment['text'])
        
        # Make intelligent decision
        result = self.make_intelligent_decision(analysis, segment['text'], segment['style'])
        
        return {
            "input": result["input"],
            "output": result["output"],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "agent_analysis": {
                "method": result["method"],
                "extraction_confidence": result["confidence"],
                "dialogue_confidence": analysis.dialogue_confidence,
                "semantic_coherence": analysis.semantic_coherence,
                "content_type": analysis.content_type
            }
        }

def test_intelligent_agent():
    """Test the intelligent agent"""
    agent = IntelligentQAAgent()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma. Not in. They learn something about PTSD, which is a specific form of trauma. But they don't learn about the traumatic basis of depression and anxiety and ADHD. And they learn nothing about it. it's very difficult to find good help within the medical system. Now, many therapists also don't get any such training. There's a lot of therapists that are designed only to change your beliefs and your behaviors, but not to address the fundamental reasons for those behaviors. a lot of psychologists trained in CBT, cognitive behavioral therapy, or dialectical behavioral therapy. A lot of them are not really. And I know this. Believe me, I know this. They just don't know much about or anything about trauma. Then they can't help you with the fundamental wound that you're carrying. They can help you with the manifestations. And that's not useless. But they can't help you heal at your core. then there are therapies that are deeper than that. There is body-based therapies such as somatic experiencing developed by my friend and teacher, Dr. Peter Levine. There's sensory motor psychotherapy developed by Pact Ogden. There's EMDR that works for some people. There's internal family systems. There's a lot of different therapies that are developed by my friend and colleague, Dr. Richard Schwartz. There's compassionate inquiry, which is based on my work. And I train therapists in that method. There are others, other names I could mention.",
        "style": "therapeutic",
        "confidence": 3.076923076923077,
        "quality": 0.7,
        "source": "doug_bopst",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== INTELLIGENT AGENT TEST ===\n")
    print(f"**Method**: {result['agent_analysis']['method']}")
    print(f"**Dialogue Confidence**: {result['agent_analysis']['dialogue_confidence']:.2f}")
    print(f"**Semantic Coherence**: {result['agent_analysis']['semantic_coherence']:.2f}")
    print(f"**Content Type**: {result['agent_analysis']['content_type']}")
    print(f"\n**Question**: {result['input']}")
    print(f"\n**Answer**: {result['output'][:300]}...")

if __name__ == "__main__":
    test_intelligent_agent()
