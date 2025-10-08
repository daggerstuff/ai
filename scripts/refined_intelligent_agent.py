#!/usr/bin/env python3
"""
Refined Intelligent Agent - Enhanced dialogue detection for embedded Q/A patterns
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('/root/pixelated/ai/scripts')
from intelligent_qa_agent import ContentAnalysis, IntelligentQAAgent

class RefinedIntelligentAgent(IntelligentQAAgent):
    def __init__(self):
        super().__init__()
    
    def detect_dialogue_patterns(self, text: str) -> Dict:
        """Enhanced dialogue detection for embedded Q/A patterns"""
        
        confidence = 0.0
        indicators = []
        
        # Enhanced pattern 1: Interview setup language
        interview_setup_patterns = [
            r'i\'ve heard you talk about',
            r'you mentioned',
            r'you said',
            r'can you tell us',
            r'what would you say about',
            r'how would you respond to'
        ]
        
        setup_found = any(re.search(pattern, text, re.IGNORECASE) for pattern in interview_setup_patterns)
        if setup_found:
            confidence += 0.3
            indicators.append("interview_setup")
        
        # Enhanced pattern 2: Question-response sequence detection
        questions = re.findall(r'[^.!?]*\?', text)
        if questions:
            # Look for response indicators after questions
            for question in questions:
                question_pos = text.find(question)
                if question_pos != -1:
                    after_question = text[question_pos + len(question):].strip()
                    
                    # Strong response indicators
                    strong_indicators = [
                        r'that\'s a (huge|big|great|good) question',
                        r'well[,\s]',
                        r'unfortunately[,\s]',
                        r'look[,\s]',
                        r'the thing is[,\s]'
                    ]
                    
                    if any(re.search(indicator, after_question, re.IGNORECASE) for indicator in strong_indicators):
                        confidence += 0.4
                        indicators.append("strong_response_marker")
                        break
                    
                    # Weaker response indicators
                    weak_indicators = [r'so[,\s]', r'what happens', r'because']
                    if any(re.search(indicator, after_question, re.IGNORECASE) for indicator in weak_indicators):
                        confidence += 0.2
                        indicators.append("weak_response_marker")
        
        # Enhanced pattern 3: Conversational flow markers
        flow_patterns = [
            r'and i guess to take this.*?further',
            r'building on that',
            r'following up',
            r'to expand on'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in flow_patterns):
            confidence += 0.2
            indicators.append("conversational_flow")
        
        # Enhanced pattern 4: Professional context indicators
        professional_patterns = [
            r'i\'m a (physician|doctor|therapist)',
            r'in my (practice|experience|training)',
            r'i\'ve been through.*?training',
            r'my friend and (teacher|colleague)'
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in professional_patterns):
            confidence += 0.1
            indicators.append("professional_context")
        
        return {
            'confidence': min(confidence, 1.0),
            'indicators': indicators
        }
    
    def extract_question_patterns(self, text: str) -> Dict:
        """Enhanced question extraction with better boundary detection"""
        
        # More sophisticated question patterns
        question_patterns = [
            # Specific therapeutic question patterns
            r'(How can [^?]+find[^?]+trained[^?]+\?)',
            r'(How can [^?]+self-regulate[^?]+\?)',
            r'(What should [^?]+know about[^?]+\?)',
            r'(How do [^?]+deal with[^?]+\?)',
            r'(Why do [^?]+become[^?]+\?)',
            
            # General question patterns
            r'(How can [^?]+\?)',
            r'(What should [^?]+\?)',
            r'(Why do [^?]+\?)',
            r'(Can you [^?]+\?)',
            r'(Would you [^?]+\?)',
            
            # Catch-all for any substantial question
            r'([A-Z][^.!?]{20,}\?)'
        ]
        
        questions = []
        for i, pattern in enumerate(question_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                question = match.group(1).strip()
                
                # Quality scoring based on pattern specificity
                base_confidence = 0.9 - (i * 0.1)  # Earlier patterns get higher confidence
                
                # Additional quality factors
                word_count = len(question.split())
                if 6 <= word_count <= 25:
                    base_confidence += 0.1
                
                # Therapeutic relevance boost
                therapeutic_terms = ['trauma', 'therapy', 'heal', 'trained', 'self-regulate']
                if any(term in question.lower() for term in therapeutic_terms):
                    base_confidence += 0.2
                
                questions.append({
                    'text': question,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': min(base_confidence, 1.0)
                })
        
        # Return highest confidence question
        if questions:
            best_question = max(questions, key=lambda q: q['confidence'])
            return {
                'question': best_question['text'],
                'position': (best_question['start'], best_question['end']),
                'confidence': best_question['confidence']
            }
        
        return {'question': None, 'position': None, 'confidence': 0.0}
    
    def detect_response_boundaries(self, text: str, question_signals: Dict) -> Dict:
        """Enhanced response boundary detection"""
        
        if not question_signals['question'] or not question_signals['position']:
            return {'response': None, 'confidence': 0.0}
        
        question_end = question_signals['position'][1]
        remaining_text = text[question_end:].strip()
        
        # Enhanced response markers with better patterns
        response_markers = [
            # Strong markers (high confidence)
            (r'[,\s]*that\'s a (huge|big|great|good) question[,\s]*', 0.9),
            (r'[,\s]*well[,\s]+', 0.8),
            (r'[,\s]*unfortunately[,\s]*,?\s*look[,\s]*', 0.9),
            (r'[,\s]*unfortunately[,\s]*', 0.7),
            (r'[,\s]*look[,\s]*,', 0.8),
            
            # Medium markers
            (r'[,\s]*so[,\s]+', 0.6),
            (r'[,\s]*the thing is[,\s]*', 0.7),
            (r'[,\s]*what happens is[,\s]*', 0.7),
            (r'[,\s]*because[,\s]+', 0.5),
            
            # Weak markers (fallback)
            (r'[,\s]*and[,\s]+', 0.3),
            (r'[,\s]*but[,\s]+', 0.4)
        ]
        
        best_response = None
        best_confidence = 0.0
        
        for marker_pattern, marker_confidence in response_markers:
            match = re.search(marker_pattern, remaining_text, re.IGNORECASE)
            if match:
                response_start = match.start()
                response = remaining_text[response_start:].strip()
                
                # Clean leading punctuation and whitespace
                response = re.sub(r'^[,\s]*', '', response)
                
                # Quality checks
                word_count = len(response.split())
                if word_count >= 15:  # Substantial response
                    
                    # Boost confidence for longer, more detailed responses
                    length_bonus = min(word_count / 100, 0.2)
                    final_confidence = marker_confidence + length_bonus
                    
                    if final_confidence > best_confidence:
                        best_response = response
                        best_confidence = final_confidence
        
        if best_response:
            return {
                'response': best_response,
                'confidence': best_confidence,
                'method': 'marker_detection'
            }
        
        # Enhanced fallback: look for natural sentence boundaries
        sentences = re.split(r'[.!?]+', remaining_text)
        if len(sentences) >= 2:
            # Skip first sentence if it's very short (likely fragment)
            start_idx = 1 if len(sentences[0].split()) < 3 else 0
            response = '. '.join(sentences[start_idx:]).strip()
            
            if len(response.split()) >= 20:
                return {
                    'response': response,
                    'confidence': 0.4,
                    'method': 'sentence_boundary'
                }
        
        return {'response': None, 'confidence': 0.0}
    
    def clean_setup_references(self, text: str) -> str:
        """Enhanced setup cleaning with better pattern recognition"""
        
        # More precise setup patterns
        setup_patterns = [
            # Exact pattern from our test case
            r'^And I guess to take this one step further,\s*',
            
            # Other common patterns
            r'^Building on that,?\s*',
            r'^Following up on.*?,\s*',
            r'^To expand on.*?,\s*',
            r'^You mentioned.*?,\s*'
        ]
        
        cleaned = text
        for pattern in setup_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        return cleaned

def test_refined_agent():
    """Test the refined intelligent agent"""
    agent = RefinedIntelligentAgent()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma. Not in. They learn something about PTSD, which is a specific form of trauma. But they don't learn about the traumatic basis of depression and anxiety and ADHD. And they learn nothing about it. it's very difficult to find good help within the medical system. Now, many therapists also don't get any such training. There's a lot of therapists that are designed only to change your beliefs and your behaviors, but not to address the fundamental reasons for those behaviors. a lot of psychologists trained in CBT, cognitive behavioral therapy, or dialectical behavioral therapy. A lot of them are not really. And I know this. Believe me, I know this. They just don't know much about or anything about trauma. Then they can't help you with the fundamental wound that you're carrying. They can help you with the manifestations. And that's not useless. But they can't help you heal at your core. then there are therapies that are deeper than that. There is body-based therapies such as somatic experiencing developed by my friend and teacher, Dr. Peter Levine. There's sensory motor psychotherapy developed by Pact Ogden. There's EMDR that works for some people. There's internal family systems. There's a lot of different therapies that are developed by my friend and colleague, Dr. Richard Schwartz. There's compassionate inquiry, which is based on my work. And I train therapists in that method. There are others, other names I could mention.",
        "style": "therapeutic",
        "confidence": 3.076923076923077,
        "quality": 0.7,
        "source": "doug_bopst",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== REFINED INTELLIGENT AGENT TEST ===\n")
    print(f"**Method**: {result['agent_analysis']['method']}")
    print(f"**Dialogue Confidence**: {result['agent_analysis']['dialogue_confidence']:.2f}")
    print(f"**Semantic Coherence**: {result['agent_analysis']['semantic_coherence']:.2f}")
    print(f"**Content Type**: {result['agent_analysis']['content_type']}")
    print(f"\n**Question**: {result['input']}")
    print(f"\n**Answer**: {result['output'][:400]}...")

if __name__ == "__main__":
    test_refined_agent()
