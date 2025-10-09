#!/usr/bin/env python3
"""
Fixed Refined Agent - Corrected question extraction patterns
"""

import re
import sys
sys.path.append('/root/pixelated/ai/scripts')
from refined_intelligent_agent import RefinedIntelligentAgent

class FixedRefinedAgent(RefinedIntelligentAgent):
    def __init__(self):
        super().__init__()
    
    def extract_question_patterns(self, text: str) -> Dict:
        """Fixed question extraction with proper punctuation handling"""
        
        # Fixed patterns that handle punctuation and sentence structure properly
        question_patterns = [
            # Direct question extraction - match everything until ?
            r'(How can [^?]*\?)',
            r'(What should [^?]*\?)',
            r'(Why do [^?]*\?)',
            r'(Can you [^?]*\?)',
            r'(Would you [^?]*\?)',
            r'(When does [^?]*\?)',
            r'(Where can [^?]*\?)',
            
            # Catch-all for any question
            r'([A-Z][^?]*\?)'
        ]
        
        questions = []
        for i, pattern in enumerate(question_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                question = match.group(1).strip()
                
                # Clean up the question
                question = re.sub(r'\s+', ' ', question)  # Normalize whitespace
                
                # Quality scoring
                word_count = len(question.split())
                
                # Skip very short questions
                if word_count < 4:
                    continue
                
                # Base confidence based on pattern specificity
                base_confidence = 0.9 - (i * 0.05)
                
                # Length bonus for substantial questions
                if 8 <= word_count <= 30:
                    base_confidence += 0.1
                
                # Therapeutic relevance boost
                therapeutic_terms = ['trauma', 'therapy', 'heal', 'trained', 'self-regulate', 'help', 'find']
                relevance_score = sum(1 for term in therapeutic_terms if term in question.lower())
                base_confidence += relevance_score * 0.05
                
                questions.append({
                    'text': question,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': min(base_confidence, 1.0),
                    'word_count': word_count
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

def test_fixed_agent():
    """Test the fixed agent"""
    agent = FixedRefinedAgent()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma. Not in. They learn something about PTSD, which is a specific form of trauma. But they don't learn about the traumatic basis of depression and anxiety and ADHD. And they learn nothing about it. it's very difficult to find good help within the medical system. Now, many therapists also don't get any such training. There's a lot of therapists that are designed only to change your beliefs and your behaviors, but not to address the fundamental reasons for those behaviors. a lot of psychologists trained in CBT, cognitive behavioral therapy, or dialectical behavioral therapy. A lot of them are not really. And I know this. Believe me, I know this. They just don't know much about or anything about trauma. Then they can't help you with the fundamental wound that you're carrying. They can help you with the manifestations. And that's not useless. But they can't help you heal at your core. then there are therapies that are deeper than that. There is body-based therapies such as somatic experiencing developed by my friend and teacher, Dr. Peter Levine. There's sensory motor psychotherapy developed by Pact Ogden. There's EMDR that works for some people. There's internal family systems. There's a lot of different therapies that are developed by my friend and colleague, Dr. Richard Schwartz. There's compassionate inquiry, which is based on my work. And I train therapists in that method. There are others, other names I could mention.",
        "style": "therapeutic",
        "confidence": 3.076923076923077,
        "quality": 0.7,
        "source": "doug_bopst",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== FIXED REFINED AGENT TEST ===\n")
    print(f"**Method**: {result['agent_analysis']['method']}")
    print(f"**Dialogue Confidence**: {result['agent_analysis']['dialogue_confidence']:.2f}")
    print(f"**Semantic Coherence**: {result['agent_analysis']['semantic_coherence']:.2f}")
    print(f"**Extraction Confidence**: {result['agent_analysis']['extraction_confidence']:.2f}")
    print(f"\n**Question**: {result['input']}")
    print(f"\n**Answer**: {result['output'][:400]}...")

if __name__ == "__main__":
    test_fixed_agent()
