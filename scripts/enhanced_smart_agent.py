#!/usr/bin/env python3
"""
Enhanced Smart Agent - Properly cleans and separates Q/A pairs
"""

import re
from typing import Dict, Tuple, Optional

class EnhancedSmartAgent:
    def __init__(self):
        pass
    
    def clean_contextual_setup(self, text: str) -> str:
        """Remove contextual setup that references previous conversation"""
        
        # Patterns that indicate contextual setup to remove
        setup_patterns = [
            r'^[^.]*?I guess to take this.*?further[^.]*?\.',
            r'^[^.]*?I\'ve heard you.*?about[^.]*?\.',
            r'^[^.]*?you mentioned.*?earlier[^.]*?\.',
            r'^[^.]*?building on that[^.]*?\.',
            r'^[^.]*?following up on[^.]*?\.',
            r'^[^.]*?you talked about[^.]*?\.'
        ]
        
        cleaned_text = text
        for pattern in setup_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
        
        return cleaned_text
    
    def extract_embedded_qa(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract embedded question and Tim's response"""
        
        # Clean contextual setup first
        cleaned_text = self.clean_contextual_setup(text)
        
        # Look for question patterns
        question_patterns = [
            r'(How can [^?]+\?)',
            r'(What should [^?]+\?)',
            r'(Why do [^?]+\?)',
            r'(When does [^?]+\?)',
            r'(Can you [^?]+\?)',
            r'([^.]*\?)'  # Any sentence ending with ?
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                
                # Find where Tim's response starts
                question_end = match.end()
                remaining_text = cleaned_text[question_end:].strip()
                
                # Look for response indicators
                response_indicators = [
                    r'[,\s]*that\'s a huge question',
                    r'[,\s]*well[,\s]',
                    r'[,\s]*so[,\s]',
                    r'[,\s]*unfortunately[,\s]',
                    r'[,\s]*look[,\s]',
                    r'[,\s]*the thing is[,\s]'
                ]
                
                for indicator_pattern in response_indicators:
                    indicator_match = re.search(indicator_pattern, remaining_text, re.IGNORECASE)
                    if indicator_match:
                        # Found response start
                        response_start = indicator_match.start()
                        response = remaining_text[response_start:].strip()
                        
                        # Clean up response start
                        response = re.sub(r'^[,\s]*', '', response)
                        
                        if len(response) > 50:  # Ensure substantial response
                            return question, response
                
                # If no clear response indicator, take everything after question
                if len(remaining_text) > 50:
                    # Clean up any leading punctuation/whitespace
                    response = re.sub(r'^[,\s.]*', '', remaining_text).strip()
                    if response:
                        return question, response
        
        return None
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment with enhanced Q/A extraction"""
        
        text = segment['text']
        
        # Try to extract embedded Q/A
        qa_pair = self.extract_embedded_qa(text)
        
        if qa_pair:
            question, answer = qa_pair
            
            return {
                "input": question,
                "output": answer,
                "style": segment['style'],
                "confidence": segment['confidence'],
                "quality": segment['quality'],
                "source": segment['source'],
                "file": segment['file'],
                "extraction_method": "embedded_qa_extracted"
            }
        
        # Fallback: create contextual question for cleaned text
        cleaned_text = self.clean_contextual_setup(text)
        
        # Generate appropriate question based on content
        if "trauma" in cleaned_text.lower() and "therap" in cleaned_text.lower():
            question = "What should someone know about finding trauma therapy?"
        elif "narcissist" in cleaned_text.lower():
            question = "Can you explain narcissistic behavior?"
        elif "shame" in cleaned_text.lower():
            question = "How does shame affect people?"
        else:
            question = f"Can you share your {segment['style']} perspective on this?"
        
        return {
            "input": question,
            "output": cleaned_text,
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "extraction_method": "contextual_fallback"
        }

def test_enhanced_agent():
    """Test the enhanced smart agent"""
    agent = EnhancedSmartAgent()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma. Not in. They learn something about PTSD, which is a specific form of trauma. But they don't learn about the traumatic basis of depression and anxiety and ADHD. And they learn nothing about it. it's very difficult to find good help within the medical system. Now, many therapists also don't get any such training. There's a lot of therapists that are designed only to change your beliefs and your behaviors, but not to address the fundamental reasons for those behaviors. a lot of psychologists trained in CBT, cognitive behavioral therapy, or dialectical behavioral therapy. A lot of them are not really. And I know this. Believe me, I know this. They just don't know much about or anything about trauma. Then they can't help you with the fundamental wound that you're carrying. They can help you with the manifestations. And that's not useless. But they can't help you heal at your core. then there are therapies that are deeper than that. There is body-based therapies such as somatic experiencing developed by my friend and teacher, Dr. Peter Levine. There's sensory motor psychotherapy developed by Pact Ogden. There's EMDR that works for some people. There's internal family systems. There's a lot of different therapies that are developed by my friend and colleague, Dr. Richard Schwartz. There's compassionate inquiry, which is based on my work. And I train therapists in that method. There are others, other names I could mention.",
        "style": "therapeutic",
        "confidence": 3.076923076923077,
        "quality": 0.7,
        "source": "doug_bopst",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== ENHANCED SMART AGENT TEST ===\n")
    print(f"**Extraction Method**: {result['extraction_method']}")
    print(f"**Question**: {result['input']}")
    print(f"\n**Answer**:")
    print(result['output'])

if __name__ == "__main__":
    test_enhanced_agent()
