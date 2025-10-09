#!/usr/bin/env python3
"""
Debug the refined agent to see why question extraction isn't working
"""

import re
import sys
sys.path.append('/root/pixelated/ai/scripts')
from refined_intelligent_agent import RefinedIntelligentAgent

def debug_question_extraction():
    """Debug question extraction step by step"""
    
    text = "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training."
    
    agent = RefinedIntelligentAgent()
    
    print("=== DEBUGGING QUESTION EXTRACTION ===\n")
    
    # Test each question pattern
    question_patterns = [
        r'(How can [^?]+find[^?]+trained[^?]+\?)',
        r'(How can [^?]+self-regulate[^?]+\?)',
        r'(How can [^?]+\?)',
        r'([A-Z][^.!?]{20,}\?)'
    ]
    
    print("**Testing question patterns:**")
    for i, pattern in enumerate(question_patterns):
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        print(f"Pattern {i+1}: {pattern}")
        print(f"Matches found: {len(matches)}")
        for match in matches:
            print(f"  - '{match.group(1)}'")
        print()
    
    # Test the actual extraction method
    print("**Agent extraction result:**")
    result = agent.extract_question_patterns(text)
    print(f"Question: {result['question']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Position: {result['position']}")
    
    # Manual search for the specific question we expect
    print("\n**Manual search for expected question:**")
    expected_question = "How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves"
    if expected_question in text:
        start_pos = text.find(expected_question)
        print(f"Found at position: {start_pos}")
        print(f"Full question: '{expected_question}?'")
    else:
        print("Expected question not found")

if __name__ == "__main__":
    debug_question_extraction()
