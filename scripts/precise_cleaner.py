#!/usr/bin/env python3
"""
Precise cleaner - removes only the specific contextual opening
"""

import re
from typing import Dict, Tuple, Optional

class PreciseCleaner:
    def __init__(self):
        pass
    
    def clean_specific_opening(self, text: str) -> str:
        """Remove only the specific contextual opening phrase"""
        
        # Remove only: "And I guess to take this one step further, I've heard you talk about"
        pattern = r'^And I guess to take this one step further, I\'ve heard you talk about\s*'
        cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
        return cleaned
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment with precise cleaning"""
        
        cleaned_text = self.clean_specific_opening(segment['text'])
        
        return {
            "input": "Test question",
            "output": cleaned_text,
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file']
        }

def test_precise_cleaner():
    """Test precise cleaning"""
    cleaner = PreciseCleaner()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician.",
        "style": "therapeutic",
        "confidence": 3.0,
        "quality": 0.7,
        "source": "test",
        "file": "test.txt"
    }
    
    result = cleaner.process_segment(test_segment)
    
    print("=== PRECISE CLEANING TEST ===\n")
    print("**Cleaned Text**:")
    print(result['output'])

if __name__ == "__main__":
    test_precise_cleaner()
