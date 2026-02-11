#!/usr/bin/env python3
"""
Intelligent Q/A Extractor - Finds embedded questions or creates contextually appropriate ones
"""

import json
import re
from typing import Dict, Tuple, Optional

class IntelligentQAExtractor:
    def __init__(self):
        pass
    
    def extract_embedded_qa(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract embedded question and answer from text"""
        
        # Look for explicit question patterns
        question_patterns = [
            r'(.+?)[\?\.].*?[,\s]+(.*)',  # Question followed by answer
            r'(.+how can.+?\?)\s*(.+)',   # "How can" questions
            r'(.+what should.+?\?)\s*(.+)', # "What should" questions  
            r'(.+why do.+?\?)\s*(.+)',    # "Why do" questions
            r'(.+when.+?\?)\s*(.+)',      # "When" questions
        ]
        
        for pattern in question_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                potential_q = match.group(1).strip()
                potential_a = match.group(2).strip()
                
                # Validate it's actually a question
                if ('?' in potential_q or any(word in potential_q.lower() for word in 
                    ['how', 'what', 'why', 'when', 'where', 'can you', 'should i'])) and len(potential_a) > 100:
                    return potential_q, potential_a
        
        return None
    
    def create_contextual_question(self, text: str, style: str) -> str:
        """Create a contextual question based on what the text actually discusses"""
        
        # Analyze the content to understand what it's explaining
        text_lower = text.lower()
        
        # Find the main subject being discussed
        if re.search(r'narcissist.*manipulat', text_lower):
            if style == "therapeutic":
                return "Why do narcissists become manipulative, and what drives this behavior?"
            else:
                return "Can you explain how narcissists use manipulation?"
                
        elif re.search(r'trauma.*therap', text_lower):
            if style == "therapeutic":
                return "How can someone find proper trauma therapy, and what should they look for?"
            else:
                return "What should I know about finding trauma-informed therapy?"
                
        elif re.search(r'shame.*acknowledg', text_lower):
            return "What happens when someone has unacknowledged shame?"
            
        elif re.search(r'crisis.*training', text_lower):
            return "What should someone do when they're in crisis?"
            
        elif re.search(r'heal.*core', text_lower):
            return "What's the difference between treating symptoms and healing at the core?"
            
        # Look for explanatory content
        elif re.search(r'because.*reason', text_lower):
            # Extract what's being explained
            sentences = re.split(r'[.!?]+', text)
            first_sentence = sentences[0].strip() if sentences else ""
            if len(first_sentence) > 20:
                return f"Can you explain why {first_sentence.lower()}?"
        
        # Look for process descriptions
        elif re.search(r'when.*happens|the process', text_lower):
            return "Can you walk me through how this process works?"
            
        # Look for advice/recommendations
        elif re.search(r'you should|try|recommend', text_lower):
            return "What would you recommend in this situation?"
            
        # Default contextual questions based on style
        if style == "therapeutic":
            return "Can you help me understand this from a therapeutic perspective?"
        elif style == "educational":
            return "Can you explain this concept to me?"
        elif style == "empathetic":
            return "I'm struggling with this. Can you help me understand?"
        else:  # practical
            return "What should I know about this?"
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment to create proper Q/A pair"""
        text = segment['text']
        style = segment['style']
        
        # First try to extract embedded Q/A
        embedded_qa = self.extract_embedded_qa(text)
        
        if embedded_qa:
            question, answer = embedded_qa
            # Clean up the question
            question = re.sub(r'^[^A-Z]*', '', question)  # Remove leading fragments
            question = question.strip()
            if not question.endswith('?'):
                question += '?'
        else:
            # Create contextual question
            question = self.create_contextual_question(text, style)
            answer = text
        
        return {
            "input": question,
            "output": answer,
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file']
        }

def test_intelligent_extraction():
    """Test the intelligent Q/A extractor"""
    extractor = IntelligentQAExtractor()
    
    # Test with the problematic segment
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma.",
        "style": "therapeutic",
        "confidence": 3.0,
        "quality": 0.7,
        "source": "test",
        "file": "test.txt"
    }
    
    result = extractor.process_segment(test_segment)
    
    print("=== INTELLIGENT Q/A EXTRACTION TEST ===\n")
    print(f"**Q**: {result['input']}")
    print(f"**A**: {result['output'][:300]}...")
    print(f"**Style**: {result['style']}")

if __name__ == "__main__":
    test_intelligent_extraction()
