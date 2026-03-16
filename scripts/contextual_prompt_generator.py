#!/usr/bin/env python3
"""
Contextual Prompt Generator - Creates questions that the segment actually answers
"""

import json
import re
from typing import Dict, List
from pathlib import Path

class ContextualPromptGenerator:
    def __init__(self):
        pass
    
    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract the main concepts the text actually discusses"""
        # Look for specific topics being explained
        concepts = []
        
        # Direct explanations
        if re.search(r'what.*is|this is|here\'s what', text, re.IGNORECASE):
            concepts.append("explanation")
        
        # Process descriptions  
        if re.search(r'when.*happens|the process|how.*works', text, re.IGNORECASE):
            concepts.append("process")
            
        # Advice/recommendations
        if re.search(r'you should|i recommend|try|do this', text, re.IGNORECASE):
            concepts.append("advice")
            
        # Personal experience/examples
        if re.search(r'i\'ve seen|in my experience|for example', text, re.IGNORECASE):
            concepts.append("experience")
            
        return concepts
    
    def analyze_content_structure(self, text: str) -> Dict:
        """Analyze what the text is actually saying"""
        analysis = {
            "main_topic": None,
            "content_type": None,
            "key_points": [],
            "context": None
        }
        
        # Find the main subject being discussed
        sentences = re.split(r'[.!?]+', text)
        first_sentence = sentences[0].strip() if sentences else ""
        
        # Determine what's being explained/discussed
        if "narcissist" in text.lower():
            analysis["main_topic"] = "narcissistic behavior"
        elif any(word in text.lower() for word in ["trauma", "ptsd", "complex trauma"]):
            analysis["main_topic"] = "trauma"
        elif any(word in text.lower() for word in ["shame", "guilt"]):
            analysis["main_topic"] = "shame"
        elif any(word in text.lower() for word in ["relationship", "attachment"]):
            analysis["main_topic"] = "relationships"
        elif any(word in text.lower() for word in ["heal", "recovery", "therapy"]):
            analysis["main_topic"] = "healing"
        else:
            # Extract from first sentence
            words = first_sentence.lower().split()
            key_nouns = [w for w in words if len(w) > 4 and w not in ['that', 'this', 'they', 'them', 'when', 'where']]
            analysis["main_topic"] = key_nouns[0] if key_nouns else "personal growth"
        
        # Determine content type
        if re.search(r'because|the reason|why.*is', text, re.IGNORECASE):
            analysis["content_type"] = "explanation"
        elif re.search(r'you should|try|do this|i recommend', text, re.IGNORECASE):
            analysis["content_type"] = "advice"
        elif re.search(r'when.*happens|if you|in situations', text, re.IGNORECASE):
            analysis["content_type"] = "scenario"
        else:
            analysis["content_type"] = "insight"
            
        return analysis
    
    def generate_contextual_question(self, segment: Dict) -> str:
        """Generate a question that the segment actually answers"""
        text = segment['text']
        style = segment['style']
        
        analysis = self.analyze_content_structure(text)
        topic = analysis["main_topic"]
        content_type = analysis["content_type"]
        
        # Create questions based on what the text actually contains
        if content_type == "explanation":
            if style == "therapeutic":
                return f"Can you help me understand {topic} and why it happens?"
            elif style == "educational":
                return f"What causes {topic} and how does it work?"
            elif style == "empathetic":
                return f"I'm struggling to understand {topic}. Can you help me make sense of it?"
            else:  # practical
                return f"I need to understand {topic} better. What should I know?"
                
        elif content_type == "advice":
            if style == "therapeutic":
                return f"I'm dealing with {topic}. What guidance can you offer?"
            elif style == "practical":
                return f"What should I do about {topic}?"
            elif style == "empathetic":
                return f"I'm struggling with {topic} and need help. What would you suggest?"
            else:  # educational
                return f"What's the best approach for handling {topic}?"
                
        elif content_type == "scenario":
            if style == "therapeutic":
                return f"What happens when someone experiences {topic}?"
            elif style == "educational":
                return f"How does {topic} typically unfold?"
            elif style == "empathetic":
                return f"I'm going through {topic}. What can I expect?"
            else:  # practical
                return f"When {topic} occurs, what should I be aware of?"
                
        else:  # insight
            if style == "therapeutic":
                return f"What insights can you share about {topic}?"
            elif style == "educational":
                return f"What should I understand about {topic}?"
            elif style == "empathetic":
                return f"I need perspective on {topic}. Can you help?"
            else:  # practical
                return f"What's important to know about {topic}?"
    
    def create_training_pair(self, segment: Dict) -> Dict:
        """Convert segment to contextually appropriate training pair"""
        question = self.generate_contextual_question(segment)
        
        return {
            "input": question,
            "output": segment['text'],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file']
        }

def test_contextual_generation():
    """Test the contextual generator with sample segments"""
    generator = ContextualPromptGenerator()
    
    # Test segments
    test_segments = [
        {
            "text": "Narcissists become master manipulators because they have deep shame they've never acknowledged. They're pretty sure that if people get to know them, nobody will ever want to meet their needs. So the only way they can get their needs met is not to express them, but to learn how to manipulate them.",
            "style": "therapeutic",
            "confidence": 1.0,
            "quality": 0.8,
            "source": "test",
            "file": "test.txt"
        },
        {
            "text": "When you're in crisis and afraid for your life, you have to fall back on that training, on that muscle memory. You have to incorporate some level of structure and ritual. Use sensory and auditory stimuli, temperature regulation like warm baths, cold packs.",
            "style": "practical", 
            "confidence": 1.0,
            "quality": 0.7,
            "source": "test",
            "file": "test.txt"
        }
    ]
    
    print("=== CONTEXTUAL PROMPT GENERATION TEST ===\n")
    
    for i, segment in enumerate(test_segments, 1):
        pair = generator.create_training_pair(segment)
        print(f"**Test {i}**")
        print(f"**Q**: {pair['input']}")
        print(f"**A**: {pair['output'][:200]}...")
        print(f"**Style**: {pair['style']}")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    test_contextual_generation()
