#!/usr/bin/env python3
"""
AI Q/A Agent - Uses actual AI to intelligently analyze content and generate appropriate Q/A pairs
"""

import json
import openai
import os
from typing import Dict, Optional
from pathlib import Path

class AIQAAgent:
    def __init__(self):
        # Set up OpenAI client (you'll need to set OPENAI_API_KEY)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def analyze_content(self, text: str, metadata: Dict) -> Dict:
        """Use AI to analyze content and determine optimal Q/A approach"""
        
        prompt = f"""
You are an expert at analyzing therapeutic and educational content to create appropriate question-answer pairs for training data.

Analyze this content segment and determine:
1. What type of content this is (interview dialogue, monologue, teaching, story, etc.)
2. What the main topic/focus is
3. What question would naturally lead to this response
4. If this contains multiple speakers, separate them appropriately

Content to analyze:
"{text}"

Metadata: {metadata}

Provide your analysis in this JSON format:
{{
    "content_type": "interview_dialogue|monologue|teaching|story|explanation",
    "main_topic": "brief description of what this is about",
    "speakers_detected": "single|multiple",
    "dialogue_structure": "description of how speakers are organized",
    "optimal_question": "the question that would naturally lead to this response",
    "response_text": "the actual response portion (if different from full text)",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your analysis"
}}

Be very careful to ensure the question and response actually match and make logical sense together.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert content analyzer specializing in therapeutic and educational materials."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis = json.loads(analysis_text)
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return self.create_fallback_analysis(text, metadata)
                
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return self.create_fallback_analysis(text, metadata)
    
    def create_fallback_analysis(self, text: str, metadata: Dict) -> Dict:
        """Fallback analysis when AI call fails"""
        return {
            "content_type": "unknown",
            "main_topic": "therapeutic content",
            "speakers_detected": "single",
            "dialogue_structure": "unclear",
            "optimal_question": "Can you share your perspective on this?",
            "response_text": text,
            "confidence": 0.3,
            "reasoning": "AI analysis unavailable, using fallback"
        }
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment using AI analysis"""
        
        # Get AI analysis
        analysis = self.analyze_content(segment['text'], {
            'style': segment['style'],
            'source': segment['source'],
            'file': segment['file']
        })
        
        # Create training pair based on AI analysis
        return {
            "input": analysis['optimal_question'],
            "output": analysis['response_text'],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "ai_analysis": analysis
        }

# Alternative: Local AI using transformers (if OpenAI not available)
class LocalAIQAAgent:
    def __init__(self):
        try:
            from transformers import pipeline
            self.analyzer = pipeline("text-generation", model="microsoft/DialoGPT-medium")
        except ImportError:
            print("Transformers not available, using rule-based fallback")
            self.analyzer = None
    
    def analyze_content_local(self, text: str, metadata: Dict) -> Dict:
        """Local AI analysis using transformers"""
        
        if not self.analyzer:
            return self.create_simple_analysis(text, metadata)
        
        # Use local model for analysis
        prompt = f"Analyze this content and create an appropriate question: {text[:500]}"
        
        try:
            result = self.analyzer(prompt, max_length=100, num_return_sequences=1)
            # Process result to extract question
            generated = result[0]['generated_text']
            
            # Extract question from generated text (simplified)
            if '?' in generated:
                question = generated.split('?')[0] + '?'
            else:
                question = self.create_contextual_question(text, metadata['style'])
            
            return {
                "content_type": "analyzed_local",
                "main_topic": "therapeutic content",
                "optimal_question": question,
                "response_text": text,
                "confidence": 0.7,
                "reasoning": "Local AI analysis"
            }
            
        except Exception as e:
            print(f"Local AI failed: {e}")
            return self.create_simple_analysis(text, metadata)
    
    def create_simple_analysis(self, text: str, metadata: Dict) -> Dict:
        """Simple rule-based analysis"""
        
        # Basic content analysis
        text_lower = text.lower()
        
        if "trauma" in text_lower and "therap" in text_lower:
            question = "What should someone know about finding trauma therapy?"
        elif "narcissist" in text_lower:
            question = "Can you explain narcissistic behavior?"
        elif "shame" in text_lower:
            question = "How does shame affect people?"
        elif "heal" in text_lower:
            question = "What does healing look like?"
        else:
            style = metadata.get('style', 'therapeutic')
            if style == "therapeutic":
                question = "Can you share your therapeutic insight on this?"
            elif style == "educational":
                question = "Can you explain this concept?"
            else:
                question = "What should I understand about this?"
        
        return {
            "content_type": "rule_based",
            "main_topic": "therapeutic content",
            "optimal_question": question,
            "response_text": text,
            "confidence": 0.5,
            "reasoning": "Rule-based analysis"
        }
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment using local analysis"""
        
        analysis = self.analyze_content_local(segment['text'], {
            'style': segment['style'],
            'source': segment['source'],
            'file': segment['file']
        })
        
        return {
            "input": analysis['optimal_question'],
            "output": analysis['response_text'],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "ai_analysis": analysis
        }

def test_ai_agent():
    """Test the AI agent"""
    
    # Try OpenAI first, fallback to local
    if os.getenv('OPENAI_API_KEY'):
        agent = AIQAAgent()
        print("Using OpenAI GPT-4 for analysis")
    else:
        agent = LocalAIQAAgent()
        print("Using local AI analysis")
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma.",
        "style": "therapeutic",
        "confidence": 3.0,
        "quality": 0.7,
        "source": "interview",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("\n=== AI AGENT TEST ===\n")
    print(f"**Analysis Method**: {result['ai_analysis']['reasoning']}")
    print(f"**Content Type**: {result['ai_analysis']['content_type']}")
    print(f"**Confidence**: {result['ai_analysis']['confidence']}")
    print(f"**Generated Q**: {result['input']}")
    print(f"**Response A**: {result['output'][:200]}...")

if __name__ == "__main__":
    test_ai_agent()
