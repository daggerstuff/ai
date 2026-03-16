#!/usr/bin/env python3
"""
Smart Q/A Agent - Intelligent content analysis without external dependencies
Uses sophisticated NLP techniques to understand content structure
"""

import json
import re
from typing import Dict, List, Tuple, Optional

class SmartQAAgent:
    def __init__(self):
        pass
    
    def deep_content_analysis(self, text: str) -> Dict:
        """Perform deep analysis of content structure and meaning"""
        
        analysis = {
            "content_type": None,
            "dialogue_structure": None,
            "main_topic": None,
            "question_embedded": None,
            "response_portion": None,
            "confidence": 0.0
        }
        
        # Clean and prepare text
        text = text.strip()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Detect dialogue structure
        dialogue_analysis = self.analyze_dialogue_structure(text, sentences)
        analysis.update(dialogue_analysis)
        
        # Extract semantic meaning
        semantic_analysis = self.analyze_semantic_content(text, sentences)
        analysis.update(semantic_analysis)
        
        return analysis
    
    def analyze_dialogue_structure(self, text: str, sentences: List[str]) -> Dict:
        """Analyze if this contains dialogue and separate speakers"""
        
        # Look for interview/conversation patterns
        interview_indicators = [
            "i've heard you talk about",
            "you mentioned",
            "can you tell us",
            "what would you say",
            "how would you",
            "you've said"
        ]
        
        has_interview_setup = any(indicator in text.lower() for indicator in interview_indicators)
        
        # Look for embedded questions
        question_sentences = [s for s in sentences if '?' in s or 
                            any(starter in s.lower() for starter in ['how can', 'what should', 'why do', 'when does'])]
        
        # Analyze structure
        if has_interview_setup and question_sentences:
            # This looks like interview content
            
            # Find the transition point
            for i, sentence in enumerate(sentences):
                if '?' in sentence:
                    # Found a question - everything after might be the response
                    question = sentence
                    
                    # Look for response indicators after the question
                    response_start = i + 1
                    for j in range(i + 1, len(sentences)):
                        response_indicators = ['that\'s', 'well', 'so', 'the thing is', 'what happens', 'unfortunately', 'look']
                        if any(indicator in sentences[j].lower() for indicator in response_indicators):
                            response_start = j
                            break
                    
                    if response_start < len(sentences):
                        response = '. '.join(sentences[response_start:])
                        
                        return {
                            "content_type": "interview_dialogue",
                            "dialogue_structure": "question_response_identified",
                            "question_embedded": question,
                            "response_portion": response,
                            "confidence": 0.8
                        }
        
        # Check for monologue/teaching structure
        teaching_indicators = ['let me explain', 'what happens is', 'the key thing', 'first', 'second', 'the reason']
        if any(indicator in text.lower() for indicator in teaching_indicators):
            return {
                "content_type": "teaching_monologue",
                "dialogue_structure": "single_speaker",
                "confidence": 0.7
            }
        
        return {
            "content_type": "unclear",
            "dialogue_structure": "unknown",
            "confidence": 0.3
        }
    
    def analyze_semantic_content(self, text: str, sentences: List[str]) -> Dict:
        """Analyze the semantic meaning and main topic"""
        
        text_lower = text.lower()
        
        # Topic detection with context
        topic_patterns = {
            "trauma_therapy": ["trauma", "therap", "ptsd", "treatment"],
            "narcissistic_behavior": ["narcissist", "manipulat", "abuse", "toxic"],
            "shame_guilt": ["shame", "guilt", "self-worth", "identity"],
            "healing_recovery": ["heal", "recover", "growth", "journey"],
            "relationships": ["relationship", "attachment", "trust", "intimacy"],
            "mental_health": ["depression", "anxiety", "mental health", "emotional"]
        }
        
        topic_scores = {}
        for topic, keywords in topic_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        main_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "general_therapeutic"
        
        return {
            "main_topic": main_topic,
            "topic_confidence": max(topic_scores.values()) / len(topic_patterns[main_topic]) if topic_scores else 0.1
        }
    
    def generate_intelligent_question(self, analysis: Dict, style: str) -> str:
        """Generate contextually appropriate question based on deep analysis"""
        
        # If we found an embedded question, clean it up
        if analysis.get("question_embedded"):
            question = analysis["question_embedded"]
            # Clean up the question
            question = re.sub(r'^[^A-Z]*', '', question).strip()
            if not question.endswith('?'):
                question += '?'
            return question
        
        # Generate based on topic and content type
        main_topic = analysis.get("main_topic", "general_therapeutic")
        content_type = analysis.get("content_type", "unclear")
        
        topic_questions = {
            "trauma_therapy": {
                "therapeutic": "How can someone find proper trauma therapy and what should they look for?",
                "educational": "What should people understand about trauma therapy?",
                "empathetic": "I'm struggling to find help for trauma. What guidance can you offer?",
                "practical": "What are the practical steps to finding trauma-informed therapy?"
            },
            "narcissistic_behavior": {
                "therapeutic": "Can you help me understand narcissistic behavior and its impact?",
                "educational": "What are the key characteristics of narcissistic behavior?",
                "empathetic": "I'm dealing with someone who seems narcissistic. Can you help me understand this?",
                "practical": "How should someone handle narcissistic behavior?"
            },
            "shame_guilt": {
                "therapeutic": "How does shame affect people and what can be done about it?",
                "educational": "Can you explain the difference between shame and guilt?",
                "empathetic": "I struggle with shame. Can you help me understand what's happening?",
                "practical": "What are effective ways to deal with shame?"
            },
            "healing_recovery": {
                "therapeutic": "What does the healing process look like for trauma survivors?",
                "educational": "Can you explain how healing from trauma works?",
                "empathetic": "I'm on a healing journey and need guidance. What should I know?",
                "practical": "What are the practical steps in trauma recovery?"
            }
        }
        
        if main_topic in topic_questions and style in topic_questions[main_topic]:
            return topic_questions[main_topic][style]
        
        # Fallback questions
        fallback_questions = {
            "therapeutic": "Can you share your therapeutic perspective on this?",
            "educational": "Can you explain this concept?",
            "empathetic": "I need help understanding this. Can you guide me?",
            "practical": "What should I know about this situation?"
        }
        
        return fallback_questions.get(style, "Can you help me understand this?")
    
    def process_segment(self, segment: Dict) -> Dict:
        """Process segment with intelligent analysis"""
        
        # Perform deep content analysis
        analysis = self.deep_content_analysis(segment['text'])
        
        # Generate appropriate question
        question = self.generate_intelligent_question(analysis, segment['style'])
        
        # Determine response text
        if analysis.get("response_portion"):
            response = analysis["response_portion"]
        else:
            response = segment['text']
        
        return {
            "input": question,
            "output": response,
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file'],
            "smart_analysis": analysis
        }

def test_smart_agent():
    """Test the smart Q/A agent"""
    agent = SmartQAAgent()
    
    test_segment = {
        "text": "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training. I'm a physician. And the average physician never hears any of the stuff I just talked about. The average psychiatrist doesn't get any training in trauma.",
        "style": "therapeutic",
        "confidence": 3.0,
        "quality": 0.7,
        "source": "interview",
        "file": "test.txt"
    }
    
    result = agent.process_segment(test_segment)
    
    print("=== SMART AI AGENT TEST ===\n")
    print(f"**Content Type**: {result['smart_analysis']['content_type']}")
    print(f"**Dialogue Structure**: {result['smart_analysis']['dialogue_structure']}")
    print(f"**Main Topic**: {result['smart_analysis']['main_topic']}")
    print(f"**Confidence**: {result['smart_analysis']['confidence']}")
    print(f"**Generated Q**: {result['input']}")
    print(f"**Response A**: {result['output'][:300]}...")
    
    if result['smart_analysis'].get('question_embedded'):
        print(f"**Embedded Q Found**: {result['smart_analysis']['question_embedded']}")

if __name__ == "__main__":
    test_smart_agent()
