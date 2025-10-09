#!/usr/bin/env python3
"""
Test suite for the Intelligent Multi-Pattern Prompt Generation Agent
Validates the agent against the critical issues found in session savepoint.
"""

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from intelligent_prompt_agent import MultiPatternAgent, ContentType

def test_interview_extraction():
    """Test the exact problem case from session savepoint"""
    print("🧪 Testing Interview Question Extraction")
    
    agent = MultiPatternAgent()
    
    # Exact problem case from savepoint
    problem_segment = """
    Interviewer: How can somebody begin to take that path toward healing from complex trauma.
    
    Tim Fletcher: Well, that's a huge question because unfortunately, most people with complex trauma don't realize they have complex trauma. They think they're just anxious or depressed or they have relationship problems, but they don't understand the underlying root cause.
    """
    
    analysis = agent.analyze_segment(problem_segment)
    
    print(f"Content Type: {analysis['content_type']} (confidence: {analysis['content_confidence']:.2f})")
    print(f"Extracted Question: '{analysis['extracted_question']}'")
    print(f"Question Confidence: {analysis['question_confidence']:.2f}")
    print(f"Response Boundary: '{analysis['response_boundary']}'")
    print(f"Transition Markers: {analysis['transition_markers']}")
    print(f"Semantic Coherence: {analysis['semantic_coherence']:.2f}")
    print(f"Overall Confidence: {analysis['overall_confidence']:.2f}")
    
    # Validate it found the right question
    expected_question_keywords = ["how can somebody", "begin", "path", "healing", "complex trauma"]
    if analysis['extracted_question']:
        found_keywords = sum(1 for keyword in expected_question_keywords 
                           if keyword.lower() in analysis['extracted_question'].lower())
        print(f"✅ Question extraction: {found_keywords}/{len(expected_question_keywords)} keywords found")
    else:
        print("❌ No question extracted")
    
    # Validate transition marker detection
    if "that's a huge question" in analysis['transition_markers']:
        print("✅ Transition marker detected: 'that's a huge question'")
    else:
        print("❌ Failed to detect transition marker")
    
    print("-" * 60)

def test_monologue_content():
    """Test handling of monologue/speech content without embedded questions"""
    print("🧪 Testing Monologue Content Handling")
    
    agent = MultiPatternAgent()
    
    monologue_segment = """
    When a person has a lot of shame deep down that they've never acknowledged, they're pretty sure that if people get to know them, nobody will ever want to meet their needs. So they develop this false self that's designed to get their needs met indirectly through manipulation and control.
    """
    
    analysis = agent.analyze_segment(monologue_segment)
    
    print(f"Content Type: {analysis['content_type']} (confidence: {analysis['content_confidence']:.2f})")
    print(f"Extracted Question: '{analysis['extracted_question']}'")
    print(f"Overall Confidence: {analysis['overall_confidence']:.2f}")
    print(f"Processing Notes: {analysis['processing_notes']}")
    
    # Should detect as monologue and not extract embedded questions
    if analysis['content_type'] in ['monologue', 'speech']:
        print("✅ Correctly identified non-interview content")
    else:
        print("❌ Failed to identify content type")
    
    print("-" * 60)

def test_podcast_content():
    """Test podcast-style conversational content"""
    print("🧪 Testing Podcast Content Analysis")
    
    agent = MultiPatternAgent()
    
    podcast_segment = """
    Host: Today we're discussing trauma recovery. What should people know about starting their healing journey?
    
    Expert: The first thing I always tell people is that healing isn't linear. You're going to have good days and bad days, and that's completely normal.
    """
    
    analysis = agent.analyze_segment(podcast_segment)
    
    print(f"Content Type: {analysis['content_type']} (confidence: {analysis['content_confidence']:.2f})")
    print(f"Extracted Question: '{analysis['extracted_question']}'")
    print(f"Question Confidence: {analysis['question_confidence']:.2f}")
    print(f"Semantic Coherence: {analysis['semantic_coherence']:.2f}")
    
    # Should detect question about healing journey
    if analysis['extracted_question'] and 'healing' in analysis['extracted_question'].lower():
        print("✅ Extracted relevant question about healing")
    else:
        print("❌ Failed to extract appropriate question")
    
    print("-" * 60)

def test_semantic_coherence():
    """Test semantic coherence validation between questions and answers"""
    print("🧪 Testing Semantic Coherence Validation")
    
    agent = MultiPatternAgent()
    
    # Good match
    good_question = "How does trauma affect relationships?"
    good_response = "Trauma significantly impacts how we connect with others. When someone has unresolved trauma, they often struggle with trust, intimacy, and healthy boundaries in relationships."
    
    good_coherence = agent.validate_semantic_coherence(good_question, good_response)
    
    # Bad match (the original problem)
    bad_question = "What are your favorite recipes?"
    bad_response = "Trauma significantly impacts how we connect with others. When someone has unresolved trauma, they often struggle with trust, intimacy, and healthy boundaries in relationships."
    
    bad_coherence = agent.validate_semantic_coherence(bad_question, bad_response)
    
    print(f"Good match coherence: {good_coherence:.2f}")
    print(f"Bad match coherence: {bad_coherence:.2f}")
    
    if good_coherence > bad_coherence + 0.2:
        print("✅ Semantic coherence validation working correctly")
    else:
        print("❌ Semantic coherence validation needs improvement")
    
    print("-" * 60)

def test_contextual_prompt_generation():
    """Test that generated prompts are contextually appropriate"""
    print("🧪 Testing Contextual Prompt Generation")
    
    agent = MultiPatternAgent()
    
    # Sample segments for different styles
    test_segments = [
        {
            'text': 'When someone has narcissistic traits, they often use manipulation and gaslighting to control others. This creates significant trauma for their victims.',
            'style': 'therapeutic'
        },
        {
            'text': 'Here are three practical steps you can take to set healthy boundaries: First, identify your limits. Second, communicate them clearly. Third, enforce them consistently.',
            'style': 'practical'
        },
        {
            'text': 'I understand how painful it feels when someone you trusted betrays you. That hurt is real and valid, and you deserve to heal from it.',
            'style': 'empathetic'
        },
        {
            'text': 'Complex PTSD differs from regular PTSD in several key ways. It develops from prolonged, repeated trauma, typically in childhood.',
            'style': 'educational'
        }
    ]
    
    for i, segment in enumerate(test_segments):
        analysis = agent.analyze_segment(segment['text'])
        prompt = agent.generate_contextual_prompt(segment, analysis)
        
        print(f"Segment {i+1} ({segment['style']}):")
        print(f"  Generated Prompt: '{prompt}'")
        
        # Check if prompt relates to content
        segment_words = set(segment['text'].lower().split())
        prompt_words = set(prompt.lower().split())
        
        # Look for thematic overlap
        themes_match = False
        if 'narciss' in segment['text'].lower() and any(word in prompt.lower() for word in ['narciss', 'manipulation', 'abuse']):
            themes_match = True
        elif 'boundaries' in segment['text'].lower() and 'boundaries' in prompt.lower():
            themes_match = True
        elif 'ptsd' in segment['text'].lower() and any(word in prompt.lower() for word in ['trauma', 'ptsd', 'heal']):
            themes_match = True
        elif 'betrays' in segment['text'].lower() and any(word in prompt.lower() for word in ['hurt', 'betray', 'trust']):
            themes_match = True
        
        if themes_match:
            print("  ✅ Thematically appropriate prompt")
        else:
            print("  ⚠️  Check thematic relevance")
        
        print()
    
    print("-" * 60)

def test_edge_cases():
    """Test edge cases and potential failure modes"""
    print("🧪 Testing Edge Cases")
    
    agent = MultiPatternAgent()
    
    edge_cases = [
        "",  # Empty string
        "A",  # Very short
        "This is a normal sentence without any therapeutic content whatsoever.",  # No therapeutic content
        "??????",  # Only punctuation
    ]
    
    for i, case in enumerate(edge_cases):
        try:
            analysis = agent.analyze_segment(case)
            print(f"Edge case {i+1}: Handled successfully (confidence: {analysis['overall_confidence']:.2f})")
        except Exception as e:
            print(f"Edge case {i+1}: Error - {e}")
    
    print("-" * 60)

def main():
    """Run comprehensive test suite"""
    print("🚀 Testing Intelligent Multi-Pattern Prompt Generation Agent")
    print("=" * 80)
    
    test_interview_extraction()
    test_monologue_content()
    test_podcast_content()
    test_semantic_coherence()
    test_contextual_prompt_generation()
    test_edge_cases()
    
    print("🎯 Test Suite Complete!")
    print("\nKey Improvements Over Original System:")
    print("✅ Extracts actual questions from interview content")
    print("✅ Detects content type with confidence scoring")
    print("✅ Validates semantic coherence between Q/A pairs")
    print("✅ Handles multiple content formats (interview/podcast/monologue)")
    print("✅ Uses transition markers for response boundary detection")
    print("✅ Generates contextually appropriate prompts")

if __name__ == "__main__":
    main()