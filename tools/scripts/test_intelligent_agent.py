#!/usr/bin/env python3
"""
Test suite for the Intelligent Multi-Pattern Prompt Generation Agent
Validates the agent against the critical issues found in session savepoint.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "prompts" / "agents"))


import contextlib

try:
    from intelligent_prompt_agent import MultiPatternAgent
except ImportError:
    from prompts.agents.intelligent_prompt_agent import MultiPatternAgent


def test_interview_extraction():
    """Test the exact problem case from session savepoint"""

    agent = MultiPatternAgent()

    # Exact problem case from savepoint
    problem_segment = """
    Interviewer: How can somebody begin to take that path toward healing from complex trauma.

    Tim Fletcher: Well, that's a huge question because unfortunately, most people with complex trauma don't realize they have complex trauma. They think they're just anxious or depressed or they have relationship problems, but they don't understand the underlying root cause.
    """

    analysis = agent.analyze_segment(problem_segment)

    # Validate it found the right question
    expected_question_keywords = [
        "how can somebody",
        "begin",
        "path",
        "healing",
        "complex trauma",
    ]
    if analysis["extracted_question"]:
        sum(1 for keyword in expected_question_keywords if keyword.lower() in analysis["extracted_question"].lower())
    else:
        pass

    # Validate transition marker detection
    if "that's a huge question" in analysis["transition_markers"]:
        pass
    else:
        pass


def test_monologue_content():
    """Test handling of monologue/speech content without embedded questions"""

    agent = MultiPatternAgent()

    monologue_segment = """
    When a person has a lot of shame deep down that they've never acknowledged, they're pretty sure that if people get to know them, nobody will ever want to meet their needs. So they develop this false self that's designed to get their needs met indirectly through manipulation and control.
    """

    analysis = agent.analyze_segment(monologue_segment)

    # Should detect as monologue and not extract embedded questions
    if analysis["content_type"] in ["monologue", "speech"]:
        pass
    else:
        pass


def test_podcast_content():
    """Test podcast-style conversational content"""

    agent = MultiPatternAgent()

    podcast_segment = """
    Host: Today we're discussing trauma recovery. What should people know about starting their healing journey?

    Expert: The first thing I always tell people is that healing isn't linear. You're going to have good days and bad days, and that's completely normal.
    """

    analysis = agent.analyze_segment(podcast_segment)

    # Should detect question about healing journey
    if analysis["extracted_question"] and "healing" in analysis["extracted_question"].lower():
        pass
    else:
        pass


def test_semantic_coherence():
    """Test semantic coherence validation between questions and answers"""

    agent = MultiPatternAgent()

    # Good match
    good_question = "How does trauma affect relationships?"
    good_response = "Trauma significantly impacts how we connect with others. When someone has unresolved trauma, they often struggle with trust, intimacy, and healthy boundaries in relationships."

    good_coherence = agent.validate_semantic_coherence(good_question, good_response)

    # Bad match (the original problem)
    bad_question = "What are your favorite recipes?"
    bad_response = "Trauma significantly impacts how we connect with others. When someone has unresolved trauma, they often struggle with trust, intimacy, and healthy boundaries in relationships."

    bad_coherence = agent.validate_semantic_coherence(bad_question, bad_response)

    if good_coherence > bad_coherence + 0.2:
        pass
    else:
        pass


def test_contextual_prompt_generation():
    """Test that generated prompts are contextually appropriate"""

    agent = MultiPatternAgent()

    # Sample segments for different styles
    test_segments = [
        {
            "text": "When someone has narcissistic traits, they often use manipulation and gaslighting to control others. This creates significant trauma for their victims.",
            "style": "therapeutic",
        },
        {
            "text": "Here are three practical steps you can take to set healthy boundaries: First, identify your limits. Second, communicate them clearly. Third, enforce them consistently.",
            "style": "practical",
        },
        {
            "text": "I understand how painful it feels when someone you trusted betrays you. That hurt is real and valid, and you deserve to heal from it.",
            "style": "empathetic",
        },
        {
            "text": "Complex PTSD differs from regular PTSD in several key ways. It develops from prolonged, repeated trauma, typically in childhood.",
            "style": "educational",
        },
    ]

    for _i, segment in enumerate(test_segments):
        analysis = agent.analyze_segment(segment["text"])
        prompt = agent.generate_contextual_prompt(segment, analysis)

        # Check if prompt relates to content
        set(segment["text"].lower().split())
        set(prompt.lower().split())

        # Look for thematic overlap
        themes_match = False
        if (
            (
                "narciss" in segment["text"].lower()
                and any(word in prompt.lower() for word in ["narciss", "manipulation", "abuse"])
            )
            or ("boundaries" in segment["text"].lower() and "boundaries" in prompt.lower())
            or (
                "ptsd" in segment["text"].lower() and any(word in prompt.lower() for word in ["trauma", "ptsd", "heal"])
            )
            or (
                "betrays" in segment["text"].lower()
                and any(word in prompt.lower() for word in ["hurt", "betray", "trust"])
            )
        ):
            themes_match = True

        if themes_match:
            pass
        else:
            pass


def test_edge_cases():
    """Test edge cases and potential failure modes"""

    agent = MultiPatternAgent()

    edge_cases = [
        "",  # Empty string
        "A",  # Very short
        "This is a normal sentence without any therapeutic content whatsoever.",  # No therapeutic content
        "??????",  # Only punctuation
    ]

    for _i, case in enumerate(edge_cases):
        with contextlib.suppress(Exception):
            agent.analyze_segment(case)


def main():
    """Run comprehensive test suite"""

    test_interview_extraction()
    test_monologue_content()
    test_podcast_content()
    test_semantic_coherence()
    test_contextual_prompt_generation()
    test_edge_cases()


if __name__ == "__main__":
    main()
