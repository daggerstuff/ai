#!/usr/bin/env python3
"""
Test question extraction with fixed patterns
"""

import re


def test_question_extraction():
    """Test question extraction directly"""

    text = "And I guess to take this one step further, I've heard you talk about that one of the main problems with society today is that a lot of people who have mental health struggles and they're struggling with situations this where they feel stuck and they can't get out of their own way at times. They feel people are against them. They're dealing with trauma. And then they might go see somebody who's not trained in trauma. They don't have the experience. How can somebody begin to take that path and make sure that they're finding somebody that is trained in that and then also that they're able to self-regulate themselves. When needed. , that's a huge question because unfortunately, look, I've been through medical training."

    # Test the specific pattern for our expected question
    pattern = r"(How can [^?]*\?)"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    for match in matches:
        match.group(1).strip()

        # Find what comes after
        after_question = text[match.end() :].strip()

        # Look for response markers
        response_markers = [
            r"[,\s]*that\'s a (huge|big|great) question",
            r"[,\s]*unfortunately[,\s]*",
            r"[,\s]*look[,\s]*",
        ]

        for marker in response_markers:
            marker_match = re.search(marker, after_question, re.IGNORECASE)
            if marker_match:
                response_start = marker_match.start()
                response = after_question[response_start:].strip()
                response = re.sub(r"^[,\s]*", "", response)
                break


if __name__ == "__main__":
    test_question_extraction()
