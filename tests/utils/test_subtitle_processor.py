import pytest
from utils.subtitle_processor import SubtitleProcessor

def test_clean_vtt_removes_metadata_and_timestamps():
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%
<c.colorE5E5E5>Hello</c>

00:00:02.500 --> 00:00:05.000 align:start position:0%
<c.colorE5E5E5>Hello world</c>
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "Hello Hello world"

def test_format_as_markdown():
    # Test 1: Full metadata and exactly 6 sentences (1 paragraph)
    text1 = "Sentence one. Sentence two! Sentence three? Sentence four. Sentence five! Sentence six."
    metadata_full = {
        "title": "Test Video",
        "channel": "Test Channel",
        "url": "https://youtube.com/test",
        "date": "2023-01-01"
    }

    md1 = SubtitleProcessor.format_as_markdown(text1, metadata_full)
    assert "# Test Video\n\n" in md1
    assert "**Channel:** Test Channel\n" in md1
    assert "**Source:** https://youtube.com/test\n" in md1
    assert "**Date:** 2023-01-01\n\n" in md1
    assert "## Transcript\n\n" in md1
    # Check paragraph chunking (all 6 sentences in one paragraph)
    assert "Sentence one. Sentence two! Sentence three? Sentence four. Sentence five! Sentence six.\n" in md1

    # Test 2: Missing metadata (fallbacks) and 7 sentences (2 paragraphs)
    text2 = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six. Sentence seven."
    metadata_empty = {}

    md2 = SubtitleProcessor.format_as_markdown(text2, metadata_empty)
    assert "# Unknown Title\n\n" in md2
    assert "**Channel:** Unknown Channel\n" in md2
    assert "**Source:** \n" in md2
    assert "**Date:** \n\n" in md2

    # Verify paragraph chunking (6 sentences in first paragraph, 1 in second)
    expected_p1 = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    expected_p2 = "Sentence seven."

    assert f"## Transcript\n\n{expected_p1}\n\n{expected_p2}\n" in md2
