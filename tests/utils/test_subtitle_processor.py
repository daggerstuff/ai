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

def test_clean_vtt_empty_string():
    assert SubtitleProcessor.clean_vtt("") == ""
    assert SubtitleProcessor.clean_vtt("   \n  ") == ""

def test_clean_vtt_no_header():
    vtt_content = """00:00:00.000 --> 00:00:02.500 align:start position:0%
<c.colorE5E5E5>Hello without header</c>
"""
    assert SubtitleProcessor.clean_vtt(vtt_content) == "Hello without header"

def test_clean_vtt_duplicate_lines():
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:02.000
Repeated line

00:00:02.000 --> 00:00:03.000
Repeated line

00:00:03.000 --> 00:00:04.000
New line
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "Repeated line New line"

def test_clean_vtt_tags_and_formatting():
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:02.000
<00:00:01.500><c> Some text </c> with <b>HTML</b> tags
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "Some text  with HTML tags"

def test_format_as_markdown_with_metadata():
    metadata = {
        "title": "Test Video",
        "channel": "Test Channel",
        "url": "https://youtube.com/test",
        "date": "2023-10-26"
    }
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6. Sentence 7."
    md = SubtitleProcessor.format_as_markdown(text, metadata)

    assert "# Test Video" in md
    assert "**Channel:** Test Channel" in md
    assert "**Source:** https://youtube.com/test" in md
    assert "**Date:** 2023-10-26" in md
    assert "## Transcript" in md

    # 7 sentences should be split into two paragraphs (6 and 1)
    paragraphs = md.split("\n\n")
    assert any("Sentence 1." in p and "Sentence 6." in p for p in paragraphs)
    assert any("Sentence 7." in p and "Sentence 1." not in p for p in paragraphs)

def test_format_as_markdown_missing_metadata():
    text = "Some transcript text."
    md = SubtitleProcessor.format_as_markdown(text, {})

    assert "# Unknown Title" in md
    assert "**Channel:** Unknown Channel" in md
    assert "**Source:** \n" in md
    assert "**Date:** \n" in md
