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

def test_format_as_markdown_edge_cases():
    # Empty metadata
    text = "Sentence one. Sentence two."
    metadata = {}
    result = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "# Unknown Title" in result
    assert "**Channel:** Unknown Channel" in result
    assert "**Source:** \n" in result
    assert "**Date:** \n" in result
    assert "Sentence one. Sentence two." in result

    # Paragraph splitting
    long_text = "S1. S2. S3. S4. S5. S6. S7. S8. S9. S10."
    result = SubtitleProcessor.format_as_markdown(long_text, metadata)
    paragraphs = result.split("\n\n")
    # Header + meta, Transcript header, paragraph 1, paragraph 2, trailing newline
    assert "S1. S2. S3. S4. S5. S6." in result
    assert "S7. S8. S9. S10." in result
