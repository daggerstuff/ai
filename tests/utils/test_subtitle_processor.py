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

def test_format_as_markdown_empty_metadata():
    text = "This is a sentence. This is another! And a third?"
    metadata = {}

    result = SubtitleProcessor.format_as_markdown(text, metadata)

    assert "# Unknown Title" in result
    assert "**Channel:** Unknown Channel" in result
    assert "This is a sentence. This is another! And a third?" in result
