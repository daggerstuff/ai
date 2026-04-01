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
    text = "Sentence one. Sentence two! Sentence three? Sentence four."
    metadata = {
        "title": "Test Video",
        "channel": "Test Channel",
        "url": "https://example.com/video",
        "date": "2023-01-01"
    }

    md = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "Test Video" in md
    assert "Test Channel" in md
    assert "https://example.com/video" in md
    assert "2023-01-01" in md
    assert "Sentence one. Sentence two! Sentence three? Sentence four." in md

def test_format_as_markdown_missing_metadata():
    text = "Just some text."
    metadata = {}

    md = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "Unknown Title" in md
    assert "Unknown Channel" in md
    assert "**Source:** \n" in md
    assert "**Date:** \n" in md
