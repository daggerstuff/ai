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


def test_format_as_markdown_success():
    text = "Hello there. How are you? I am fine."
    metadata = {"title": "Test Title", "channel": "Test Channel", "url": "http://test", "date": "2023-01-01"}
    result = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "# Test Title" in result
    assert "**Channel:** Test Channel" in result
    assert "**Source:** http://test" in result
    assert "**Date:** 2023-01-01" in result
    assert "Hello there. How are you? I am fine." in result


def test_format_as_markdown_missing_metadata():
    text = "Hello there. How are you? I am fine."
    metadata = {}
    result = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "# Unknown Title" in result
    assert "**Channel:** Unknown Channel" in result
    assert "**Source:** \n" in result
    assert "**Date:** \n" in result
    assert "Hello there. How are you? I am fine." in result


def test_clean_vtt_removes_duplicates():
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%
<c.colorE5E5E5>This is repeated</c>

00:00:02.500 --> 00:00:05.000 align:start position:0%
<c.colorE5E5E5>This is repeated</c>

00:00:05.000 --> 00:00:07.500 align:start position:0%
<c.colorE5E5E5>This is new</c>
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "This is repeated This is new"


def test_clean_vtt_empty_content():
    assert SubtitleProcessor.clean_vtt("") == ""


def test_clean_vtt_empty_cleaned_line():
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%
<c.colorE5E5E5></c>
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == ""


def test_clean_vtt_empty_and_whitespace():
    assert SubtitleProcessor.clean_vtt("") == ""
    assert SubtitleProcessor.clean_vtt("   \n   \n") == ""


def test_clean_vtt_duplicate_lines():
    vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500 align:start position:0%
Hello world

00:00:02.500 --> 00:00:05.000 align:start position:0%
Hello world
"""
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "Hello world"


def test_clean_vtt_no_header():
    vtt_content = "00:00:00.000 --> 00:00:02.500\n<c>Just text</c>"
    result = SubtitleProcessor.clean_vtt(vtt_content)
    assert result == "Just text"
