from utils.subtitle_processor import SubtitleProcessor


def test_clean_vtt_edge_cases():
    # Empty and edge case content
    assert SubtitleProcessor.clean_vtt("") == ""
    assert SubtitleProcessor.clean_vtt("WEBVTT\n\n") == ""

    # Valid VTT with headers, timestamps, tags, and duplicates
    vtt = (
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:00.000 --> 00:00:02.000\n<c>Hello</c> world\n\n"
        "00:00:02.000 --> 00:00:04.000\nHello world\n\n"
        "00:00:04.000 --> 00:00:06.000\nHello world there"
    )
    assert SubtitleProcessor.clean_vtt(vtt) == "Hello world Hello world there"


def test_format_as_markdown_missing_metadata():
    expected = (
        "# Unknown Title\n\n**Channel:** Unknown Channel\n"
        "**Source:** \n**Date:** \n\n## Transcript\n\nSome text.\n"
    )
    assert SubtitleProcessor.format_as_markdown("Some text.", {}) == expected
