import pytest
from utils.subtitle_processor import SubtitleProcessor

class TestSubtitleProcessor:
    def test_clean_vtt_basic(self):
        vtt = (
            "WEBVTT\n"
            "00:00:00.000 --> 00:00:02.000 align:start position:0%\n"
            "<c>Hello</c> <c>world</c>\n\n"
            "00:00:01.000 --> 00:00:03.000\nHello world\n\n"
            "00:00:02.000 --> 00:00:04.000\nThis is a test\n"
        )
        assert SubtitleProcessor.clean_vtt(vtt) == "Hello world This is a test"

    def test_clean_vtt_edge_cases(self):
        # Empty and missing header
        assert SubtitleProcessor.clean_vtt("") == ""
        assert SubtitleProcessor.clean_vtt("00:00:00.000 --> 00:00:02.000\nText") == "Text"
        assert SubtitleProcessor.clean_vtt("00:00:00.000 --> 00:00:02.000\n<c></c>") == ""

    def test_format_as_markdown(self):
        text = "T1. T2. T3. T4. T5. T6. T7."
        metadata = {"title": "A", "channel": "B", "url": "C", "date": "D"}
        markdown = SubtitleProcessor.format_as_markdown(text, metadata)
        assert "# A" in markdown
        assert "**Channel:** B" in markdown

        paragraphs = [p for p in markdown.split("\n\n") if p and not p.startswith("#") and not p.startswith("**")]
        assert len(paragraphs) == 2
        assert "T1." in paragraphs[0]
        assert "T7." in paragraphs[1]
