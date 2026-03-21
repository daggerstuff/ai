import pytest
from utils.subtitle_processor import SubtitleProcessor

def test_clean_vtt_removes_timestamps():
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hello world

00:00:01.000 --> 00:00:02.000
Hello world
This is a test"""

    cleaned = SubtitleProcessor.clean_vtt(vtt_content)
    assert cleaned == "Hello world This is a test"

def test_clean_vtt_empty_string():
    cleaned = SubtitleProcessor.clean_vtt("")
    assert cleaned == ""
