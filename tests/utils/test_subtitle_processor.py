from utils.subtitle_processor import SubtitleProcessor


def test_clean_vtt_empty_string():
    """Test clean_vtt with an empty string."""
    assert SubtitleProcessor.clean_vtt("") == ""

def test_clean_vtt_only_metadata():
    """Test clean_vtt with only header and metadata."""
    vtt = "WEBVTT\nKind: captions\nLanguage: en\n"
    assert SubtitleProcessor.clean_vtt(vtt) == ""
