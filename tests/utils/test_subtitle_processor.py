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


def test_format_as_markdown_with_metadata():
    text = "This is a test transcript."
    metadata = {
        "title": "Test Video",
        "channel": "Test Channel",
        "url": "https://youtube.com/test",
        "date": "2023-01-01",
    }
    result = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "# Test Video" in result
    assert "**Channel:** Test Channel" in result
    assert "**Source:** https://youtube.com/test" in result
    assert "**Date:** 2023-01-01" in result
    assert "## Transcript" in result
    assert "This is a test transcript." in result


def test_format_as_markdown_missing_metadata():
    text = "This is a transcript without metadata."
    metadata = {}
    result = SubtitleProcessor.format_as_markdown(text, metadata)
    assert "# Unknown Title" in result
    assert "**Channel:** Unknown Channel" in result
    assert "**Source:** \n" in result
    assert "**Date:** \n" in result
    assert "This is a transcript without metadata." in result


def test_format_as_markdown_paragraph_splitting():
    # 7 sentences should be split into 2 paragraphs (6 in the first, 1 in the second)
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6. Sentence 7."
    metadata = {"title": "Paragraph Test"}
    result = SubtitleProcessor.format_as_markdown(text, metadata)

    # Split by empty lines to find paragraphs, skipping header info
    parts = result.split("\n\n")
    # Transcript header is "## Transcript", the text follows it
    transcript_index = parts.index("## Transcript")
    paragraphs = parts[transcript_index + 1 :]

    # Exclude empty parts or the final newline if it gets split
    paragraphs = [p for p in paragraphs if p.strip()]

    assert len(paragraphs) == 2
    assert (
        "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6."
        in paragraphs[0]
    )
    assert "Sentence 7." in paragraphs[1]
