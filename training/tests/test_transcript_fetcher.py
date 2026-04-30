"""Tests for the YouTube transcript fetcher."""

from __future__ import annotations

import json
import textwrap
from unittest.mock import patch

import pytest

from training.transcript_fetcher import (
    _clean_subtitle,
    _slugify,
    _read_urls,
    build_parser,
)


# ---------------------------------------------------------------------------
# _clean_subtitle
# ---------------------------------------------------------------------------

class TestCleanSubtitle:
    def test_strips_vtt_timestamps(self):
        raw = textwrap.dedent("""\
            WEBVTT

            00:00:01.000 --> 00:00:05.000
            Hello this is the first line

            00:00:06.000 --> 00:00:10.000
            And the second line
        """)
        result = _clean_subtitle(raw)
        assert "Hello this is the first line" in result
        assert "And the second line" in result
        assert "-->" not in result
        assert "00:00" not in result

    def test_strips_short_timestamps(self):
        raw = "0:15.000 --> 1:30.000\nSome text here"
        result = _clean_subtitle(raw)
        assert "-->" not in result
        assert "Some text here" in result

    def test_strips_sequence_numbers(self):
        raw = "1\n00:00:01.000 --> 00:00:05.000\nActual content"
        result = _clean_subtitle(raw)
        lines = [l for l in result.split("\n") if l.strip()]
        assert lines == ["Actual content"]

    def test_strips_html_tags(self):
        raw = '<b>Bold text</b> and <i>italic</i> and <font color="white">styled</font>'
        result = _clean_subtitle(raw)
        assert "<b>" not in result
        assert "</b>" not in result
        assert "Bold text" in result
        assert "italic" in result
        assert "styled" in result

    def test_deduplicates_consecutive_lines(self):
        raw = "Same line\nSame line\nDifferent line"
        result = _clean_subtitle(raw)
        assert result.count("Same line") == 1
        assert "Different line" in result

    def test_removes_music_markers(self):
        raw = "♪ La la la ♪\nReal content here\n♫ Instrumental ♫"
        result = _clean_subtitle(raw)
        assert "♪" not in result
        assert "♫" not in result
        assert "Real content here" in result

    def test_removes_bracket_annotations(self):
        raw = "[music playing]\n[applause]\nActual dialogue\n[laughter]"
        result = _clean_subtitle(raw)
        assert "[music" not in result
        assert "[applause" not in result
        assert "Actual dialogue" in result

    def test_paragraph_separation(self):
        raw = "Line one\n\nLine two\n\nLine three"
        result = _clean_subtitle(raw)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result

    def test_empty_input(self):
        assert _clean_subtitle("") == ""
        assert _clean_subtitle("   \n\n   ") == ""

    def test_only_timestamps_produces_empty(self):
        raw = "1\n00:00:01.000 --> 00:00:05.000\n\n2\n00:00:06.000 --> 00:00:10.000"
        result = _clean_subtitle(raw)
        assert result == ""

    def test_mixed_content(self):
        raw = textwrap.dedent("""\
            WEBVTT
            Kind: captions
            Language: en

            1
            00:00:01.000 --> 00:00:05.000
            Welcome to the channel

            2
            00:00:06.000 --> 00:00:10.000
            <b>Today we're discussing trauma</b>

            [music]

            3
            00:00:11.000 --> 00:00:15.000
            Today we're discussing trauma

            4
            00:00:16.000 --> 00:00:20.000
            And how it affects relationships
        """)
        result = _clean_subtitle(raw)
        assert "Welcome to the channel" in result
        assert "Today we're discussing trauma" in result
        assert "how it affects relationships" in result
        assert "WEBVTT" not in result
        assert "[music]" not in result


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello_world"

    def test_special_characters(self):
        assert _slugify("What's going on? (Part 1/3)") == "whats_going_on_part_13"

    def test_truncation(self):
        long = "a" * 200
        assert len(_slugify(long)) == 120

    def test_multiple_hyphens_spaces(self):
        assert _slugify("foo   bar---baz") == "foo_bar_baz"

    def test_empty(self):
        assert _slugify("") == ""
        assert _slugify("   ") == ""


# ---------------------------------------------------------------------------
# _read_urls
# ---------------------------------------------------------------------------

class TestReadUrls:
    def test_deduplicates(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://youtube.com/watch?v=abc\nhttps://youtube.com/watch?v=abc\n")
        urls = _read_urls(url_file)
        assert urls == ["https://youtube.com/watch?v=abc"]

    def test_skips_comments_and_blanks(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text("# comment\n\nhttps://youtube.com/watch?v=1\n  \nhttps://youtube.com/watch?v=2\n")
        urls = _read_urls(url_file)
        assert len(urls) == 2

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            _read_urls(tmp_path / "nonexistent.txt")

    def test_preserves_order(self, tmp_path):
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://a.com\nhttps://b.com\nhttps://a.com\nhttps://c.com\n")
        urls = _read_urls(url_file)
        assert urls == ["https://a.com", "https://b.com", "https://c.com"]


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_required_url_file(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--url_file", "urls.txt"])
        assert args.output_dir == "training/data/transcripts"
        assert args.lang == "en"
        assert args.rate_limit == 2.0
        assert args.max_urls == 0

    def test_custom_values(self):
        parser = build_parser()
        args = parser.parse_args([
            "--url_file", "my_urls.txt",
            "--output_dir", "data/subs",
            "--lang", "de",
            "--rate_limit", "5",
            "--max_urls", "10",
        ])
        assert args.url_file == "my_urls.txt"
        assert args.output_dir == "data/subs"
        assert args.lang == "de"
        assert args.rate_limit == 5.0
        assert args.max_urls == 10


# ---------------------------------------------------------------------------
# Integration: run_fetch with mocked yt-dlp
# ---------------------------------------------------------------------------

class TestRunFetch:
    @staticmethod
    def _mock_metadata(title="Test Video", channel="TestChannel", vid="abc123"):
        return json.dumps({
            "title": title,
            "channel": channel,
            "id": vid,
            "duration": 300,
        })

    @patch("training.transcript_fetcher._fetch_subtitles")
    @patch("training.transcript_fetcher._fetch_video_metadata")
    def test_fetch_single_video(self, mock_meta, mock_subs, tmp_path):
        mock_meta.return_value = {
            "channel": "Tim Fletcher",
            "title": "Complex Trauma Part 1",
            "id": "abc123",
            "duration": 300,
        }
        sub_file = tmp_path / "subs" / "temp.en.vtt"
        sub_file.parent.mkdir(parents=True, exist_ok=True)
        sub_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello world this is a long enough subtitle content for testing purposes\n")
        mock_subs.return_value = sub_file

        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://youtube.com/watch?v=abc123\n")
        output_dir = tmp_path / "output"

        parser = build_parser()
        args = parser.parse_args([
            "--url_file", str(url_file),
            "--output_dir", str(output_dir),
            "--rate_limit", "0",
        ])

        from training.transcript_fetcher import run_fetch
        run_fetch(args)

        channel_dir = output_dir / "Tim Fletcher"
        assert channel_dir.is_dir()
        txt_files = list(channel_dir.glob("*.txt"))
        assert len(txt_files) == 1
        content = txt_files[0].read_text()
        assert "Hello world this is a long enough subtitle content" in content
        assert "-->" not in content

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["stats"]["fetched"] == 1

    @patch("training.transcript_fetcher._fetch_subtitles")
    @patch("training.transcript_fetcher._fetch_video_metadata")
    def test_skip_no_subtitles(self, mock_meta, mock_subs, tmp_path):
        mock_meta.return_value = {
            "channel": "SomeChannel",
            "title": "No Subs",
            "id": "xyz",
            "duration": 60,
        }
        mock_subs.return_value = None

        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://youtube.com/watch?v=xyz\n")
        output_dir = tmp_path / "output"

        parser = build_parser()
        args = parser.parse_args([
            "--url_file", str(url_file),
            "--output_dir", str(output_dir),
            "--rate_limit", "0",
        ])

        from training.transcript_fetcher import run_fetch
        run_fetch(args)

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["stats"]["no_subtitles"] == 1
        assert manifest["stats"]["fetched"] == 0

    @patch("training.transcript_fetcher._fetch_subtitles")
    @patch("training.transcript_fetcher._fetch_video_metadata")
    def test_skip_too_short_subtitle(self, mock_meta, mock_subs, tmp_path):
        mock_meta.return_value = {
            "channel": "ShortChannel",
            "title": "Tiny Video",
            "id": "short1",
            "duration": 10,
        }
        sub_file = tmp_path / "subs" / "temp.en.vtt"
        sub_file.parent.mkdir(parents=True, exist_ok=True)
        sub_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHi\n")
        mock_subs.return_value = sub_file

        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://youtube.com/watch?v=short1\n")
        output_dir = tmp_path / "output"

        parser = build_parser()
        args = parser.parse_args([
            "--url_file", str(url_file),
            "--output_dir", str(output_dir),
            "--rate_limit", "0",
        ])

        from training.transcript_fetcher import run_fetch
        run_fetch(args)

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["stats"]["no_subtitles"] == 1

    @patch("training.transcript_fetcher._fetch_subtitles")
    @patch("training.transcript_fetcher._fetch_video_metadata")
    def test_max_urls_limit(self, mock_meta, mock_subs, tmp_path):
        mock_meta.return_value = {
            "channel": "Ch",
            "title": "Vid",
            "id": "id1",
            "duration": 100,
        }
        sub_file = tmp_path / "subs" / "temp.en.vtt"
        sub_file.parent.mkdir(parents=True, exist_ok=True)
        sub_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nContent here for the test\n")
        mock_subs.return_value = sub_file

        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://a.com/1\nhttps://a.com/2\nhttps://a.com/3\n")
        output_dir = tmp_path / "output"

        parser = build_parser()
        args = parser.parse_args([
            "--url_file", str(url_file),
            "--output_dir", str(output_dir),
            "--rate_limit", "0",
            "--max_urls", "1",
        ])

        from training.transcript_fetcher import run_fetch
        run_fetch(args)

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["stats"]["total_urls"] == 1

    @patch("training.transcript_fetcher._fetch_subtitles")
    @patch("training.transcript_fetcher._fetch_video_metadata")
    def test_multiple_channels(self, mock_meta, mock_subs, tmp_path):
        meta_responses = [
            {"channel": "ChannelA", "title": "Video 1", "id": "a1", "duration": 100},
            {"channel": "ChannelB", "title": "Video 2", "id": "b2", "duration": 200},
        ]
        mock_meta.side_effect = meta_responses

        sub_file = tmp_path / "subs" / "temp.en.vtt"
        sub_file.parent.mkdir(parents=True, exist_ok=True)
        sub_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nSome therapeutic content for testing that is long enough to pass the minimum threshold check\n")
        mock_subs.return_value = sub_file

        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://a.com/1\nhttps://b.com/2\n")
        output_dir = tmp_path / "output"

        parser = build_parser()
        args = parser.parse_args([
            "--url_file", str(url_file),
            "--output_dir", str(output_dir),
            "--rate_limit", "0",
        ])

        from training.transcript_fetcher import run_fetch
        run_fetch(args)

        assert (output_dir / "ChannelA").is_dir()
        assert (output_dir / "ChannelB").is_dir()
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert "ChannelA" in manifest["stats"]["channels"]
        assert "ChannelB" in manifest["stats"]["channels"]
