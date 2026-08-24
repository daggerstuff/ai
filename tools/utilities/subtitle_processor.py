import re

from utils.text_segmentation import split_sentences


class SubtitleProcessor:
    """Utility for cleaning and formatting YouTube VTT subtitles."""

    @staticmethod
    def clean_vtt(vtt_content: str) -> str:
        """
        Clean VTT content by removing timestamps, tags, and duplicates.
        YouTube automatic captions often repeat lines with incremental words.
        """
        # Remove header
        lines = vtt_content.split("\n")
        if lines and lines[0].startswith("WEBVTT"):
            lines = lines[1:]

        # Remove metadata lines (Kind:, Language:, etc)
        lines = [
            line
            for line in lines
            if not any(line.startswith(prefix) for prefix in ["Kind:", "Language:", "align:", "position:"])
        ]

        # Remove timestamp lines and tags
        # Pattern for 00:00:00.000 --> 00:00:00.000
        timestamp_pattern = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*")
        # Pattern for <00:00:00.000><c> etc
        tag_pattern = re.compile(r"<[^>]+>")

        current_text = []

        seen_lines = set()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if timestamp_pattern.match(line):
                continue

            # Clean tags
            cleaned_line = tag_pattern.sub("", line).strip()

            if not cleaned_line:
                continue

            # YouTube auto-subs repeat text heavily.
            # We want to keep unique sentences/segments.
            if cleaned_line in seen_lines:
                continue

            seen_lines.add(cleaned_line)
            current_text.append(cleaned_line)

        # Merge lines and remove redundant parts of sentences
        return " ".join(current_text)

        # Simple cleanup of redundant repeated segments (YouTube specific)
        # e.g. "Hello world Hello world there" -> "Hello world there"
        # This is a bit complex to do perfectly without NLP, but we can do some basics.

    @staticmethod
    def format_as_markdown(text: str, metadata: dict) -> str:
        """Format the cleaned text as a structured Markdown file."""
        title = metadata.get("title", "Unknown Title")
        channel = metadata.get("channel", "Unknown Channel")
        video_url = metadata.get("url", "")
        date = metadata.get("date", "")

        md = f"# {title}\n\n"
        md += f"**Channel:** {channel}\n"
        md += f"**Source:** {video_url}\n"
        md += f"**Date:** {date}\n\n"
        md += "## Transcript\n\n"

        # Split into paragraphs of roughly 5-7 sentences
        sentences = split_sentences(text)
        paragraphs = []
        for i in range(0, len(sentences), 6):
            paragraphs.append(" ".join(sentences[i : i + 6]))

        md += "\n\n".join(paragraphs)
        md += "\n"

        return md
