import asyncio
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import sys
import re

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

from ai.core.utils.subtitle_processor import SubtitleProcessor
from ai.core.pipelines.logger import setup_logger


class MassSubtitleIngestor:
    """Ingests pre-downloaded YouTube transcripts into standard dataset Markdown formats."""

    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.source_dir = workspace_root / "ai/youtube_transcriptions/transcripts"
        self.output_dir = workspace_root / "ai/data/transcripts/ingested"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("mass_subtitle_ingest")

    async def ingest_local_file(self, txt_file: Path, channel_name: str):
        """Read a local transcript text file and save as Markdown."""
        self.logger.info(f"Ingesting local file: {txt_file.name}")

        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            video_title = txt_file.stem

            metadata = {
                "title": video_title,
                "channel": channel_name,
                "url": f"local://{channel_name}/{video_title}",
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            # If it's a VTT file masquerading as .txt, clean it. Otherwise, use raw text.
            if "WEBVTT" in content or "-->" in content:
                cleaned_text = SubtitleProcessor.clean_vtt(content)
            else:
                # Basic cleanup for raw text: remove multiple newlines/spaces if desired
                # Or just treat the entire block as cleaned_text. We'll join lines for formatting.
                # SubtitleProcessor.format_as_markdown expects paragraphs!
                # Let's ensure it's a continuous string without breaking sentences unnaturally.
                cleaned_text = " ".join([line.strip() for line in content.split("\n") if line.strip()])

            markdown_content = SubtitleProcessor.format_as_markdown(
                cleaned_text, metadata
            )

            # Save to ingested directory
            safe_title = (
                "".join(
                    [c for c in video_title if c.isalnum() or c in (" ", "-", "_")]
                )
                .strip()
                .replace(" ", "_")
            )
            output_file_name = f"{channel_name.replace(' ', '')}_{safe_title}.md"
            output_file = self.output_dir / output_file_name

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            self.logger.info(f"Successfully ingested {output_file.name}")

        except Exception as e:
            self.logger.error(f"Error ingesting file {txt_file.name}: {e}")

    async def run_mass_ingest(self, pilot_count: int = 0):
        if not self.source_dir.exists():
            self.logger.error(f"Source directory not found: {self.source_dir}")
            return

        channel_dirs = [d for d in self.source_dir.iterdir() if d.is_dir()]

        if pilot_count > 0:
            channel_dirs = channel_dirs[:pilot_count]
            self.logger.info(f"Pilot mode: processing first {pilot_count} channels out of {len(channel_dirs)}")

        for channel_dir in channel_dirs:
            channel_name = channel_dir.name
            self.logger.info(f"--- Processing Channel: {channel_name} ---")

            txt_files = list(channel_dir.glob("*.txt"))
            self.logger.info(f"Found {len(txt_files)} transcript files for {channel_name}")

            for txt_file in txt_files:
                await self.ingest_local_file(txt_file, channel_name)
                # Brief yield for asyncio event loop
                await asyncio.sleep(0.01)

if __name__ == "__main__":
    # If a number is passed as an argument, use it as pilot_count
    pilot = 0
    if len(sys.argv) > 1:
        try:
            pilot = int(sys.argv[1])
        except ValueError:
            print("Warning: pilot count must be an integer, executing full run instead.")

    # Workspace root assuming script is in pixelated/ai/pipelines/orchestrator/ingestion/
    ws_root = Path(__file__).resolve().parents[4]

    async def run():
        ingestor = MassSubtitleIngestor(ws_root)
        await ingestor.run_mass_ingest(pilot)

    asyncio.run(run())
