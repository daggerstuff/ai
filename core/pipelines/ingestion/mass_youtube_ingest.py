import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ai.core.pipelines.ingestion.youtube_processor import (
    YouTubePlaylistProcessor,
    AntiDetectionConfig,
)
from ai.core.pipelines.processing.transcript_quality_pipeline import (
    TranscriptQualityPipeline,
)

logger = logging.getLogger(__name__)


class MassYouTubeIngestor:
    """
    Mass ingestion system for curated YouTube channels (PIX-32).
    Coordinates download, transcription, and multi-pass quality checks.
    """

    def __init__(self, workspace_root: Path, limit_per_channel: int = 5):
        self.workspace_root = workspace_root
        self.limit_per_channel = limit_per_channel
        self.processor = YouTubePlaylistProcessor(
            output_dir=str(self.workspace_root / "ai/voice_data"),
            max_concurrent=1,  # Sequential for pilot to avoid rate limits
            anti_detection_config=AntiDetectionConfig(use_cookies=False),
        )
        self.quality_pipeline = TranscriptQualityPipeline()
        self.transcripts_dir = workspace_root / "ai/data/transcripts/ingested"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def load_handles(self, handles_file: Path) -> List[str]:
        """Loads curated handles from file."""
        if not handles_file.exists():
            logger.error(f"Handles file not found: {handles_file}")
            return []

        with open(handles_file, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    async def ingest_channel(self, handle: str) -> Dict[str, Any]:
        """
        Ingests a single channel by handle.
        Attempts to get the latest videos and process them.
        """
        logger.info(f"--- Ingesting Channel: {handle} ---")

        # Handle can be @handle or just handle
        if not handle.startswith("@"):
            handle = f"@{handle}"

        channel_url = f"https://www.youtube.com/{handle}/videos"

        # Use yt-dlp to get the latest video URLs
        try:
            cmd = [
                "yt-dlp",
                "--get-id",
                "--flat-playlist",
                "--playlist-end",
                str(self.limit_per_channel),
                channel_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_ids = result.stdout.strip().split("\n")
            logger.info(f"Found {len(video_ids)} videos for {handle}")
        except Exception as e:
            logger.error(f"Failed to fetch video IDs for {handle}: {e}")
            return {"handle": handle, "success": False, "error": str(e)}

        results = []
        for v_id in video_ids:
            video_url = f"https://www.youtube.com/watch?v={v_id}"
            res = await self._process_video(v_id, video_url, handle)
            results.append(res)

        return {"handle": handle, "success": True, "processed_videos": results}

    async def _process_video(
        self, video_id: str, url: str, handle: str
    ) -> Dict[str, Any]:
        """Processes a single video: download -> 3-pass quality -> save."""
        logger.info(f"Processing Video: {video_id} ({url})")

        # 1. Download Audio
        # We use a temp dir for audio processing
        temp_audio_dir = self.workspace_root / "ai/data/audio/temp" / video_id
        temp_audio_dir.mkdir(parents=True, exist_ok=True)

        download_result = await self.processor.download_playlist_audio(
            url, temp_audio_dir
        )
        if not download_result.success or not download_result.audio_files:
            return {"video_id": video_id, "status": "failed_download"}

        audio_file = download_result.audio_files[0]

        # 2. Run Multi-Pass Quality Pipeline
        pipeline_result = self.quality_pipeline.process_audio(audio_file)
        if not pipeline_result.get("success"):
            return {
                "video_id": video_id,
                "status": "failed_quality_pipeline",
                "error": pipeline_result.get("error"),
            }

        # 3. Save Final Transcript (Markdown format for RAG system)
        final_markdown = self._format_as_markdown(video_id, handle, pipeline_result)

        output_file = self.transcripts_dir / f"{video_id}.md"
        output_file.write_text(final_markdown, encoding="utf-8")

        # Cleanup audio (optional, keep if needed for training)
        # os.remove(audio_file)

        return {"video_id": video_id, "status": "success", "output": str(output_file)}

    def _format_as_markdown(
        self, video_id: str, handle: str, result: Dict[str, Any]
    ) -> str:
        """
        Formats the result as the standardized markdown expected by
        YouTubeRAGSystem.
        """
        now = datetime.now(timezone.utc).isoformat()
        return f"""# {video_id} | {handle}
**Speaker:** {handle}
**Video ID:** {video_id}
**Processed:** {now}
**Language:** en
**Duration:** 0.0

## Summary
Auto-generated transcript via multi-pass quality pipeline.

## Transcript
{result["corrected_text"]}
"""

    async def run_mass_ingest(self, handles_file: Path, pilot_count: int = 0):
        """Runs the mass ingestion for all handles."""
        handles = self.load_handles(handles_file)
        logger.info(f"Loaded {len(handles)} handles from {handles_file}")
        if pilot_count > 0:
            handles = handles[:pilot_count]
            logger.info(f"Running PILOT mode for {pilot_count} handles: {handles}")

        for handle in handles:
            try:
                await self.ingest_channel(handle)
            except Exception as e:
                logger.error(f"Fatal error ingesting {handle}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Resolve workspace root (should be repo root /home/vivi/pixelated)
    # File is at ai/pipelines/orchestrator/ingestion/mass_youtube_ingest.py
    # So parents[3] is the repo root.
    repo_root = Path(__file__).resolve().parents[4]

    handles_path = repo_root / "ai/docs/curated_youtube_handles.txt"
    ingestor = MassYouTubeIngestor(
        workspace_root=repo_root, limit_per_channel=2
    )  # Small limit for now

    # Run pilot of 3 channels
    asyncio.run(ingestor.run_mass_ingest(handles_path, pilot_count=3))
