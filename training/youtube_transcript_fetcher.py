#!/usr/bin/env python3
"""
YouTube transcript fetcher using youtube-transcript-api for therapeutic AI training.
Bypasses yt-dlp bot detection issues by using the official transcript API.
"""

import argparse
import json
import logging
import re
import sys
from datetime import UTC
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Try to import youtube-transcript-api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        CouldNotRetrieveTranscript,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("youtube_transcript_fetcher")


def read_youtube_playlists(playlists_file: Path) -> list[str]:
    """Read YouTube URLs from playlists file and deduplicate."""
    if not playlists_file.exists():
        logger.error(f"YouTube playlists file not found: {playlists_file}")
        return []

    urls = []
    with open(playlists_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    logger.info(f"Read {len(urls)} URLs, deduplicated to {len(unique_urls)} unique URLs")
    return unique_urls


def extract_video_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    # Handle youtu.be format
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?", maxsplit=1)[0]

    # Handle youtube.com/watch format
    if "youtube.com/watch" in url:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            return query_params["v"][0]

    # Handle youtube.com/embed format
    if "youtube.com/embed/" in url:
        return url.split("youtube.com/embed/")[1].split("?", maxsplit=1)[0]

    # Handle youtube.com/v format
    if "youtube.com/v/" in url:
        return url.split("youtube.com/v/")[1].split("?", maxsplit=1)[0]

    # If we can't extract, log warning and return None
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None


def sanitize_filename(text: str) -> str:
    """Sanitize text for use as filename."""
    # Remove or replace problematic characters
    text = re.sub(r'[<>:\"/\\\\|?*\\x00-\\x1f]', "_", text)
    # Remove leading/trailing spaces and dots
    text = text.strip(" .")
    # Limit length
    if len(text) > 200:
        text = text[:200]
    return text or "unknown"


def fetch_transcript(video_id: str, language_priority: list[str] | None = None) -> str | None:
    """
    Fetch transcript for a single video using youtube-transcript-api.

    Args:
        video_id: YouTube video ID
        language_priority: List of language codes to try in order

    Returns:
        Transcript text if successful, None otherwise
    """
    if language_priority is None:
        language_priority = ["en", "en-US", "en-GB", "de", "de-DE"]

    try:
        # Create an instance of the API
        ytt_api = YouTubeTranscriptApi()

        # Method 1: Try the simple get_transcript approach first
        try:
            transcript_list = ytt_api.get_transcript(video_id, languages=language_priority)
            # Convert to plain text
            text = " ".join([entry["text"] for entry in transcript_list])
            logger.info(f"Successfully fetched transcript for {video_id} via get_transcript")
            return text
        except Exception as e1:
            logger.debug(f"get_transcript failed for {video_id}: {e1}")
            # Fall back to list_transcript approach

        # Method 2: Use list_transcripts for more control
        transcript_list = ytt_api.list(video_id)

        # Try each language in priority order
        for lang in language_priority:
            try:
                transcript = transcript_list.find_transcript([lang])
                transcript_data = transcript.fetch()

                # Convert to plain text
                text = " ".join([entry["text"] for entry in transcript_data])
                logger.info(f"Successfully fetched transcript for {video_id} ({lang}) via {transcript.type}")
                return text
            except Exception as e2:
                logger.debug(f"Failed to get {lang} transcript for {video_id}: {e2}")
                continue

        # If priority languages failed, try to get any available transcript
        try:
            # Get the first available transcript
            available_transcripts = list(transcript_list)
            if available_transcripts:
                transcript = available_transcripts[0]
                transcript_data = transcript.fetch()
                text = " ".join([entry["text"] for entry in transcript_data])
                logger.info(f"Successfully fetched transcript for {video_id} ({transcript.language_code}) via {transcript.type}")
                return text
        except Exception as e3:
            logger.debug(f"No transcripts found for {video_id}: {e3}")

    except TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video {video_id}")
        return None
    except NoTranscriptFound:
        logger.warning(f"No transcript found for video {video_id}")
        return None
    except VideoUnavailable:
        logger.warning(f"Video {video_id} is unavailable")
        return None
    except CouldNotRetrieveTranscript:
        logger.warning(f"Could not retrieve transcript for video {video_id}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching transcript for video {video_id}: {e}")
        return None

    return None


def save_transcript(video_id: str, text: str, output_path: Path) -> bool:
    """Save transcript text to file."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean up the transcript text
        # Remove extra whitespace, normalize spaces
        text = re.sub(r"\s+", " ", text.strip())

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.debug(f"Saved transcript for {video_id} to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving transcript for {video_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts using youtube-transcript-api for therapeutic AI training."
    )
    parser.add_argument(
        "--playlists-file",
        type=str,
        default="ai/docs/youtube_playlists.txt",
        help="Path to YouTube playlists file containing URLs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/data/transcripts",
        help="Base directory for storing transcripts (will create channel subdirs)"
    )
    parser.add_argument(
        "--channel-name",
        type=str,
        default="YouTube_Mix",
        help="Channel name to use for organizing transcripts"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Preferred language for transcripts (comma-separated list: en,de,etc.)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum number of videos to process (0 for all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have transcript files"
    )

    args = parser.parse_args()

    # Check if API is available
    if not YOUTUBE_API_AVAILABLE:
        logger.error("youtube-transcript-api is not available. Please install it.")
        return 1

    # Parse arguments
    playlists_file = Path(args.playlists_file)
    output_dir = Path(args.output_dir) / args.channel_name
    channel_name = args.channel_name
    languages = [lang.strip() for lang in args.language.split(",")]
    max_videos = args.max_videos
    skip_existing = args.skip_existing

    logger.info("Starting YouTube transcript fetcher (API method)")
    logger.info(f"Playlists file: {playlists_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Channel name: {channel_name}")
    logger.info(f"Languages: {languages}")

    # Read and deduplicate URLs
    urls = read_youtube_playlists(playlists_file)
    if not urls:
        logger.error("No URLs to process")
        return 1

    # Apply max videos limit if specified
    if max_videos > 0:
        urls = urls[:max_videos]
        logger.info(f"Limited to first {max_videos} videos")

    # Process each URL
    successful = 0
    failed = 0
    skipped = 0

    for i, url in enumerate(urls, 1):
        logger.info(f"Processing [{i}/{len(urls)}]: {url}")

        try:
            video_id = extract_video_id(url)
            if not video_id:
                failed += 1
                continue

            # Check if file already exists
            output_file = output_dir / f"{video_id}.txt"
            if skip_existing and output_file.exists():
                logger.debug(f"Skipping existing transcript for {video_id}")
                skipped += 1
                continue

            # Fetch transcript
            transcript_text = fetch_transcript(video_id, languages)
            if transcript_text:
                if save_transcript(video_id, transcript_text, output_file):
                    successful += 1
                else:
                    failed += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            failed += 1

    # Create manifest
    from datetime import datetime

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "channel": channel_name,
        "total_urls_processed": len(urls),
        "successful": successful,
        "failed": failed,
        "skipped_existing": skipped,
        "output_directory": str(output_dir),
        "languages_attempted": languages,
        "method": "youtube-transcript-api"
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Processing complete:")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped (existing): {skipped}")
    logger.info(f"  Manifest saved to: {manifest_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
