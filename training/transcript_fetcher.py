#!/usr/bin/env python3
"""
YouTube transcript fetcher using yt-dlp for therapeutic AI training.

Downloads YouTube transcripts/subtitles and organizes them for the youtube_ingestion.py pipeline.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("transcript_fetcher")


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


def extract_video_id(url: str) -> str:
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

    # If we can't extract, return the URL itself (will likely fail but let yt-dlp handle it)
    return url


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


def fetch_transcript(video_id: str, output_path: Path, language_priority: list[str] | None = None, cookies_from_browser: str | None = None) -> bool:
    """
    Fetch transcript for a single video using yt-dlp.

    Args:
        video_id: YouTube video ID
        output_path: Path to save transcript (.txt file)
        language_priority: List of language codes to try in order
        cookies_from_browser: Browser to load cookies from (firefox, chrome, etc.)

    Returns:
        True if transcript was successfully fetched and saved
    """
    if language_priority is None:
        language_priority = ["en", "en-US", "en-GB", "de", "de-DE"]

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try each language in priority order
    for lang in language_priority:
        try:
            cmd = [
                "yt-dlp",
                "--write-auto-sub",  # Write automatic subtitles
                "--sub-lang", lang,  # Language
                "--skip-download",   # Don't download video
                "--output", str(output_path.with_suffix("")),  # Output template (without extension)
                # Add headers to appear more like a regular browser
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "--referer", "https://www.youtube.com/",
                "--sleep-interval", "1",  # Sleep between requests to avoid rate limiting
                "--max-sleep-interval", "5",
            ]

            # Add cookies if specified
            if cookies_from_browser:
                cmd.extend(["--cookies-from-browser", cookies_from_browser])
                logger.debug(f"Using cookies from {cookies_from_browser}")

            cmd.append(f"https://www.youtube.com/watch?v={video_id}")

            logger.debug(f"Running yt-dlp for video {video_id}, lang {lang}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout per video
            )

            # Log stderr for debugging if needed
            if result.stderr and "WARNING:" in result.stderr:
                logger.debug(f"yt-dlp warnings for {video_id}: {result.stderr.strip()}")

            # Check if subtitle file was created
            subtitle_file = output_path.with_suffix(f".{lang}.vtt")
            if subtitle_file.exists():
                # Convert VTT to plain text
                vtt_to_txt(subtitle_file, output_path)
                subtitle_file.unlink()  # Remove VTT file
                logger.info(f"Successfully fetched transcript for {video_id} ({lang})")
                return True

            # Also check for manual subtitles
            manual_subtitle = output_path.with_suffix(f".{lang}.vtt")
            if manual_subtitle.exists():
                vtt_to_txt(manual_subtitle, output_path)
                manual_subtitle.unlink()
                logger.info(f"Successfully fetched manual transcript for {video_id} ({lang})")
                return True

        except subprocess.TimeoutExpired:
            logger.warning(f"yt-dlp timeout for video {video_id}, lang {lang}")
            continue
        except Exception as e:
            logger.warning(f"yt-dlp error for video {video_id}, lang {lang}: {e}")
            continue

    logger.warning(f"Failed to fetch transcript for video {video_id} in all attempted languages")
    return False


def vtt_to_txt(vtt_path: Path, txt_path: Path) -> None:
    """Convert VTT subtitle file to plain text."""
    try:
        with open(vtt_path, encoding="utf-8") as f:
            content = f.read()

        # Remove VTT headers and timestamps
        lines = content.split("\n")
        text_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines, headers, and timestamp lines
            if not line:
                continue
            if line.startswith("WEBVTT"):
                continue
            if line.startswith("NOTE"):
                continue
            if re.match(r"^\d+$", line):  # Line numbers
                continue
            if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", line):  # Timestamps
                continue
            # Remove HTML tags
            line = re.sub(r"<[^>]+>", "", line)
            # Remove duplicate spaces
            line = re.sub(r"\s+", " ", line)
            if line:
                text_lines.append(line)

        # Join lines and clean up
        text = " ".join(text_lines)
        # Remove duplicate consecutive lines (common in subtitles)
        lines = text.split(". ")
        cleaned_lines = []
        prev_line = None
        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line

        # Write cleaned text
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(". ".join(cleaned_lines))

    except Exception as e:
        logger.error(f"Error converting VTT to TXT: {e}")
        # Create empty file to avoid missing file errors
        txt_path.write_text("", encoding="utf-8")


def _clean_subtitle(raw: str) -> str:
    """Clean raw VTT subtitle text: strip headers, timestamps, HTML, music markers, bracket annotations, and deduplicate lines."""
    if not raw or not raw.strip():
        return ""
    lines = raw.split("\n")
    cleaned: list[str] = []
    prev_line: str | None = None
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
            continue
        if re.match(r"^\d{1,2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}\.\d{3}", line):
            continue
        if re.match(r"^\d{1,2}:\d{2}\.\d{3}\s*-->\s*\d{1,2}:\d{2}\.\d{3}", line):
            continue
        if re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)
        line = line.replace("♪", "").replace("♫", "").strip()
        line = re.sub(r"\[[^\]]*\]", "", line).strip()
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if line == prev_line:
            continue
        cleaned.append(line)
        prev_line = line
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _read_urls(url_file: Path) -> list[str]:
    """Read URLs from file, deduplicate, skip comments and blanks. Exits on missing file."""
    if not url_file.exists():
        logger.error("URL file not found: %s", url_file)
        sys.exit(1)

    urls: list[str] = []
    with open(url_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _slugify(text: str, max_len: int = 120) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "_", text)
    text = text.strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch YouTube transcripts for therapeutic AI training.")
    parser.add_argument("--url_file", required=True, help="File containing YouTube URLs (one per line)")
    parser.add_argument("--output_dir", default="training/data/transcripts", help="Output directory for transcripts")
    parser.add_argument("--lang", default="en", help="Language code (e.g. en, de)")
    parser.add_argument("--rate_limit", type=float, default=2.0, help="Minimum seconds between requests")
    parser.add_argument("--max_urls", type=int, default=0, help="Maximum URLs to process (0 = all)")
    return parser


def _fetch_video_metadata(url: str) -> dict:
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", url],
            capture_output=True, text=True, timeout=30,
        )
        result.check_returncode()
        return json.loads(result.stdout)
    except Exception as e:
        logger.error("Failed to fetch metadata for %s: %s", url, e)
        return {}


def _fetch_subtitles(video_id: str, lang: str) -> Path | None:
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    out_template = tmp / "%(id)s.%(ext)s"
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", lang,
                "--skip-download",
                "--output", str(out_template),
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "--sleep-interval", "1",
                "--max-sleep-interval", "2",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=30,
        )
        vtt_files = list(tmp.glob(f"{video_id}*.vtt"))
        if vtt_files:
            return vtt_files[0]
        return None
    except Exception:
        return None


MIN_CONTENT_LENGTH = 30


def run_fetch(args: argparse.Namespace) -> None:
    url_file = Path(args.url_file)
    output_dir = Path(args.output_dir)
    lang = args.lang
    rate_limit = args.rate_limit
    max_urls = args.max_urls

    urls = _read_urls(url_file)
    if max_urls > 0:
        urls = urls[:max_urls]

    stats: dict = {
        "total_urls": len(urls),
        "fetched": 0,
        "no_subtitles": 0,
        "channels": {},
    }

    for url in urls:
        meta = _fetch_video_metadata(url)
        if not meta:
            continue

        video_id = meta.get("id", "")
        channel = meta.get("channel", "Unknown")
        title = meta.get("title", "")
        meta.get("duration", 0)

        sub_path = _fetch_subtitles(video_id, lang)

        if sub_path is None:
            stats["no_subtitles"] += 1
            continue

        raw = sub_path.read_text(encoding="utf-8")
        cleaned = _clean_subtitle(raw)

        if len(cleaned) < MIN_CONTENT_LENGTH:
            stats["no_subtitles"] += 1
            continue

        channel_dir = output_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _slugify(title) if title else video_id
        out_file = channel_dir / f"{safe_name}.txt"
        out_file.write_text(cleaned, encoding="utf-8")

        stats["fetched"] += 1
        stats["channels"][channel] = stats["channels"].get(channel, 0) + 1

        if rate_limit > 0:
            import time
            time.sleep(rate_limit)

    import time as time_mod
    manifest = {
        "generated_at": time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", time_mod.gmtime()),
        "stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Fetch complete: %d fetched, %d no subtitles (total %d URLs)",
        stats["fetched"], stats["no_subtitles"], stats["total_urls"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts using yt-dlp for therapeutic AI training."
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
        default="Tim Fletcher",
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
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        help="Load cookies from browser (firefox, chrome, chromium, brave, etc.)"
    )

    args = parser.parse_args()

    # Parse arguments
    playlists_file = Path(args.playlists_file)
    output_dir = Path(args.output_dir) / args.channel_name
    channel_name = args.channel_name
    languages = [lang.strip() for lang in args.language.split(",")]
    max_videos = args.max_videos
    skip_existing = args.skip_existing
    cookies_from_browser = args.cookies_from_browser

    logger.info("Starting YouTube transcript fetcher")
    logger.info(f"Playlists file: {playlists_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Channel name: {channel_name}")
    logger.info(f"Languages: {languages}")
    if cookies_from_browser:
        logger.info(f"Using cookies from: {cookies_from_browser}")

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
            if not video_id or len(video_id) != 11:  # Standard YouTube video ID length
                logger.warning(f"Could not extract valid video ID from URL: {url}")
                failed += 1
                continue

            # Check if file already exists
            output_file = output_dir / f"{video_id}.txt"
            if skip_existing and output_file.exists():
                logger.debug(f"Skipping existing transcript for {video_id}")
                skipped += 1
                continue

            # Fetch transcript
            if fetch_transcript(video_id, output_file, languages, cookies_from_browser):
                successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {e}")
            failed += 1

    # Create manifest
    manifest = {
        "generated_at": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "channel": channel_name,
        "total_urls_processed": len(urls),
        "successful": successful,
        "failed": failed,
        "skipped_existing": skipped,
        "output_directory": str(output_dir),
        "languages_attempted": languages
    }

    manifest_path = output_dir / "manifest.json"
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
