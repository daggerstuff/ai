#!/usr/bin/env python3
"""
Batch YouTube transcript downloader using yt-dlp.
Reads playlist files, downloads auto-generated subtitles, converts to plain text.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yt_transcript_batch")


def extract_video_id(url: str) -> str | None:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?", maxsplit=1)[0]
    if "youtube.com/watch" in url:
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(url).query)
        return qs.get("v", [None])[0]
    return None


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        logger.error(f"File not found: {path}")
        return []
    urls = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    logger.info(f"Read {len(urls)} URLs, {len(unique)} unique from {path.name}")
    return unique


def build_ytdlp_cmd(output_dir: Path, url: str) -> list[str]:
    """Build yt-dlp command, optionally with cookies for local execution."""
    cmd = [
        "uv",
        "run",
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-langs",
        "en",
        "-o",
        str(output_dir / "%(id)s"),
    ]
    if getattr(build_ytdlp_cmd, "cookies_file", None):
        cmd.extend(["--cookies", build_ytdlp_cmd.cookies_file])
    if getattr(build_ytdlp_cmd, "cookies_from_browser", None):
        cmd.extend(["--cookies-from-browser", build_ytdlp_cmd.cookies_from_browser])
    cmd.append(url)
    return cmd


def fetch_one(video_id: str, output_dir: Path) -> tuple[str, bool]:
    """Download a single video's auto-subtitles using yt-dlp."""
    output_dir / f"{video_id}.vtt"
    txt_file = output_dir / f"{video_id}.txt"

    # Skip if .txt already exists
    if txt_file.exists():
        return video_id, True  # already done

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        result = subprocess.run(
            build_ytdlp_cmd(output_dir, url),
            capture_output=True,
            text=True,
            timeout=90,
        )

        if result.returncode != 0:
            logger.warning(f"{video_id}: yt-dlp failed (rc={result.returncode})")
            return video_id, False

        # Find the .vtt file that was created
        vtt_files = list(output_dir.glob(f"{video_id}*.vtt"))
        if not vtt_files:
            logger.warning(f"{video_id}: no subtitle file created")
            return video_id, False

        # Convert VTT to plain text
        vtt_path = vtt_files[0]
        text = convert_vtt_to_text(vtt_path)
        if not text:
            logger.warning(f"{video_id}: empty transcript after conversion")
            vtt_path.unlink(missing_ok=True)
            return video_id, False

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)

        # Remove .vtt file
        vtt_path.unlink(missing_ok=True)

        word_count = len(text.split())
        logger.info(f"{video_id}: SUCCESS - {word_count} words")
        return video_id, True

    except subprocess.TimeoutExpired:
        logger.warning(f"{video_id}: timeout")
        return video_id, False
    except Exception as e:
        logger.warning(f"{video_id}: error - {e}")
        return video_id, False


def convert_vtt_to_text(vtt_path: Path) -> str:
    """Convert VTT subtitle format to plain text."""
    with open(vtt_path, encoding="utf-8") as f:
        content = f.read()

    # Remove VTT header
    lines = content.split("\n")
    text_lines = []
    for line in lines:
        line = line.strip()
        # Skip VTT metadata, timestamps, and empty lines
        if (
            line.startswith(("WEBVTT", "Kind:", "Language:", "<c.", "</c>")) or "-->" in line or not line
        ):
            continue
        # Remove VTT tags like <c.hand> etc
        line = re.sub(r"<[^>]+>", "", line)
        line = line.strip()
        if line:
            text_lines.append(line)

    return " ".join(text_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Batch YouTube transcript downloader using yt-dlp",
        epilog=(
            "NOTE: YouTube blocks cloud IPs. For local execution, use:\n"
            "  --cookies-from-browser chrome   (auto-detect from your browser)\n"
            "  --cookies cookies.txt           (exported cookies file)\n"
            "See: https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies"
        ),
    )
    parser.add_argument("--playlists-dir", type=str, default="ai/docs/playlists")
    parser.add_argument("--output-dir", type=str, default="ai/training/youtube_transcripts")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel downloads (default 3)")
    parser.add_argument("--channel", type=str, help="Single channel to process (optional)")
    parser.add_argument(
        "--cookies", type=str, default=None, help="Path to Netscape-format cookies.txt file for YouTube auth"
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        help="Browser to extract cookies from (e.g. 'chrome', 'firefox')",
    )
    args = parser.parse_args()

    playlists_dir = Path(args.playlists_dir)
    output_base = Path(args.output_dir)

    # Wire cookie options into fetch_one via function attribute
    if args.cookies:
        build_ytdlp_cmd.cookies_file = args.cookies
        logger.info("Using cookies file: %s", args.cookies)
    if args.cookies_from_browser:
        build_ytdlp_cmd.cookies_from_browser = args.cookies_from_browser
        logger.info("Using cookies from browser: %s", args.cookies_from_browser)

    # Build channel list
    channels = []
    if args.channel:
        # Single channel mode
        playlist_file = playlists_dir / f"{args.channel}.txt"
        if not playlist_file.exists():
            logger.error(f"Playlist not found: {playlist_file}")
            return 1
        urls = read_urls(playlist_file)
        channel_name = args.channel
        output_dir = output_base / channel_name
        channels.append((channel_name, urls, output_dir))
    else:
        # All playlists
        for playlist_path in sorted(playlists_dir.glob("*.txt")):
            if playlist_path.name == "youtube_playlists.txt":
                continue
            channel_name = playlist_path.stem
            urls = read_urls(playlist_path)
            if urls:
                channels.append((channel_name, urls, output_base / channel_name))

    total_success = 0
    total_fail = 0
    total_skipped = 0

    for channel_name, urls, output_dir in channels:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Channel: {channel_name} ({len(urls)} videos)")
        logger.info(f"Output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract video IDs
        video_ids = []
        for url in urls:
            vid = extract_video_id(url)
            if vid:
                video_ids.append(vid)
            else:
                logger.warning(f"Could not extract ID from: {url}")

        # Count existing
        existing = sum(1 for vid in video_ids if (output_dir / f"{vid}.txt").exists())
        if existing:
            logger.info(f"  {existing}/{len(video_ids)} already have .txt files")
            total_skipped += existing

        # Download in parallel
        success = 0
        fail = 0
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(fetch_one, vid, output_dir): vid for vid in video_ids}
            for future in as_completed(futures):
                vid, ok = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1

        logger.info(f"  {channel_name}: {success} success, {fail} failed")
        total_success += success
        total_fail += fail

        # Save manifest
        manifest = {
            "generated_at": datetime.now(UTC).isoformat(),
            "channel": channel_name,
            "urls_processed": len(urls),
            "successful": success,
            "failed": fail,
            "skipped_existing": existing,
            "total_files": len(list(output_dir.glob("*.txt"))),
        }
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"TOTAL: {total_success} success, {total_fail} failed, {total_skipped} skipped")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
