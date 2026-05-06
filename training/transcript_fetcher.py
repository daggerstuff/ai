#!/usr/bin/env python3
"""YouTube transcript fetcher for therapeutic AI training data.

Downloads subtitles/transcripts from YouTube video URLs using yt-dlp,
cleans subtitle formatting, and organizes output into the per-channel
.txt directory structure that youtube_ingestion.py consumes.

Input:  A file with one YouTube URL per line (deduplicates automatically).
Output: training/data/transcripts/<channel>/<slug>.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("transcript_fetcher")

# Regex patterns for cleaning
_HTML_TAG_RE = re.compile(r"<(?!/?v\b)[^>]+>")
_SPEAKER_TAG_RE = re.compile(r"<(/?v\b[^>]*)>")
_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}")
_TIMESTAMP_SHORT_RE = re.compile(r"^\s*\d{1,2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}[.,]\d{3}")
_SEQUENCE_NUM_RE = re.compile(r"^\s*\d+\s*$")
_BLANK_LINE_RE = re.compile(r"^\s*$")
_MUSIC_RE = re.compile(r"♪|♫|🎵|🎶|\[music\]|\[Music\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[.*?\]")
_VTT_HEADER_RE = re.compile(r"^(WEBVTT|Kind:|Language:|NOTE)", re.IGNORECASE)


def sync_from_gdrive(playlist_path: Path):
    """Sync the playlist file from GDrive using rclone."""
    if not shutil.which("rclone"):
        logger.error("rclone binary not found. GDrive sync skipped.")
        return

    logger.info("Syncing playlist from GDrive to %s", playlist_path)
    source = "gdrive:pixelated/.notes/youtube_playlists.txt"
    try:
        subprocess.run(
            ["rclone", "copy", source, str(playlist_path.parent)],
            check=True, capture_output=True, text=True
        )
        logger.info("GDrive sync successful.")
    except subprocess.CalledProcessError as e:
        logger.error("GDrive sync failed: %s", e.stderr.strip())


def _clean_subtitle(raw: str) -> str:
    """Strip timestamp lines, sequence numbers, non-speaker HTML tags, and music markers."""
    lines: list[str] = []
    prev = ""
    for line in raw.splitlines():
        if _VTT_HEADER_RE.match(line) or _TIMESTAMP_RE.match(line) or \
           _TIMESTAMP_SHORT_RE.match(line) or _SEQUENCE_NUM_RE.match(line) or \
           _BLANK_LINE_RE.match(line):
            continue
            
        line = _MUSIC_RE.sub("", line)
        line = _BRACKET_RE.sub("", line)
        line = _HTML_TAG_RE.sub("", line).strip()
        
        if not line:
            continue
            
        is_speaker = line.startswith("SPEAKER:") or _SPEAKER_TAG_RE.search(line)
        if line == prev and not is_speaker:
            continue
            
        lines.append(line)
        prev = line
        
    return "\n\n".join(lines)


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:120]


def _run_ytdlp(args: list[str], config: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    """Helper to run yt-dlp with common flags."""
    cmd = ["yt-dlp"] + args
    if config.get("cookies"):
        cmd.extend(["--cookies", config["cookies"]])
    if config.get("cookies_from_browser"):
        cmd.extend(["--cookies-from-browser", config["cookies_from_browser"]])
    if config.get("user_agent"):
        cmd.extend(["--user-agent", config["user_agent"]])
    if config.get("node_path"):
        cmd.extend(["--js-runtimes", f"node:{config['node_path']}"])
    
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _fetch_video_metadata(url: str, config: dict) -> dict | None:
    """Get video metadata (channel, title) via yt-dlp --dump-json."""
    try:
        result = _run_ytdlp(["--dump-json", "--no-download", "--no-playlist", url], config)
        if result.returncode != 0:
            logger.warning("Metadata failed for %s: %s", url, result.stderr[:200])
            return None
        meta = json.loads(result.stdout)
        return {
            "channel": meta.get("channel", "Unknown").strip(),
            "title": meta.get("title", "Untitled").strip(),
            "id": meta.get("id", ""),
            "duration": meta.get("duration", 0),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.warning("Metadata error for %s: %s", url, exc)
        return None


def _fetch_subtitles(url: str, output_dir: Path, lang: str, config: dict) -> Path | None:
    """Download subtitles for a single video. Returns the .vtt/.srt path or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_ytdlp(
            [
                "--write-subs", "--write-auto-subs", "--sub-lang", lang,
                "--skip-download", "--no-playlist", "--sub-format", "vtt/srt",
                "-o", str(output_dir / "temp"), url
            ],
            config, timeout=120
        )
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if "no subtitle" in stderr_lower or "subtitles" not in result.stdout.lower():
                logger.debug("No subtitles found for %s (%s)", url, lang)
                return None
        candidates = sorted(output_dir.glob("temp*.vtt")) + sorted(output_dir.glob("temp*.srt"))
        return candidates[0] if candidates else None
    except subprocess.TimeoutExpired:
        logger.warning("Subtitle download timed out for %s", url)
        return None


def process_video(url: str, output_dir: Path, lang: str, config: dict, stats: dict) -> None:
    """Process a single video: fetch metadata, download subs, clean, and save."""
    meta = _fetch_video_metadata(url, config)
    if not meta:
        stats["no_metadata"] += 1
        stats["errors"].append({"url": url, "error": "metadata_failed"})
        return

    channel, video_id, title = meta["channel"], meta["id"], meta["title"]
    slug = _slugify(f"{video_id}_{title}") if video_id else _slugify(title)
    
    channel_dir = output_dir / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    dest_path = channel_dir / f"{slug}.txt"

    if dest_path.exists():
        logger.debug("Already exists: %s", dest_path.name)
        stats["fetched"] += 1
        stats["channels"][channel] = stats["channels"].get(channel, 0) + 1
        return

    temp_dir = output_dir / ".tmp_subs"
    temp_dir.mkdir(parents=True, exist_ok=True)

    sub_path = _fetch_subtitles(url, temp_dir, lang, config)
    if not sub_path:
        stats["no_subtitles"] += 1
        stats["errors"].append({"url": url, "channel": channel, "error": "no_subtitles"})
        return

    raw_text = sub_path.read_text(encoding="utf-8", errors="replace")
    cleaned = _clean_subtitle(raw_text)

    # Clean up temp files
    for f in temp_dir.glob("temp*"):
        f.unlink(missing_ok=True)

    if len(cleaned.strip()) < 50:
        logger.debug("Subtitle too short for %s", url)
        stats["no_subtitles"] += 1
        stats["errors"].append({"url": url, "channel": channel, "error": "subtitle_too_short"})
        return

    dest_path.write_text(cleaned, encoding="utf-8")
    stats["fetched"] += 1
    stats["channels"][channel] = stats["channels"].get(channel, 0) + 1
    logger.info("  → %s/%s (%d chars)", channel, slug[:40], len(cleaned))


def _read_urls(path: Path) -> list[str]:
    """Read URLs from a file, deduplicate, and return as list."""
    if not path.exists():
        logger.error("URL file not found: %s", path)
        sys.exit(1)
    raw = path.read_text(encoding="utf-8", errors="replace")
    urls: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        urls.append(line)
    return urls


def run_fetch(args: argparse.Namespace) -> None:
    url_file, output_dir = Path(args.url_file), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_gdrive:
        sync_from_gdrive(url_file)

    config = {
        "cookies": args.cookies,
        "cookies_from_browser": args.cookies_from_browser,
        "user_agent": args.user_agent,
        "node_path": args.node_path
    }

    urls = _read_urls(url_file)
    if args.max_urls > 0:
        urls = urls[:args.max_urls]
    logger.info("Loaded %d unique URLs", len(urls))

    stats = {
        "total_urls": len(urls), "fetched": 0, "no_subtitles": 0,
        "no_metadata": 0, "channels": {}, "errors": []
    }

    for i, url in enumerate(urls):
        logger.info("[%d/%d] Processing %s", i + 1, len(urls), url)
        process_video(url, output_dir, args.lang, config, stats)
        if args.rate_limit > 0:
            time.sleep(args.rate_limit)

    shutil.rmtree(output_dir / ".tmp_subs", ignore_errors=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(url_file), "language": args.lang, "stats": stats
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Fetch complete: %d/%d fetched, %d no subs, %d no meta, %d channels",
        stats["fetched"], stats["total_urls"], stats["no_subtitles"],
        stats["no_metadata"], len(stats["channels"])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch YouTube transcripts via yt-dlp.")
    parser.add_argument("--url_file", type=str, required=True, help="File with one YouTube URL per line.")
    parser.add_argument("--sync-gdrive", action="store_true", help="Sync playlist file from GDrive.")
    parser.add_argument("--output_dir", type=str, default="training/data/transcripts", help="Output directory.")
    parser.add_argument("--lang", type=str, default="en", help="Subtitle language.")
    parser.add_argument("--cookies", type=str, default="youtube_cookies.txt", help="Path to cookies file.")
    parser.add_argument("--cookies-from-browser", type=str, default=None, help="Fetch cookies from browser.")
    parser.add_argument("--user-agent", type=str, 
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        help="Custom User-Agent.")
    parser.add_argument("--node_path", type=str, default="/home/vivi/.config/nvm/versions/node/v24.14.1/bin/node",
        help="Path to node binary.")
    parser.add_argument("--rate_limit", type=float, default=2.0, help="Seconds to wait between videos.")
    parser.add_argument("--max_urls", type=int, default=0, help="Max URLs to process.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_fetch(build_parser().parse_args())


if __name__ == "__main__":
    main()
