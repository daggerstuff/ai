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
import random
import re
import shutil
import subprocess
import sys
import tempfile
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


def _exec_cmd(cmd: list[str], timeout: int = 60, check: bool = False) -> subprocess.CompletedProcess | None:
    """Centralized subprocess execution with logging and timeout handling."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ds: %s", timeout, " ".join(cmd))
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (code %d): %s\n%s", e.returncode, " ".join(cmd), e.stderr)
    except Exception as e:
        logger.error("Unexpected error executing %s: %s", " ".join(cmd), e)
    return None


def sync_from_gdrive(playlist_path: Path):
    if not shutil.which("rclone"):
        logger.error("rclone binary not found. GDrive sync skipped.")
        return

    res = _exec_cmd(["rclone", "config", "show", "gdrive:"])
    if not res or res.returncode != 0:
        logger.error("rclone 'gdrive:' remote not configured. GDrive sync skipped.")
        return

    logger.info("Syncing playlist from GDrive to %s", playlist_path)
    if _exec_cmd(["rclone", "copy", "gdrive:pixelated/.notes/youtube_playlists.txt", str(playlist_path.parent)]):
        logger.info("GDrive sync successful.")


def jittered_sleep(base_delay: float):
    if base_delay <= 0:
        return
    total_sleep = base_delay + random.uniform(0, base_delay * 0.5)
    logger.debug("Sleeping for %.2fs", total_sleep)
    time.sleep(total_sleep)


def _clean_subtitle(raw: str) -> str:
    lines: list[str] = []
    prev = ""
    for line in raw.splitlines():
        if _VTT_HEADER_RE.match(line) or _TIMESTAMP_RE.match(line) or \
           _TIMESTAMP_SHORT_RE.match(line) or _SEQUENCE_NUM_RE.match(line) or \
           _BLANK_LINE_RE.match(line):
            continue
            
        line = _HTML_TAG_RE.sub("", _BRACKET_RE.sub("", _MUSIC_RE.sub("", line))).strip()
        if not line or (line == prev and not (line.startswith("SPEAKER:") or _SPEAKER_TAG_RE.search(line))):
            continue
            
        lines.append(line)
        prev = line
        
    return "\n\n".join(lines)


def _slugify(text: str) -> str:
    return re.sub(r"[-\s]+", "_", re.sub(r"[^\w\s-]", "", text.lower().strip()))[:120]


def _run_ytdlp(args: list[str], config: dict, timeout: int = 60) -> subprocess.CompletedProcess | None:
    cmd = ["yt-dlp"] + args
    if config.get("cookies"):
        cmd.extend(["--cookies", config["cookies"]])
    elif config.get("cookies_from_browser"):
        cmd.extend(["--cookies-from-browser", config["cookies_from_browser"]])
    
    if config.get("user_agent"):
        cmd.extend(["--user-agent", config["user_agent"]])
    if config.get("node_path"):
        cmd.extend(["--js-runtimes", f"node:{config['node_path']}"])
    
    return _exec_cmd(cmd, timeout=timeout)


def _fetch_video_metadata(url: str, config: dict) -> dict | None:
    res = _run_ytdlp(["--dump-json", "--no-download", "--no-playlist", url], config)
    if not res or res.returncode != 0:
        return None
    try:
        meta = json.loads(res.stdout)
        return {
            "channel": meta.get("channel", "Unknown").strip(),
            "title": meta.get("title", "Untitled").strip(),
            "id": meta.get("id", ""),
            "duration": meta.get("duration", 0),
        }
    except json.JSONDecodeError as e:
        logger.warning("Metadata JSON error for %s: %s", url, e)
        return None


def _fetch_subtitles(url: str, output_dir: Path, lang: str, config: dict) -> Path | None:
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="subs_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        res = _run_ytdlp(
            [
                "--write-subs", "--write-auto-subs", "--sub-lang", lang,
                "--skip-download", "--no-playlist", "--sub-format", "vtt/srt",
                "-o", str(tmp_path / "temp"), url
            ],
            config, timeout=120
        )
        if not res or res.returncode != 0:
            return None
            
        candidates = sorted(tmp_path.glob("temp*.vtt")) + sorted(tmp_path.glob("temp*.srt"))
        if not candidates:
            return None
        
        final_path = output_dir / f"{candidates[0].name}_{random.getrandbits(32)}"
        shutil.copy(candidates[0], final_path)
        return final_path


def process_video(url: str, output_dir: Path, lang: str, config: dict, stats: dict) -> None:
    """Process a single video: fetch metadata, download subs, clean, and save."""
    meta = _fetch_video_metadata(url, config)
    if not meta:
        stats["no_metadata"] += 1
        stats["errors"].append({"url": url, "error": "metadata_failed"})
        return

    channel, video_id, title, duration = meta["channel"], meta["id"], meta["title"], meta.get("duration", 0)
    slug = _slugify(f"{video_id}_{title}") if video_id else _slugify(title)
    
    channel_dir = output_dir / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    dest_path = channel_dir / f"{slug}.txt"

    def update_stats():
        stats["fetched"] += 1
        stats["total_duration"] += duration
        if channel not in stats["channels"]:
            stats["channels"][channel] = {"count": 0, "duration": 0}
        stats["channels"][channel]["count"] += 1
        stats["channels"][channel]["duration"] += duration

    if dest_path.exists():
        logger.debug("Already exists: %s", dest_path.name)
        update_stats()
        return

    temp_root = output_dir / ".tmp_subs"
    temp_root.mkdir(parents=True, exist_ok=True)

    sub_path = _fetch_subtitles(url, temp_root, lang, config)
    if not sub_path:
        stats["no_subtitles"] += 1
        stats["errors"].append({"url": url, "channel": channel, "error": "no_subtitles"})
        return

    try:
        raw_text = sub_path.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean_subtitle(raw_text)
        
        if len(cleaned.strip()) < 50:
            logger.debug("Subtitle too short for %s", url)
            stats["no_subtitles"] += 1
            stats["errors"].append({"url": url, "channel": channel, "error": "subtitle_too_short"})
        else:
            dest_path.write_text(cleaned, encoding="utf-8")
            update_stats()
            logger.info("  → %s/%s (%d chars)", channel, slug[:40], len(cleaned))
    except Exception as e:
        logger.error("Error processing %s: %s", url, e)
        stats["errors"].append({"url": url, "channel": channel, "error": str(e)})
    finally:
        # Clean up the individual subtitle file
        if sub_path.exists():
            sub_path.unlink(missing_ok=True)


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


def _format_duration(seconds: float | int) -> str:
    """Format duration in seconds to HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def run_fetch(args: argparse.Namespace) -> None:
    url_file, output_dir = Path(args.url_file), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_gdrive:
        sync_from_gdrive(url_file)

    cookies_path = Path(args.cookies)
    config = {
        "cookies": str(cookies_path) if cookies_path.exists() else None,
        "cookies_from_browser": args.cookies_from_browser,
        "user_agent": args.user_agent,
        "node_path": args.node_path
    }

    if not config["cookies"] and not config["cookies_from_browser"]:
        logger.warning("No cookies provided. Rate limiting or age-restricted content may cause failures.")

    urls = _read_urls(url_file)
    if args.max_urls > 0:
        urls = urls[:args.max_urls]
    logger.info("Loaded %d unique URLs", len(urls))

    stats = {
        "total_urls": len(urls),
        "fetched": 0,
        "no_subtitles": 0,
        "no_metadata": 0,
        "total_duration": 0,
        "channels": {},
        "errors": []
    }

    try:
        for i, url in enumerate(urls):
            logger.info("[%d/%d] Processing %s", i + 1, len(urls), url)
            process_video(url, output_dir, args.lang, config, stats)
            if args.rate_limit > 0 and (i + 1) < len(urls):
                jittered_sleep(args.rate_limit)
    finally:
        shutil.rmtree(output_dir / ".tmp_subs", ignore_errors=True)

    # Final formatting of durations
    stats["total_duration_hms"] = _format_duration(stats["total_duration"])
    for chan in stats["channels"].values():
        chan["duration_hms"] = _format_duration(chan["duration"])

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(url_file),
        "language": args.lang,
        "stats": stats
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("--- FETCH SUMMARY ---")
    logger.info("  Total URLs: %d", stats["total_urls"])
    logger.info("  Fetched:    %d", stats["fetched"])
    logger.info("  No Subs:    %d", stats["no_subtitles"])
    logger.info("  No Meta:    %d", stats["no_metadata"])
    logger.info("  Duration:   %s", stats["total_duration_hms"])
    logger.info("  Channels:   %d", len(stats["channels"]))

    if stats["errors"]:
        logger.error("--- ERROR DETAIL ---")
        for err in stats["errors"][:20]:  # Show first 20 errors
            logger.error("  %s: %s", err.get("url"), err.get("error"))
        if len(stats["errors"]) > 20:
            logger.error("  ... and %d more errors", len(stats["errors"]) - 20)

    if stats["fetched"] > 0:
        logger.info("")
        logger.info("--- DATASET READY FOR S3 SYNC ---")
        logger.info("  Target: s3://pixelated-training-data/transcripts/")
        logger.info("  Source: %s", output_dir)
        logger.info("---------------------------------")
    else:
        logger.warning("No videos were fetched successfully.")


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
    parser.add_argument("--node_path", type=str, default=None,
        help="Path to node binary (if required by yt-dlp).")
    parser.add_argument("--rate_limit", type=float, default=2.0, help="Seconds to wait between videos.")
    parser.add_argument("--max_urls", type=int, default=0, help="Max URLs to process.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_fetch(build_parser().parse_args())


if __name__ == "__main__":
    main()
