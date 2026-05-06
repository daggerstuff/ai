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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("transcript_fetcher")

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TIMESTAMP_RE = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}")
_TIMESTAMP_SHORT_RE = re.compile(r"^\s*\d{1,2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}[.,]\d{3}")
_SEQUENCE_NUM_RE = re.compile(r"^\s*\d+\s*$")
_BLANK_LINE_RE = re.compile(r"^\s*$")
_MUSIC_RE = re.compile(r"♪|♫|🎵|🎶|\[music\]|\[Music\]", re.IGNORECASE)
_BRACKET_RE = re.compile(r"^\[.*\]\s*$")
_VTT_HEADER_RE = re.compile(r"^(WEBVTT|Kind:|Language:|NOTE)", re.IGNORECASE)


def _clean_subtitle(raw: str) -> str:
    """Strip timestamp lines, sequence numbers, HTML tags, and music markers."""
    lines: list[str] = []
    prev = ""
    for line in raw.splitlines():
        if _VTT_HEADER_RE.match(line):
            continue
        if _TIMESTAMP_RE.match(line) or _TIMESTAMP_SHORT_RE.match(line):
            continue
        if _SEQUENCE_NUM_RE.match(line):
            continue
        if _BLANK_LINE_RE.match(line):
            continue
        line = _HTML_TAG_RE.sub("", line).strip()
        if not line:
            continue
        if _MUSIC_RE.search(line):
            continue
        if _BRACKET_RE.match(line):
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    return "\n\n".join(lines)


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
    slug = re.sub(r"[-\s]+", "_", slug)
    return slug[:120]


def _run_ytdlp(
    args: list[str], 
    cookies: str | None = None, 
    node_path: str | None = None, 
    timeout: int = 60
) -> subprocess.CompletedProcess:
    """Helper to run yt-dlp with common flags."""
    cmd = ["yt-dlp"] + args
    if cookies:
        cmd.extend(["--cookies", cookies])
    if node_path:
        cmd.extend(["--js-runtimes", f"node:{node_path}"])
    
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )


def _fetch_video_metadata(url: str, cookies: str | None = None, node_path: str | None = None) -> dict | None:
    """Get video metadata (channel, title) via yt-dlp --dump-json."""
    try:
        result = _run_ytdlp(
            ["--dump-json", "--no-download", "--no-playlist", url],
            cookies=cookies, node_path=node_path
        )
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


def _fetch_subtitles(
    url: str, 
    output_dir: Path, 
    lang: str = "en", 
    cookies: str | None = None, 
    node_path: str | None = None
) -> Path | None:
    """Download subtitles for a single video. Returns the .vtt/.srt path or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_ytdlp(
            [
                "--write-subs", "--write-auto-subs",
                "--sub-lang", lang,
                "--skip-download",
                "--no-playlist",
                "--sub-format", "vtt/srt",
                "-o", str(output_dir / "temp"),
                url
            ],
            cookies=cookies, node_path=node_path, timeout=120
        )
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if "no subtitle" in stderr_lower or "subtitles" not in result.stdout.lower():
                logger.debug("No subtitles found for %s (%s): %s", url, lang, result.stderr[:200])
                return None
        # Find the downloaded subtitle file
        candidates = sorted(output_dir.glob("temp*.vtt")) + sorted(output_dir.glob("temp*.srt"))
        if not candidates:
            logger.debug("No subtitle file written for %s (%s)", url, lang)
            return None
        return candidates[0]
    except subprocess.TimeoutExpired:
        logger.warning("Subtitle download timed out for %s", url)
        return None


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
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        urls.append(line)
    return urls


def run_fetch(args: argparse.Namespace) -> None:
    url_file = Path(args.url_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lang = args.lang
    cookies = args.cookies
    node_path = args.node_path
    rate_limit = args.rate_limit
    max_urls = args.max_urls

    urls = _read_urls(url_file)
    if max_urls > 0:
        urls = urls[:max_urls]
    logger.info("Loaded %d unique URLs from %s", len(urls), url_file)

    channel_dirs: dict[str, Path] = {}
    stats = {
        "total_urls": len(urls),
        "fetched": 0,
        "no_subtitles": 0,
        "no_metadata": 0,
        "channels": {},
        "errors": [],
    }

    for i, url in enumerate(urls):
        logger.info("[%d/%d] Processing %s", i + 1, len(urls), url)

        meta = _fetch_video_metadata(url, cookies=cookies, node_path=node_path)
        if not meta:
            stats["no_metadata"] += 1
            stats["errors"].append({"url": url, "error": "metadata_failed"})
            if rate_limit > 0:
                time.sleep(rate_limit)
            continue

        channel = meta["channel"]
        video_id = meta["id"]
        title = meta["title"]
        slug = _slugify(f"{video_id}_{title}") if video_id else _slugify(title)

        if channel not in channel_dirs:
            channel_dir = output_dir / channel
            channel_dir.mkdir(parents=True, exist_ok=True)
            channel_dirs[channel] = channel_dir
            stats["channels"][channel] = 0

        channel_dir = channel_dirs[channel]
        dest_path = channel_dir / f"{slug}.txt"

        if dest_path.exists():
            logger.debug("Already exists: %s", dest_path.name)
            stats["channels"][channel] = stats["channels"].get(channel, 0) + 1
            stats["fetched"] += 1
            if rate_limit > 0:
                time.sleep(rate_limit)
            continue

        temp_dir = output_dir / ".tmp_subs"
        temp_dir.mkdir(parents=True, exist_ok=True)

        sub_path = _fetch_subtitles(url, temp_dir, lang, cookies=cookies, node_path=node_path)
        if not sub_path:
            stats["no_subtitles"] += 1
            stats["errors"].append({"url": url, "channel": channel, "error": "no_subtitles"})
            if rate_limit > 0:
                time.sleep(rate_limit)
            continue

        raw_text = sub_path.read_text(encoding="utf-8", errors="replace")
        cleaned = _clean_subtitle(raw_text)

        # Clean up temp subtitle file
        for f in temp_dir.glob("temp*"):
            f.unlink(missing_ok=True)

        if len(cleaned.strip()) < 50:
            logger.debug("Subtitle too short for %s (%d chars)", url, len(cleaned.strip()))
            stats["no_subtitles"] += 1
            stats["errors"].append({"url": url, "channel": channel, "error": "subtitle_too_short"})
            if rate_limit > 0:
                time.sleep(rate_limit)
            continue

        dest_path.write_text(cleaned, encoding="utf-8")
        stats["fetched"] += 1
        stats["channels"][channel] = stats["channels"].get(channel, 0) + 1

        logger.info("  → %s/%s (%d chars)", channel, slug[:40], len(cleaned))

        if rate_limit > 0:
            time.sleep(rate_limit)

    # Clean up temp dir
    temp_dir = output_dir / ".tmp_subs"
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()

    # Write manifest
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(url_file),
        "language": lang,
        "stats": stats,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    logger.info(
        "Fetch complete: %d/%d fetched, %d no subtitles, %d no metadata, %d channels",
        stats["fetched"], stats["total_urls"],
        stats["no_subtitles"], stats["no_metadata"],
        len(stats["channels"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube transcripts via yt-dlp for training data.",
    )
    parser.add_argument(
        "--url_file", type=str, required=True,
        help="File with one YouTube URL per line.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="training/data/transcripts",
        help="Output directory for per-channel .txt transcripts.",
    )
    parser.add_argument(
        "--lang", type=str, default="en",
        help="Subtitle language to download (default: en). Comma-separated for multiple.",
    )
    parser.add_argument(
        "--cookies", type=str, default="youtube_cookies.txt",
        help="Path to yt-dlp cookies file (default: youtube_cookies.txt).",
    )
    parser.add_argument(
        "--node_path", type=str, default="/home/vivi/.config/nvm/versions/node/v24.14.1/bin/node",
        help="Path to node binary for yt-dlp js-runtimes.",
    )
    parser.add_argument(
        "--rate_limit", type=float, default=2.0,
        help="Seconds to wait between videos (default: 2.0).",
    )
    parser.add_argument(
        "--max_urls", type=int, default=0,
        help="Max URLs to process (0 = all). Useful for testing.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_fetch(args)


if __name__ == "__main__":
    main()
