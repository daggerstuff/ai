#!/usr/bin/env uv run
"""
YouTube Transcript Fetcher (Phase 2 acquisition step)

* Reads a list of YouTube URLs (one per line) from a playlist file.
* Uses ``yt-dlp`` to download automatic subtitles for English and German.
* Stores cleaned ``.txt`` transcripts in ``training/data/transcripts/<channel>/``
  matching the layout expected by ``youtube_ingestion.py``.
* Generates a ``manifest.json`` mapping channel names to the number of transcripts
  written.

The script is deliberately lightweight – it relies on ``yt-dlp`` (which is
already a dependency of the repo) and does not require a YouTube API key.
"""

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger("transcript_fetcher")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(handler)

# Regular expression to strip timestamps from .vtt files
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")


def clean_vtt(vtt_path: Path) -> str:
    """Return plain‑text transcript from a .vtt file.

    Removes timestamp lines, empty lines and HTML‑style tags.
    """
    lines = []
    for line in vtt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if TIMESTAMP_RE.match(line):
            continue
        # Remove any HTML tags that may appear in WebVTT captions
        line = re.sub(r"<[^>]+>", "", line)
        lines.append(line)
    return "\n".join(lines)


def download_subtitles(url: str, out_dir: Path, cookies_path: Path | None = None) -> Path:
    """Download auto subtitles for a YouTube URL.

    Returns the path to the downloaded ``.vtt`` file (English preferred,
    falling back to German if English is unavailable). Uses ``yt-dlp`` with
    ``--skip-download``.
    """
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # yt‑dlp output template places files in the target directory with the
    # uploader name as a sub‑directory. ``%(uploader)s`` expands to the channel name.
    template = str(out_dir / "%(uploader)s/%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--write-auto-sub",
        "--sub-lang", "en,de",
        "--skip-download",
        "--output", template,
        "--convert-subs", "srt",  # ensure a simple text subtitle format
    ]
    if cookies_path:
        cmd.extend(["--cookies", str(cookies_path)])
    cmd.append(url)
    log.info(f"Running yt-dlp for {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"yt-dlp failed for {url}: {result.stderr.strip()}")
        raise RuntimeError(f"yt-dlp error for {url}")
    # yt-dlp will create a .srt file; we rename it to .vtt for the cleaner step
    # Locate the generated file (it will be the most recent file in out_dir)
    generated = max(out_dir.rglob("*.srt"), key=lambda p: p.stat().st_mtime)
    vtt_path = generated.with_suffix('.vtt')
    generated.rename(vtt_path)
    return vtt_path


def main():
    parser = argparse.ArgumentParser(description="Download YouTube transcripts via yt‑dlp")
    parser.add_argument(
        "--playlist",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "youtube_playlists.txt"),
        help="Path to file containing YouTube URLs (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "training" / "data" / "transcripts"),
        help="Base directory for channel sub‑folders",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to yt‑dlp cookies file for authenticated downloads",
    )
    args = parser.parse_args()
    playlist_path = Path(args.playlist)
    out_base = Path(args.output_dir)
    if not playlist_path.is_file():
        log.error(f"Playlist file not found: {playlist_path}")
        return
    manifest = {}
    for line in playlist_path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url:
            continue
        try:
            vtt_path = download_subtitles(url, out_base)
            text = clean_vtt(vtt_path)
            # Write cleaned transcript to a .txt file alongside the .vtt
            txt_path = vtt_path.with_suffix('.txt')
            txt_path.write_text(text, encoding="utf-8")
            # Update manifest
            channel = vtt_path.parent.name
            manifest.setdefault(channel, 0)
            manifest[channel] += 1
            log.info(f"Saved transcript for {channel}: {txt_path.name}")
        except Exception as e:
            log.error(f"Failed processing {url}: {e}")
    # Write manifest.json
    manifest_path = out_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
