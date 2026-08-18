#!/usr/bin/env python3
"""
YouTube transcript fetcher using Selenium with undetected-chromedriver.
Bypasses bot detection by using a real browser.
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("youtube_selenium_fetcher")


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


def fetch_transcript_with_selenium(video_id: str, language_code: str = "en") -> str | None:
    """
    Fetch transcript for a single video using Selenium and undetected-chromedriver.

    Args:
        video_id: YouTube video ID
        language_code: Language code for transcript (default: 'en')

    Returns:
        Transcript text if successful, None otherwise
    """
    driver = None
    try:
        # Set up undetected-chromedriver
        options = uc.ChromeOptions()
        options.add_argument("--headless")  # Run in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        # Initialize the driver
        driver = uc.Chrome(options=options, version_main=None)
        wait = WebDriverWait(driver, 20)

        # Load the YouTube video page
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Loading URL: {url}")
        driver.get(url)

        # Wait for the page to load (wait for the video player to be present)
        wait.until(EC.presence_of_element_located((By.ID, "movie_player")))
        time.sleep(2)  # Additional wait for page to stabilize

        # Try to click the "More actions" button (three dots below the video)
        # The button has an aria-label that contains "More actions"
        try:
            more_actions_button = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'More actions')]"))
            )
            more_actions_button.click()
            time.sleep(1)
        except TimeoutException:
            logger.warning(f"Could not find 'More actions' button for video {video_id}")
            # Fallback: try to open transcript directly via URL? Not reliable.
            # We'll try another approach: look for the transcript button in the menu below the video.

        # Try to click the "Open transcript" option in the menu
        try:
            # The transcript option is in the menu that appears after clicking "More actions"
            # We'll look for an item with the text "Open transcript"
            transcript_option = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//yt-formatted-string[text()='Open transcript']"))
            )
            transcript_option.click()
            time.sleep(2)
        except TimeoutException:
            logger.warning(f"Could not find 'Open transcript' option for video {video_id}")
            # If we can't open the transcript, we might try to get it from the API directly via the browser's network?
            # But for now, we'll return None.
            return None

        # Wait for the transcript panel to load
        try:
            # The transcript panel is contained in a div with id "body" inside the transcript renderer
            wait.until(
                EC.presence_of_element_located((By.ID, "body"))
            )
            time.sleep(1)
        except TimeoutException:
            logger.warning(f"Transcript panel did not load for video {video_id}")
            return None

        # Extract the transcript text from the panel
        # The transcript lines are in elements with class "cue-group" or similar
        # We'll look for all elements that contain the text of the transcript.
        transcript_lines = driver.find_elements(By.CSS_SELECTOR, "#body .cue-group")
        if not transcript_lines:
            # Fallback: try another selector
            transcript_lines = driver.find_elements(By.CSS_SELECTOR, "#body div")

        transcript_texts = []
        for line in transcript_lines:
            text = line.text.strip()
            if text:
                transcript_texts.append(text)

        if not transcript_texts:
            logger.warning(f"No transcript text found for video {video_id}")
            return None

        # Join the lines with a space
        full_text = " ".join(transcript_texts)
        logger.info(f"Successfully fetched transcript for {video_id} via Selenium ({len(full_text)} characters)")
        return full_text

    except Exception as e:
        logger.error(f"Error fetching transcript for video {video_id}: {e}")
        return None
    finally:
        if driver:
            driver.quit()


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
        description="Fetch YouTube transcripts using Selenium with undetected-chromedriver."
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
        help="Language code for transcripts (default: en)"
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

    # Parse arguments
    playlists_file = Path(args.playlists_file)
    output_dir = Path(args.output_dir) / args.channel_name
    channel_name = args.channel_name
    language_code = args.language
    max_videos = args.max_videos
    skip_existing = args.skip_existing

    logger.info("Starting YouTube transcript fetcher (Selenium method)")
    logger.info(f"Playlists file: {playlists_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Channel name: {channel_name}")
    logger.info(f"Language: {language_code}")

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
            transcript_text = fetch_transcript_with_selenium(video_id, language_code)
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
        "languages_attempted": [language_code],
        "method": "selenium-undetected-chromedriver"
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
