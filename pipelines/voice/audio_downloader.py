import concurrent.futures
import logging
import os
import sys

import yt_dlp

# Configuration
INPUT_FILE = ".notes/pixel/youtube.md"
OUTPUT_DIR = "data/voice_raw"
LOG_FILE = "logs/audio_downloader.log"
MAX_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def download_audio(url: str) -> None:
    """
    Download audio from a YouTube URL and save it as a WAV file.

    Args:
        url (str): The YouTube URL to download audio from.
    """
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{OUTPUT_DIR}/%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "noplaylist": True,
            "ignoreerrors": True,
            "retries": 3,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.info(f"Downloading: {url}")
            ydl.download([url])
            logging.info(f"Completed: {url}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")


def main() -> None:
    """
    Main function to download audio from URLs listed in the input file.
    Processes multiple downloads concurrently using ThreadPoolExecutor.
    """
    if not os.path.exists(INPUT_FILE):
        logging.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)
    with open(INPUT_FILE) as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]
    if not urls:
        logging.error("No valid URLs found in input file.")
        sys.exit(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(download_audio, urls)
    logging.info("All downloads complete.")


if __name__ == "__main__":
    main()
