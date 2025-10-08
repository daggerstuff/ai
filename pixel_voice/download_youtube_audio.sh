#!/bin/bash
# Download audio from all YouTube URLs in .notes/pixel/youtube.md using yt-dlp

INPUT_FILE=".notes/pixel/youtube.md"
OUTPUT_DIR="data/voice_raw"

mkdir -p "$OUTPUT_DIR"

echo "Starting download of YouTube audio files..."

while read -r url; do
    # Skip empty lines or lines that do not look like URLs
    if [[ "$url" =~ ^https?:// ]]; then
        echo "Downloading: $url"
        yt-dlp -x --audio-format wav --audio-quality 0 -o "$OUTPUT_DIR/%(title)s.%(ext)s" "$url"
    fi
done < "$INPUT_FILE"

echo "All downloads complete. Audio files saved to $OUTPUT_DIR"