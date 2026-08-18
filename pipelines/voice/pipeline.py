#!/usr/bin/env python3
"""
Stage 4 Voice Persona Processing Pipeline.

Parses Tim Fletcher YouTube transcripts into dialogue turns for voice training.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single dialogue turn from a transcript."""

    speaker: str
    text: str
    source_file: str
    turn_index: int
    metadata: dict | None = None


class TranscriptParser:
    """Parses raw transcript text files into structured dialogue turns."""

    SPEAKER_PATTERNS = [
        re.compile(r"^(Speaker\s*\d+|Person\s*\d+|Client|Therapist|User|Assistant):\s*", re.IGNORECASE),
        re.compile(r"^([A-Z][a-z]*(?:\s+\d+)?):\s+"),
        re.compile(r"^\[([A-Z][a-z]*(?:\s+\d+)?)\]\s*"),
    ]

    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_transcript(self, file_path: Path) -> list[DialogueTurn]:
        """Parse a single transcript file into dialogue turns."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        lines = content.strip().split("\n")
        turns: list[DialogueTurn] = []
        current_speaker = "Unknown"
        current_text: list[str] = []
        turn_index = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            speaker_match = None
            for pattern in self.SPEAKER_PATTERNS:
                speaker_match = pattern.match(line)
                if speaker_match:
                    break

            if speaker_match:
                if current_text:
                    turns.append(
                        DialogueTurn(
                            speaker=current_speaker,
                            text=" ".join(current_text),
                            source_file=file_path.name,
                            turn_index=turn_index,
                        )
                    )
                    turn_index += 1
                    current_text = []

                current_speaker = speaker_match.group(1) if speaker_match.lastindex else "Unknown"
                remaining = line[speaker_match.end() :].strip()
                if remaining:
                    current_text.append(remaining)
            else:
                current_text.append(line)

        if current_text:
            turns.append(
                DialogueTurn(
                    speaker=current_speaker,
                    text=" ".join(current_text),
                    source_file=file_path.name,
                    turn_index=turn_index,
                )
            )

        return turns

    def process_all(self) -> dict[str, int]:
        """Process all transcript files in input directory."""
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return {}

        transcript_files = list(self.input_dir.glob("*.txt"))
        logger.info(f"Found {len(transcript_files)} transcript files")

        stats = {"files_processed": 0, "total_turns": 0, "errors": 0}

        for file_path in transcript_files:
            try:
                turns = self.parse_transcript(file_path)
                output_file = self.output_dir / f"{file_path.stem}_turns.jsonl"

                with open(output_file, "w", encoding="utf-8") as f:
                    for turn in turns:
                        record = {
                            "speaker": turn.speaker,
                            "text": turn.text,
                            "source_file": turn.source_file,
                            "turn_index": turn.turn_index,
                        }
                        f.write(json.dumps(record) + "\n")

                stats["files_processed"] += 1
                stats["total_turns"] += len(turns)
                logger.info(f"Processed {file_path.name}: {len(turns)} turns")

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Transcript parsing complete: {stats['files_processed']} files, "
            f"{stats['total_turns']} turns, {stats['errors']} errors"
        )
        return stats


def main() -> None:
    """CLI entry point for transcript parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse transcripts into dialogue turns")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ai/training_data_unified/raw/transcripts/youtube"),
        help="Directory containing transcript .txt files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/data/tim_fletcher_voice/turns"),
        help="Output directory for parsed turns",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser_obj = TranscriptParser(args.input_dir, args.output_dir)
    stats = parser_obj.process_all()

    print(f"\nTranscript Parsing Summary:")
    print(f"  Files processed: {stats.get('files_processed', 0)}")
    print(f"  Total turns: {stats.get('total_turns', 0)}")
    print(f"  Errors: {stats.get('errors', 0)}")


if __name__ == "__main__":
    main()
