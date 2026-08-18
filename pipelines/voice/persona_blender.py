#!/usr/bin/env python3
"""
Stage 4 Voice Persona Blender.

Injects Tim Fletcher therapeutic tone metadata and voice signature tokens
into dialogue pairs for voice training data generation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class VoiceSignatureToken:
    """A voice signature token with metadata."""

    token: str
    tone_category: str
    intensity: float
    therapeutic_technique: str | None = None


@dataclass
class DialoguePair:
    """A dialogue pair with voice persona metadata."""

    prompt: str
    response: str
    voice_tokens: list[VoiceSignatureToken]
    therapeutic_tone: dict
    source_file: str
    pair_index: int


class PersonaBlender:
    """Blends Tim Fletcher voice persona into dialogue pairs."""

    THERAPEUTIC_TONE_MARKERS = {
        "reflective_listening": [
            r"what I'm hearing is",
            r"it sounds like",
            r"if I understand correctly",
            r"you're saying that",
        ],
        "validation": [
            r"that makes sense",
            r"I can see why",
            r"it's understandable",
            r"your feelings are valid",
        ],
        "empathy": [
            r"I can imagine that",
            r"that must be",
            r"I hear the pain",
            r"that's really difficult",
        ],
        "curiosity": [
            r"can you tell me more",
            r"what was that like",
            r"how did that feel",
            r"what do you think",
        ],
        "reframing": [
            r"another way to look at it",
            r"consider the possibility",
            r"what if we",
            r"perhaps",
        ],
    }

    VOICE_SIGNATURE_TOKENS = {
        "warmth": ["gentle", "compassionate", "caring", "supportive"],
        "curiosity": ["curious", "exploratory", "inquisitive", "open"],
        "validation": ["validating", "affirming", "acknowledging", "accepting"],
        "reflection": ["reflective", "thoughtful", "considered", "mindful"],
    }

    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_therapeutic_tone(self, text: str) -> dict[str, float]:
        """Detect therapeutic tone markers in text."""
        tone_scores: dict[str, float] = {}

        for tone, patterns in self.THERAPEUTIC_TONE_MARKERS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1.0
            tone_scores[tone] = score

        return tone_scores

    def extract_voice_tokens(self, text: str) -> list[VoiceSignatureToken]:
        """Extract voice signature tokens from text."""
        tokens: list[VoiceSignatureToken] = []
        text_lower = text.lower()

        for category, words in self.VOICE_SIGNATURE_TOKENS.items():
            for word in words:
                if word in text_lower:
                    tokens.append(
                        VoiceSignatureToken(
                            token=word,
                            tone_category=category,
                            intensity=0.8,
                            therapeutic_technique=category,
                        )
                    )

        return tokens

    def create_dialogue_pairs(self, turns_file: Path) -> list[DialoguePair]:
        """Create dialogue pairs from parsed turns."""
        if not turns_file.exists():
            logger.error(f"Turns file not found: {turns_file}")
            return []

        turns: list[dict] = []
        with open(turns_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))

        pairs: list[DialoguePair] = []
        pair_index = 0

        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]

            if current_turn["speaker"] != next_turn["speaker"]:
                prompt = current_turn["text"]
                response = next_turn["text"]

                tone = self.detect_therapeutic_tone(response)
                tokens = self.extract_voice_tokens(response)

                pairs.append(
                    DialoguePair(
                        prompt=prompt,
                        response=response,
                        voice_tokens=tokens,
                        therapeutic_tone=tone,
                        source_file=turns_file.name,
                        pair_index=pair_index,
                    )
                )
                pair_index += 1

        return pairs

    def process_all(self) -> dict[str, int]:
        """Process all turns files and generate voice training pairs."""
        if not self.input_dir.exists():
            logger.error(f"Input directory not found: {self.input_dir}")
            return {}

        turns_files = list(self.input_dir.glob("*_turns.jsonl"))
        logger.info(f"Found {len(turns_files)} turns files")

        stats = {"files_processed": 0, "total_pairs": 0, "errors": 0}

        for turns_file in turns_files:
            try:
                pairs = self.create_dialogue_pairs(turns_file)
                output_file = self.output_dir / f"{turns_file.stem.replace('_turns', '')}_voice_pairs.jsonl"

                with open(output_file, "w", encoding="utf-8") as f:
                    for pair in pairs:
                        record = {
                            "prompt": pair.prompt,
                            "response": pair.response,
                            "voice_tokens": [
                                {
                                    "token": t.token,
                                    "tone_category": t.tone_category,
                                    "intensity": t.intensity,
                                    "therapeutic_technique": t.therapeutic_technique,
                                }
                                for t in pair.voice_tokens
                            ],
                            "therapeutic_tone": pair.therapeutic_tone,
                            "source_file": pair.source_file,
                            "pair_index": pair.pair_index,
                        }
                        f.write(json.dumps(record) + "\n")

                stats["files_processed"] += 1
                stats["total_pairs"] += len(pairs)
                logger.info(f"Processed {turns_file.name}: {len(pairs)} voice pairs")

            except Exception as e:
                logger.error(f"Failed to process {turns_file.name}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Persona blending complete: {stats['files_processed']} files, "
            f"{stats['total_pairs']} pairs, {stats['errors']} errors"
        )
        return stats


def main() -> None:
    """CLI entry point for persona blending."""
    import argparse

    parser = argparse.ArgumentParser(description="Blend voice persona into dialogue pairs")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ai/data/tim_fletcher_voice/turns"),
        help="Directory containing parsed turns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/data/tim_fletcher_voice/pairs"),
        help="Output directory for voice pairs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    blender = PersonaBlender(args.input_dir, args.output_dir)
    stats = blender.process_all()

    print("\nPersona Blending Summary:")
    print(f"  Files processed: {stats.get('files_processed', 0)}")
    print(f"  Total pairs: {stats.get('total_pairs', 0)}")
    print(f"  Errors: {stats.get('errors', 0)}")


if __name__ == "__main__":
    main()
