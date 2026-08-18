#!/usr/bin/env python3
"""
Convert various dialogue JSONL datasets to plain text transcript format
expected by the TranscriptParser (speaker labels like "Speaker 1:", "Therapist:", etc.)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_google_synthetic_persona_chat(input_path: Path, output_dir: Path):
    """Convert google/Synthetic-Persona-Chat format to transcript."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                conversation = data.get("data", {}).get("Best Generated Conversation", "")
                if not conversation:
                    continue

                # Write as plain text with speaker labels
                output_file = output_dir / f"google_persona_chat_{count:05d}.txt"
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(conversation)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Converted {count} conversations from {input_path.name}")
    return count


def convert_hieunguyenminh_roleplay(input_path: Path, output_dir: Path):
    """Convert hieunguyenminh/roleplay format to transcript."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get("data", {}).get("text", "")
                if not text:
                    continue

                # Convert <|system|>, <|user|>, <|assistant|> to speaker labels
                text = text.replace("<|system|>", "System:")
                text = text.replace("<|user|>", "User:")
                text = text.replace("<|assistant|>", "Assistant:")

                output_file = output_dir / f"hieunguyenminh_roleplay_{count:05d}.txt"
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(text)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Converted {count} conversations from {input_path.name}")
    return count


def convert_nazlicanto_persona_chat(input_path: Path, output_dir: Path):
    """Convert nazlicanto/persona-based-chat format to transcript."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                dialogue = data.get("data", {}).get("dialogue", [])
                if not dialogue:
                    continue

                # Convert dialogue array to plain text
                lines = []
                for turn in dialogue:
                    if turn.startswith("Persona A:"):
                        lines.append(turn.replace("Persona A:", "Speaker 1:"))
                    elif turn.startswith("Persona B:"):
                        lines.append(turn.replace("Persona B:", "Speaker 2:"))
                    else:
                        lines.append(turn)

                if lines:
                    output_file = output_dir / f"nazlicanto_persona_chat_{count:05d}.txt"
                    with open(output_file, "w", encoding="utf-8") as out:
                        out.write("\n".join(lines))
                    count += 1
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Converted {count} conversations from {input_path.name}")
    return count


def convert_edge_case_dialogues(input_path: Path, output_dir: Path):
    """Convert edge_case_dialogues format to transcript."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                response = data.get("response", "")
                if not response:
                    continue

                # Already has Therapist: and Client: labels
                output_file = output_dir / f"edge_case_dialogue_{count:05d}.txt"
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(response)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to parse line: {e}")

    logger.info(f"Converted {count} conversations from {input_path.name}")
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert dialogue JSONL datasets to transcript format")
    parser.add_argument(
        "--input-dir", type=Path, default=Path("ai/data/voice_training"), help="Input directory with JSONL files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/training_data_unified/raw/transcripts/youtube"),
        help="Output directory for transcript .txt files",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    total = 0

    # Convert each dataset
    files_to_convert = [
        ("google_Synthetic-Persona-Chat.jsonl", convert_google_synthetic_persona_chat),
        ("hieunguyenminh_roleplay.jsonl", convert_hieunguyenminh_roleplay),
        ("nazlicanto_persona-based-chat.jsonl", convert_nazlicanto_persona_chat),
        ("edge_case_dialogues.jsonl", convert_edge_case_dialogues),
    ]

    for filename, converter in files_to_convert:
        input_path = args.input_dir / filename
        if input_path.exists():
            total += converter(input_path, args.output_dir)
        else:
            logger.warning(f"File not found: {input_path}")

    print("\nConversion Summary:")
    print(f"  Total transcripts created: {total}")


if __name__ == "__main__":
    main()
