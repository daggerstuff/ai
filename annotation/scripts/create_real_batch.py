import argparse
import json
import re
from pathlib import Path

DEFAULT_INPUT_FILE = Path("/home/vivi/pixelated/ai/datasets/tier7/augesc/train.jsonl")
DEFAULT_OUTPUT_FILE = Path("/home/vivi/pixelated/ai/annotation/batches/batch_real_001.jsonl")
DEFAULT_MAX_RECORDS = 100
ROLE_MAP = {
    "usr": "user",
    "user": "user",
    "user:": "user",
    "client": "user",
    "assistant": "assistant",
    "sys": "assistant",
    "therapist": "user",
    "bot": "assistant",
}


def _coerce_turn_messages(raw_text: str) -> list[dict[str, str]]:
    """
    Convert raw conversation text into an annotation-friendly message list.
    """
    if not isinstance(raw_text, str):
        return []

    content = raw_text.strip()
    if not content:
        return []

    # Legacy format: '[["usr","..."], ["sys","..."]]'.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and parsed:
            messages = []
            for turn in parsed:
                if not isinstance(turn, (list, tuple)) or len(turn) < 2:
                    continue
                role = ROLE_MAP.get(str(turn[0]).strip().lower(), "user")
                messages.append({"role": role, "content": str(turn[1])})
            if messages:
                return messages
    except json.JSONDecodeError:
        pass

    return []


def process_file(
    input_file_path: str | Path,
    output_file_path: str | Path,
    max_records: int = DEFAULT_MAX_RECORDS,
):
    input_path = Path(input_file_path)
    output_path = Path(output_file_path)
    records = []

    with open(input_path) as infile:
        for line in infile:
            if not line.strip():
                continue

            try:
                # Parse the original JSON line.
                original_data = json.loads(line)

                messages: list[dict[str, str]] = []
                if "messages" in original_data:
                    raw_messages = original_data.get("messages", [])
                    if isinstance(raw_messages, list):
                        for raw_message in raw_messages:
                            if not isinstance(raw_message, dict):
                                continue
                            role = str(raw_message.get("role", "user")).lower()
                            content = str(raw_message.get("content", "")).strip()
                            if not role or not content:
                                continue
                            messages.append({"role": role, "content": content})
                else:
                    # Prefer explicit prompt/response pairs if present.
                    prompt = original_data.get("prompt")
                    response = original_data.get("response")
                    if isinstance(prompt, str) and isinstance(response, str):
                        messages = [
                            {"role": "user", "content": prompt.strip()},
                            {"role": "assistant", "content": response.strip()},
                        ]

                    # Legacy/alternate text formats.
                    if not messages:
                        messages = _coerce_turn_messages(original_data.get("text"))

                        if not messages:
                            conversation_text = str(original_data.get("text", ""))
                            turn_lines = [line.strip() for line in conversation_text.splitlines() if line.strip()]
                            for turn_line in turn_lines:
                                match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_\\-\\s]*?)\s*:\s*(.+)$", turn_line)
                                if not match:
                                    continue
                                raw_role = match.group(1).strip().lower()
                                content = match.group(2).strip()
                                role = ROLE_MAP.get(raw_role, "user")
                                if content:
                                    messages.append({"role": role, "content": content})

                if not messages:
                    continue

                # Create the new record structure
                record = {
                    "task_id": f"real_{len(records):05d}",
                    "data": {
                        "id": f"real_{len(records):05d}",
                        "messages": messages,
                        "dataset": "augesc_train",
                    },
                    "annotations": [],
                }
                records.append(record)

            except json.JSONDecodeError:
                pass
            except Exception:
                pass

    # Process a larger subset for the batch, defaulting to a Phase 1.3 target of 100.
    subset_records = records[:max_records]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to the output file
    with open(output_path, "w") as outfile:
        for record in subset_records:
            outfile.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an annotation batch JSONL from the raw augesc train split.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Source train.jsonl path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output batch JSONL path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Maximum number of records to include in the batch.",
    )

    args = parser.parse_args()
    process_file(
        input_file_path=args.input,
        output_file_path=args.output,
        max_records=args.max_records,
    )
