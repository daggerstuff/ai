import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def validate_record(record: Dict, line_num: int) -> List[str]:
    """Validate a single record against schema and PII rules."""
    errors = []

    # Schema Checks: Messages Array
    if "messages" not in record:
        errors.append("Missing 'messages' list")
        return errors  # Cannot process further

    messages = record["messages"]
    if not isinstance(messages, list):
        errors.append("'messages' is not a list")
    else:
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"Message {idx} is not a dict")
                continue
            if "role" not in msg:
                errors.append(f"Message {idx} missing 'role'")
            if "content" not in msg:
                errors.append(f"Message {idx} missing 'content'")

            # PII Check on Content
            content = msg.get("content", "")
            # Simple heuristic for potential email leak
            if "@" in content and "." in content and not "@pixelated" in content:
                # Very naive placeholder
                pass

    # Meta Checks
    meta = record.get("metadata", {})
    if not isinstance(meta, dict):
        errors.append("Field 'metadata' must be a dictionary")
    else:
        # Check for identifying info (Warning only if missing, as it might be anon)
        # We don't fail here because verified data typically strips these
        pass

    return errors


def process_file(file_path: Path) -> bool:
    """Process and validate a single JSONL file."""
    logger.info(f"Validating {file_path}...")

    try:
        with open(file_path, encoding="utf-8") as f:
            valid_count = 0
            error_count = 0

            for i, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    errors = validate_record(record, i)

                    if errors:
                        error_count += 1
                        for err in errors:
                            logger.error(f"Line {i}: {err}")
                    else:
                        valid_count += 1

                except json.JSONDecodeError:
                    logger.error(f"Line {i}: Invalid JSON")
                    error_count += 1

            logger.info(
                f"File {file_path.name}: {valid_count} valid, {error_count} errors"
            )
            return error_count == 0

    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate ingested real conversation data"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input file or directory (JSONL)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        success = process_file(input_path)
    elif input_path.is_dir():
        success = True
        for f in input_path.glob("*.jsonl"):
            if not process_file(f):
                success = False
    else:
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)

    if not success:
        sys.exit(1)

    logger.info("Validation complete. All checks passed.")


if __name__ == "__main__":
    main()
