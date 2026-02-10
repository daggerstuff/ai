import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def monitor(batch_id="batch_001"):
    log_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}.log")
    output_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}_annotated.jsonl")

    print(f"--- Monitoring {batch_id} ---")

    if log_file.exists():
        print("Log status: Active")
        # Efficiently read last line of log file
        try:
            file_size = Path(log_file).stat().st_size
            if file_size > 0:
                with open(log_file, "rb") as f:
                    f.seek(-min(500, file_size), 2)
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].decode().strip()
                        print(f"Latest status: {last_line}")
        except Exception:
            print("Could not read status from log file")
    else:
        print("Log status: NOT FOUND")

    if output_file.exists():
        count = 0
        last_line = None
        with open(output_file, "r") as f:
            for line in f:
                count += 1
                last_line = line
        # Process records more efficiently
        # Get total from first record or metadata if possible, else default to 5000
        TOTAL_ITEMS = 5000
        print(f"Completed items: {count} / {TOTAL_ITEMS}")

        if count > 0 and last_line:
            # Check agreement for last item
            _inspect_last_output(last_line)
    else:
        print("Output status: WAITING")


def _inspect_last_output(last_line: str):
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        print("⚠️ Latest output line is not valid JSON yet.")
        return

    agreement = data.get("agreement_metrics", {})
    print(
        f"Latest Agreement: Crisis={agreement.get('crisis_agreement', 0):.2f}, Emotion={agreement.get('emotion_agreement', 0):.2f}"
    )

    # Check for "No conversation data" bug again
    first_ann = data.get("individual_annotations", [{}])[0]
    if "No conversation data" in first_ann.get("notes", ""):
        print("🚨 ERROR: Data extraction bug still present!")
    else:
        print("✅ Data extraction: OK")


if __name__ == "__main__":
    monitor()
