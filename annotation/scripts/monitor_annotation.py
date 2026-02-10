import json
from pathlib import Path


def monitor(batch_id="batch_001"):
    log_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}.log")
    output_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}_annotated.jsonl")

    print(f"--- Monitoring {batch_id} ---")

    if log_file.exists():
        print("Log status: Active")
        # Get last line of log
        with open(log_file, "r") as f:
            if lines := f.readlines():
                print(f"Latest status: {lines[-1].strip()}")
    else:
        print("Log status: NOT FOUND")

    if output_file.exists():
        count = 0
        last_line = None
        with open(output_file, "r") as f:
            for line in f:
                count += 1
                last_line = line
        print(f"Completed items: {count} / 500")

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
