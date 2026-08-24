#!/usr/bin/env python3
"""
Monitor Multi-Dataset Processing Progress
Track the intelligent agent processing and prepare for next steps.
"""

import json
import subprocess
import time

from path_utils import get_unified_training_dir


def check_processing_status():
    """Check if the multi-dataset pipeline is still running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "multi_dataset_intelligent_pipeline"],
            capture_output=True,
            text=True,
        )
        return len(result.stdout.strip()) > 0
    except:
        return False


def check_output_progress():
    """Check progress in output directory"""

    output_dir = get_unified_training_dir()

    if not output_dir.exists():
        return {"status": "processing", "files": [], "total_conversations": 0}

    files = list(output_dir.glob("*.json"))
    total_conversations = 0

    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_conversations += len(data)
        except:
            continue

    return {
        "status": "completed" if any(f.name == "unified_lightning_config.json" for f in files) else "processing",
        "files": [f.name for f in files],
        "total_conversations": total_conversations,
    }


def main():

    while True:
        is_running = check_processing_status()
        progress = check_output_progress()

        if progress["files"]:
            pass

        if not is_running and progress["status"] == "completed":
            # Show final results

            config_file = get_unified_training_dir() / "unified_lightning_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

                config.get("dataset_stats", {}).get("processing_stats", {})

            break

        if not is_running:
            break

        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    main()
