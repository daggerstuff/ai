import argparse
import json
import random
from pathlib import Path


def prepare_batches(input_file, output_dir, batch_size=100, num_batches=1):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading from {input_path}...")
    data = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Total records found: {len(data)}")

    # Shuffle data
    random.shuffle(data)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        if start_idx >= len(data):
            break

        batch_data = data[start_idx:end_idx]

        # Ensure ID exists for tracking
        for idx, item in enumerate(batch_data):
            if (
                "id" not in item
                and "metadata" in item
                and "content_hash" in item["metadata"]
            ):
                item["id"] = item["metadata"]["content_hash"][:12]
            elif "id" not in item:
                item["id"] = f"sample_{start_idx + idx}"

        batch_filename = output_path / f"batch_{i + 1:03d}.jsonl"
        print(f"Writing {len(batch_data)} records to {batch_filename}...")

        with open(batch_filename, "w") as f:
            for item in batch_data:
                # We only need the input for annotation, but keeping full context
                # is good
                # Let's create a simplified 'annotation_task' structure wrapping
                # the item
                task = {"task_id": item["id"], "data": item, "annotations": []}
                f.write(json.dumps(task) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare annotation batches from JSONL dataset"
    )
    parser.add_argument("--input", required=True, help="Path to source JSONL file")
    parser.add_argument(
        "--output", default="../batches", help="Output directory for batches"
    )
    parser.add_argument("--size", type=int, default=50, help="Records per batch")
    parser.add_argument(
        "--count", type=int, default=2, help="Number of batches to generate"
    )

    args = parser.parse_args()

    prepare_batches(args.input, args.output, args.size, args.count)
