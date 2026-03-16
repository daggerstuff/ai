import json

# Input and output file paths
input_file_path = "/home/vivi/pixelated/ai/datasets/tier7/augesc/train.jsonl"
output_file_path = "/home/vivi/pixelated/ai/annotation/batches/batch_real_001.jsonl"


def process_file():
    records = []

    with open(input_file_path, "r") as infile:
        for line in infile:
            if not line.strip():
                continue

            try:
                # Parse the original JSON line
                original_data = json.loads(line)

                # The 'text' field is a string representation of a list of lists.
                # We need to parse this string into a Python object.
                # Example: "[[\"usr\", \"...\"], [\"sys\", \"...\"]]"
                conversation_raw = json.loads(original_data["text"])

                # Convert to the expected 'messages' format for our annotation agent
                messages = []
                for turn in conversation_raw:
                    role_map = {"usr": "user", "sys": "assistant"}
                    role = role_map.get(turn[0], "unknown")
                    content = turn[1]
                    messages.append({"role": role, "content": content})

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
                print(f"Skipping invalid JSON line: {line[:50]}...")
            except Exception as e:
                print(f"Error processing line: {e}")

    # Process a subset (e.g., first 50 records) for the batch
    subset_records = records[:50]

    # Write to the output file
    with open(output_file_path, "w") as outfile:
        for record in subset_records:
            outfile.write(json.dumps(record) + "\n")

    print(f"Successfully created {len(subset_records)} records in {output_file_path}")


if __name__ == "__main__":
    process_file()
