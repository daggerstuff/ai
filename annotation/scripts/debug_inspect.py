import json
from pathlib import Path

files = list(Path("ai/annotation/results").glob("*.jsonl"))
print(f"Found files: {files}")

target_id = "crisis_9864"

for f in files:
    print(f"--- {f.name} ---")
    with open(f) as fp:
        for line in fp:
            data = json.loads(line)
            if data["task_id"] == target_id:
                print(f"Annotator: {data.get('annotator_id')}")
                print(f"Crisis Label: {data['annotations'].get('crisis_label')}")
