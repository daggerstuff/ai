import json
from pathlib import Path

files = list(Path("ai/annotation/results").glob("*.jsonl"))

target_id = "crisis_9864"

for f in files:
    with open(f) as fp:
        for line in fp:
            data = json.loads(line)
            if data["task_id"] == target_id:
                pass
