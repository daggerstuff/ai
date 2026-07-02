import json
import shutil
from pathlib import Path

source_dir = Path("tmp/youtube_ingestion_final")
target_dir = Path("data/therapeutic")

target_dir.mkdir(parents=True, exist_ok=True)

for file_path in source_dir.glob("*.jsonl"):
    target_file = target_dir / file_path.name
    with open(file_path, encoding="utf-8") as f_in, open(target_file, "w", encoding="utf-8") as f_out:
        for raw_line in f_in:
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            data = json.loads(stripped_line)

            chatml_data = {
                "messages": [
                    {"role": "user", "content": data.get("instruction", "")},
                    {"role": "assistant", "content": data.get("output", "")},
                ]
            }

            # Carry over metadata
            for key in ["language", "provenance", "source_channel"]:
                if key in data:
                    chatml_data[key] = data[key]

            f_out.write(json.dumps(chatml_data, ensure_ascii=False) + "\n")

# Copy the processing report and manifest
for file_name in ["processing_report.json", "manifest.json"]:
    src = source_dir / file_name
    if src.exists():
        shutil.copy2(src, target_dir / file_name)

print("Finished converting and moving files to data/therapeutic/")  # noqa: T201
