#!/usr/bin/env python3
import argparse
import json
import os
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--source-type")
    parser.add_argument("--source-url")
    parser.add_argument("--acquired-at")
    args = parser.parse_args()

    records_changed = 0
    records = []
    with open(args.file) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "provenance" in record:
                records.append(record)
                continue
            record["provenance"] = {
                "source_type": args.source_type,
                "source_url": args.source_url,
                "license": "NOASSERTION",
                "transformations": ["legacy_jsonl_backfill"],
                "metadata": {"channel": record.get("source_channel", "unknown")},
            }
            if args.acquired_at:
                record["provenance"]["acquired_at"] = args.acquired_at
            records.append(record)
            records_changed += 1

    dirpath = os.path.dirname(os.path.abspath(args.file))
    with tempfile.NamedTemporaryFile(mode="w", dir=dirpath, suffix=".jsonl", delete=False) as tmp:
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp_path = tmp.name
    os.replace(tmp_path, args.file)



if __name__ == "__main__":
    main()
