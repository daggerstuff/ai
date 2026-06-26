"""
Concatenates ULTIMATE_V5_EVERYTHING.jsonl + all v5_shards into one
deduplicated MASTER_TRAINING_SET.jsonl, streamed directly to S3.
"""
import json
import hashlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dataset_pipeline.extractors.s3_streamer import S3Streamer

SOURCE_MAIN = "final_dataset/ULTIMATE_V5_EVERYTHING.jsonl"
SOURCE_SHARDS_PREFIX = "final_dataset/v5_shards/"
OUTPUT_KEY = "final_dataset/MASTER_TRAINING_SET.jsonl"

def record_hash(record):
    """Hash the message content only — ignores metadata drift between runs."""
    msgs = record.get("messages", [])
    content = "".join(m.get("content", "") for m in msgs)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def main():
    streamer = S3Streamer()
    seen = set()
    total_in = 0
    total_out = 0

    def deduplicated_stream():
        nonlocal total_in, total_out

        # 1. Stream the main V5 file
        print(f"Streaming {SOURCE_MAIN} ...")
        for record in streamer.stream_jsonl(SOURCE_MAIN):
            total_in += 1
            if total_in % 100000 == 0:
                print(f"  Read {total_in:,} | kept {total_out:,} | dupes dropped {total_in - total_out:,}")
            h = record_hash(record)
            if h not in seen:
                seen.add(h)
                total_out += 1
                yield record

        # 2. Stream all v5 shards (GDrive + Local run)
        shards = sorted(streamer.list_files(SOURCE_SHARDS_PREFIX))
        print(f"\nFound {len(shards)} shards to merge ...")
        for shard_key in shards:
            print(f"  Merging {shard_key} ...")
            for record in streamer.stream_jsonl(shard_key):
                total_in += 1
                h = record_hash(record)
                if h not in seen:
                    seen.add(h)
                    total_out += 1
                    yield record

        print(f"\nStream complete. Total in: {total_in:,} | Total unique out: {total_out:,} | Dupes dropped: {total_in - total_out:,}")

    print(f"Writing deduplicated master dataset → s3://pixeldata/{OUTPUT_KEY}")
    streamer.write_jsonl(OUTPUT_KEY, deduplicated_stream())
    print(f"\n✅ Done! MASTER_TRAINING_SET.jsonl: {total_out:,} records")

if __name__ == "__main__":
    main()
