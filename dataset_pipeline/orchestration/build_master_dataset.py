"""
Concatenates ULTIMATE_V5_EVERYTHING.jsonl + all v5_shards into one
deduplicated MASTER_TRAINING_SET.jsonl, streamed directly to S3.
"""

import hashlib
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset_pipeline.extractors.s3_streamer import S3Streamer

logger = logging.getLogger(__name__)

SOURCE_MAIN = "final_dataset/MASTER_TRAINING_SET_PREV.jsonl"
SOURCE_SHARDS_PREFIX = "final_dataset/v5_shards/"
OUTPUT_KEY = "final_dataset/MASTER_TRAINING_SET.jsonl"


def record_hash(record):
    """Hash the message content only — ignores metadata drift between runs."""
    msgs = record.get("messages", [])
    content = "".join(str(m.get("content") or "") for m in msgs)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    streamer = S3Streamer()
    seen = set()
    total_in = 0
    total_out = 0

    def deduplicated_stream():
        nonlocal total_in, total_out

        # 1. Stream the main V5 file
        logger.info("Streaming %s ...", SOURCE_MAIN)
        for record in streamer.stream_jsonl(SOURCE_MAIN):
            total_in += 1
            if total_in % 100000 == 0:
                logger.info(
                    "  Read %s | kept %s | dupes dropped %s",
                    f"{total_in:,}",
                    f"{total_out:,}",
                    f"{total_in - total_out:,}",
                )
            h = record_hash(record)
            if h not in seen:
                seen.add(h)
                total_out += 1
                yield record

        # 2. Stream all v5 shards (GDrive + Local run)
        shards = sorted(streamer.list_files(SOURCE_SHARDS_PREFIX))
        logger.info("\nFound %d shards to merge ...", len(shards))
        for shard_key in shards:
            logger.info("  Merging %s ...", shard_key)
            for record in streamer.stream_jsonl(shard_key):
                total_in += 1
                h = record_hash(record)
                if h not in seen:
                    seen.add(h)
                    total_out += 1
                    yield record

        logger.info(
            "\nStream complete. Total in: %s | Total unique out: %s | Dupes dropped: %s",
            f"{total_in:,}",
            f"{total_out:,}",
            f"{total_in - total_out:,}",
        )

    logger.info("Writing deduplicated master dataset \u2192 s3://pixeldata/%s", OUTPUT_KEY)
    streamer.write_jsonl(OUTPUT_KEY, deduplicated_stream())
    logger.info("\nDone! MASTER_TRAINING_SET.jsonl: %s records", f"{total_out:,}")


if __name__ == "__main__":
    main()
