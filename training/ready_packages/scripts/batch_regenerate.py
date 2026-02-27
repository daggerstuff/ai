#!/usr/bin/env python3
"""
batch_regenerate.py - PIX-148 Persona Re-Generation GPU Batch Job

Entry point for the OVH Phase 3 training job. Reads Stage 2 dataset from S3,
streams it through the GestaltSimulator (GPU-accelerated PsyDefDetect + Gemini
API), re-generates assistant responses to include defense-aware persona
behaviors, and uploads 5000 records back to S3.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BatchRegenerate")

try:
    from core.gestalt_simulator import GestaltSimulator
    from utils.s3_dataset_loader import S3DatasetLoader
except ImportError as exc:
    logger.error(
        "Failed to import core modules. Ensure PYTHONPATH includes ai/. Error: %s",
        exc,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GestaltSimulator batch regeneration pipeline."
    )
    parser.add_argument(
        "--input-s3-prefix",
        type=str,
        default="final_dataset/shards/curriculum/stage2/",
        help="S3 prefix to source the input dataset shards from.",
    )
    parser.add_argument(
        "--output-s3-key",
        type=str,
        default=(
            "final_dataset/shards/curriculum/stage2/"
            "synthetic_persona_batch_5k.jsonl"
        ),
        help="S3 key where the augmented dataset will be saved.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=5000,
        help="Maximum number of records to re-generate.",
    )
    parser.add_argument(
        "--defense-model-path",
        type=str,
        required=True,
        help=(
            "Local path to the PsyDefDetect checkpoint "
            "(e.g., /workspace/checkpoints/model.ckpt)"
        ),
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default="pixel-data",
        help="The OVH Object Storage bucket to use.",
    )

    args = parser.parse_args()

    if not os.environ.get("OVH_S3_ACCESS_KEY") or not os.environ.get(
        "OVH_S3_SECRET_KEY"
    ):
        logger.warning(
            "OVH_S3_ACCESS_KEY / OVH_S3_SECRET_KEY not set. "
            "S3 operations will fail if not using instance roles."
        )

    if not os.environ.get("GEMINI_API_KEY"):
        logger.error(
            "GEMINI_API_KEY is not set. "
            "Generation will fall back to mocked responses."
        )

    loader = S3DatasetLoader(bucket=args.s3_bucket)

    try:
        input_files = loader.list_datasets(prefix=args.input_s3_prefix)
    except Exception as exc:
        logger.error("Failed to list S3 datasets at %s: %s", args.input_s3_prefix, exc)
        sys.exit(1)

    if not input_files:
        logger.error(
            "No files found in s3://%s/%s", args.s3_bucket, args.input_s3_prefix
        )
        sys.exit(1)

    logger.info(
        "Found %d dataset file(s). Will pull until %d records are processed.",
        len(input_files),
        args.max_records,
    )

    device = (
        "cuda"
        if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/dev/nvidia0")
        else "cpu"
    )
    logger.info("Initializing GestaltSimulator on device '%s'...", device)
    simulator = GestaltSimulator(
        defense_model_path=args.defense_model_path, device=device
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as temp_out:
        temp_out_path = temp_out.name
        logger.info("Writing intermediate results to %s", temp_out_path)

        records_processed = 0

        for s3_file in input_files:
            if records_processed >= args.max_records:
                break

            logger.info("Processing source file: %s", s3_file)

            if not s3_file.endswith(".jsonl"):
                logger.warning("Skipping non-JSONL file: %s", s3_file)
                continue

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_in:
                tmp_in_path = tmp_in.name
                loader.download_file(s3_file, tmp_in_path)

            try:
                with open(tmp_in_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        if records_processed >= args.max_records:
                            break

                        try:
                            record = json.loads(line)
                            messages = record.get("messages", [])

                            if len(messages) < 3:
                                continue
                            if (
                                messages[-1]["role"] != "assistant"
                                or messages[-2]["role"] != "user"
                            ):
                                continue

                            target = messages[-2]["content"]

                            history = []
                            for msg in messages[:-2]:
                                if msg["role"] == "system":
                                    continue
                                history.append(
                                    {
                                        "speaker": (
                                            "therapist"
                                            if msg["role"] == "user"
                                            else "client"
                                        ),
                                        "text": msg["content"],
                                    }
                                )

                            res = simulator.simulate_turn(history, target)

                            record["messages"][-1]["content"] = res["new_response"]
                            record.setdefault("metadata", {})["gestalt_simulation"] = {
                                "persona_id": res["persona_id"],
                                "directive": res["directive_used"],
                            }

                            temp_out.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                            records_processed += 1

                            if records_processed % 100 == 0:
                                logger.info(
                                    "Processed %d/%d records...",
                                    records_processed,
                                    args.max_records,
                                )
                        except Exception as exc:
                            logger.error(
                                "Error on record %d: %s", records_processed, exc
                            )
            finally:
                os.remove(tmp_in_path)

    logger.info(
        "Finished generating %d records. Uploading to s3://%s/%s",
        records_processed,
        args.s3_bucket,
        args.output_s3_key,
    )
    try:
        loader.upload_file(temp_out_path, args.output_s3_key)
        logger.info("Upload complete!")
    except Exception as exc:
        logger.error("Failed to upload to S3: %s", exc)
        logger.warning(
            "Generated dataset preserved locally at: %s", temp_out_path
        )
        sys.exit(1)

    os.remove(temp_out_path)
    logger.info("Batch Persona Re-Generation Job Complete.")


if __name__ == "__main__":
    main()
