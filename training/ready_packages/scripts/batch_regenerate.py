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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BatchRegenerate")

# Robust project root detection
_script_path = Path(__file__).resolve()
# We expect to be at ai/training/ready_packages/scripts/batch_regenerate.py
# If inside Docker /workspace/, project root is /workspace/
# If in local repo, project root is 5 levels up.

potential_roots = [
    _script_path.parents[4],  # Local development: repo root
    Path("/workspace"),  # Docker standard
    Path.cwd(),  # Fallback to CWD
]

for root in potential_roots:
    # 1. Flattened: root IS the 'ai' directory package content (contains 'core' and 'training')
    # We check this first to avoid being misled by an empty/incomplete 'ai/' subdirectory
    if (root / "core").is_dir() and (root / "training").is_dir():
        logger.info("Detected flattened AI directory at: %s", root)
        # Create a dynamic symlink so 'import ai.core' finds 'root/core'
        try:
            tmp_pkg_root = Path(tempfile.gettempdir()) / "pixelated_ai_pkg"
            tmp_pkg_root.mkdir(parents=True, exist_ok=True)
            ai_link = tmp_pkg_root / "ai"
            if ai_link.exists() and not ai_link.is_symlink():
                import shutil

                shutil.rmtree(ai_link)
            if not ai_link.exists():
                ai_link.symlink_to(root, target_is_directory=True)
            if str(tmp_pkg_root) not in sys.path:
                sys.path.insert(0, str(tmp_pkg_root))
            logger.info("Created package shim at: %s", ai_link)
            break
        except Exception as e:
            logger.warning("Failed to create package shim: %s", e)

    # 2. Standard: root contains 'ai' directory which actually contains 'core'
    elif (root / "ai" / "core").is_dir():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        logger.info("Found project root (standard): %s", root)
        break
else:
    logger.warning(
        "Could not definitively find project root. Current sys.path: %s", sys.path
    )

try:
    from ai.core.gestalt_simulator import GestaltSimulator
    from ai.utils.s3_dataset_loader import S3DatasetLoader
except ImportError as exc:
    logger.error(
        "Failed to import core modules. Ensure PYTHONPATH includes ai/. Error: %s",
        exc,
    )
    sys.exit(1)


def _extract_history(msgs):
    """Extract and format earlier chat history."""
    history = []
    for m in msgs:
        role = m.get("role", "unknown")
        if role == "system":
            continue
        speaker = "therapist" if role in ["user", "human"] else "client"
        history.append(
            {"speaker": speaker, "text": m.get("content", m.get("value", ""))}
        )
    return history


def _extract_multi_turn(msgs):
    """Extract logic from a multi-turn conversation format."""
    if not msgs or not isinstance(msgs, list) or len(msgs) < 2:
        return "", "", []

    last_msg = msgs[-1]
    prev_msg = msgs[-2]

    if last_msg.get("role") not in ["assistant", "gpt", "bot"]:
        return "", "", []

    if prev_msg.get("role") not in ["user", "human"]:
        return "", "", []

    original_response = last_msg.get("content", last_msg.get("value", ""))
    target = prev_msg.get("content", prev_msg.get("value", ""))
    history = _extract_history(msgs[:-2])

    return target, original_response, history


def _extract_single_turn(record):
    """Extract logic from a single-turn structured dataset format."""
    if "instruction" in record and "response" in record:
        return record["instruction"], record["response"]
    if "question" in record and "answer" in record:
        return record["question"], record["answer"]
    if "prompt" in record and "response" in record:
        return record["prompt"], record["response"]
    if "text" in record:
        return "Please respond to the following context.", record["text"]
    return "", ""


def _extract_turn_target_and_history(record):
    """Extract target prompt, original response, and chat history from a record."""
    msgs = record.get("messages") or record.get("conversations")
    target, original_response, history = _extract_multi_turn(msgs)

    if not target:
        target, original_response = _extract_single_turn(record)

    return target, original_response, history


def process_single_record(record, source_file, simulator):
    """Extract context, simulate a turn, and reformat a single JSON dataset record."""
    target, original_response, history = _extract_turn_target_and_history(record)

    if not target or not original_response:
        return None

    reasoning = ""
    if "metadata" in record and isinstance(record["metadata"], dict):
        reasoning = record["metadata"].get("reasoning", "")

    sim_input = target
    if reasoning:
        context_prefix = "Clinical Reasoning Context:"
        sim_input = f"{target}\n\n{context_prefix}\n{reasoning}"

    res = simulator.simulate_turn(history, sim_input)

    new_messages = [
        {
            "role": "user" if h["speaker"] == "therapist" else "assistant",
            "content": h["text"],
        }
        for h in history
    ]
    new_messages.extend(
        [
            {"role": "user", "content": target},
            {"role": "assistant", "content": res["new_response"]},
        ]
    )

    out_record = {
        "messages": new_messages,
        "metadata": {
            "source_file": source_file,
            "original_response": original_response,
            "gestalt_simulation": {
                "persona_id": res["persona_id"],
                "directive": res["directive_used"],
            },
        },
    }
    if reasoning:
        out_record["metadata"]["original_reasoning"] = reasoning

    return out_record


def _setup_simulator(args, loader, device):
    """Ensure defense model is present and initialize the GestaltSimulator."""
    model_path = Path(args.defense_model_path)
    if not model_path.exists():
        logger.info(
            "Defense model not found at %s. Attempting to download from s3://%s/%s",
            args.defense_model_path,
            args.s3_bucket,
            args.defense_model_s3_key,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            loader.download_file(args.defense_model_s3_key, str(model_path))
            logger.info("Defense model downloaded successfully.")
        except Exception as exc:
            logger.error("Failed to download defense model from S3: %s", exc)
            sys.exit(1)

    logger.info("Initializing GestaltSimulator on device '%s'...", device)
    return GestaltSimulator(defense_model_path=str(model_path), device=device)


def _process_records(args, loader, simulator, input_files):
    """Process records from input files and write to a temporary output file."""
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

                            out_record = process_single_record(
                                record, s3_file, simulator
                            )
                            if not out_record:
                                continue

                            temp_out.write(
                                json.dumps(out_record, ensure_ascii=False) + "\n"
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

    return temp_out_path, records_processed


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
            "final_dataset/shards/curriculum/stage2/synthetic_persona_batch_5k.jsonl"
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
        default="/workspace/checkpoints/model.ckpt",
        help="Local path where the defense model is (or will be) located.",
    )
    parser.add_argument(
        "--defense-model-s3-key",
        type=str,
        default="models/psydefdetect/psydef_deberta_v3_base.ckpt",
        help="S3 key for the defense model if not present locally.",
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
            "GEMINI_API_KEY is not set. Generation will fall back to mocked responses."
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

    simulator = _setup_simulator(args, loader, device)
    temp_out_path, records_processed = _process_records(
        args, loader, simulator, input_files
    )

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
        logger.warning("Generated dataset preserved locally at: %s", temp_out_path)
        sys.exit(1)

    os.remove(temp_out_path)
    logger.info("Batch Persona Re-Generation Job Complete.")


if __name__ == "__main__":
    main()
