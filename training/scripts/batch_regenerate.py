#!/usr/bin/env python3
"""
batch_regenerate.py - PIX-148 Persona Re-Generation GPU Batch Job

Entry point for the OVH Phase 3 training job. Reads Stage 2 dataset from S3,
streams it through the GestaltSimulator (GPU-accelerated PsyDefDetect + Gemini
API), re-generates assistant responses to include defense-aware persona
behaviors, and uploads the configured number of valid regenerated records back
to S3.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.core.gestalt_simulator import GestaltSimulator
from ai.core.utils.s3_dataset_loader import S3DatasetLoader
from ai.core.validation.persona_quality import (
    _fails_human_likeness,
    _is_refusal_or_fallback,
    _stable_message_hash,
    last_assistant_content,
    validate_record,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("BatchRegenerate")


# Robust project root detection for mixed packaging contexts
_script_path = Path(__file__).resolve()
potential_roots = [Path("/app"), _script_path.parents[3], Path.cwd()]
for root in potential_roots:
    if (root / "ai" / "core").is_dir():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        break


def _extract_history(msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract and format earlier chat history."""
    history: list[dict[str, str]] = []
    for m in msgs:
        role = m.get("role", "unknown")
        if role == "system":
            continue
        speaker = "therapist" if role in ["user", "human"] else "client"
        history.append({"speaker": speaker, "text": m.get("content", m.get("value", ""))})
    return history


def _extract_multi_turn(msgs: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, str]]]:
    """Extract a target + response pair from multi-turn conversation records."""
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


def _extract_single_turn(record: dict[str, Any]) -> tuple[str, str]:
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


def _extract_turn_target_and_history(
    record: dict[str, Any],
) -> tuple[str, str, list[dict[str, str]]]:
    """Extract target prompt, original response, and chat history from a record."""
    msgs = record.get("messages") or record.get("conversations", [])
    target, original_response, history = _extract_multi_turn(msgs)

    if not target:
        target, original_response = _extract_single_turn(record)

    return target, original_response, history


def _build_source_record(
    record: dict[str, Any],
    source_file: str,
    target: str,
    original_response: str,
    history: list[dict[str, str]],
    res: dict[str, Any],
    reasoning: str,
) -> dict[str, Any]:
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


@dataclass
class ProcessResult:
    record: dict[str, Any] | None
    reason: str
    attempts: int
    assistant_hash: str | None
    persona_id: str | None


def process_single_record(
    record: dict[str, Any],
    source_file: str,
    simulator: GestaltSimulator,
    *,
    line_number: int,
    max_retries: int = 1,
    persona_id: str | None = None,
) -> ProcessResult:
    """Extract context, simulate a turn, and return a validated record payload."""
    target, original_response, history = _extract_turn_target_and_history(record)
    if not target or not original_response:
        return ProcessResult(None, "missing target or original response", 0, None, None)

    reasoning = ""
    if isinstance(record.get("metadata"), dict):
        reasoning = record["metadata"].get("reasoning", "")

    sim_input = target
    if reasoning:
        sim_input = f"{target}\n\nClinical Reasoning Context:\n{reasoning}"

    for attempt in range(max_retries + 1):
        tries = attempt + 1
        try:
            res = simulator.simulate_turn(
                history,
                sim_input,
                persona_id_hint=persona_id,
            )
        except Exception as exc:
            if attempt < max_retries:
                continue
            return ProcessResult(None, f"simulation failure: {exc}", tries, None, None)

        out_record = _build_source_record(
            record,
            source_file,
            target,
            original_response,
            history,
            res,
            reasoning,
        )

        errors: list[str] = validate_record(out_record, line_number)
        generated = out_record["messages"][-1]["content"]
        if _is_refusal_or_fallback(generated):
            errors.append(f"line {line_number}: generated refusal/fallback text")
        if _fails_human_likeness(generated):
            errors.append(f"line {line_number}: generated text failed human-likeness")

        if not errors:
            return ProcessResult(
                out_record,
                "",
                tries,
                _stable_message_hash(generated),
                res.get("persona_id"),
            )

        if attempt < max_retries:
            logger.debug(
                "Record validation retry for line %s (attempt=%s). Errors=%s",
                line_number,
                tries,
                errors,
            )
            continue

        return ProcessResult(None, "; ".join(errors), tries, None, None)

    return ProcessResult(None, "unreachable", max_retries + 1, None, None)


def _setup_simulator(args: argparse.Namespace, loader: S3DatasetLoader, device: str) -> GestaltSimulator:
    """Ensure model presence (optional) and init simulator."""
    model_path = Path(args.defense_model_path) if args.defense_model_path else None

    if model_path and not model_path.exists():
        download_uri = args.defense_model_s3_key
        if not download_uri.startswith("s3://"):
            download_uri = f"s3://{args.s3_bucket}/{download_uri}"

        logger.info(
            "Defense model not found at %s. Attempting to download from %s",
            args.defense_model_path,
            download_uri,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            loader.download_file(args.defense_model_s3_key, str(model_path))
            logger.info("Defense model downloaded successfully.")
        except Exception as exc:
            logger.error("Failed to download defense model from S3: %s", exc)
            logger.info(
                "Proceeding without local model (will fallback to NIM/default directives)."
            )
            model_path = None

    logger.info("Initializing GestaltSimulator on device '%s'...", device)
    return GestaltSimulator(
        defense_model_path=str(model_path) if model_path else None,
        device=device,
        nim_only=args.nim_only,
    )


def _resolve_checkpoint_s3_key(args: argparse.Namespace) -> str:
    if args.checkpoint_s3_key:
        return args.checkpoint_s3_key
    prefix = (args.checkpoint_prefix or "checkpoints/persona-regeneration").rstrip("/")
    return f"{prefix}/{args.checkpoint_job_name}.json"


def _load_checkpoint_state(
    loader: S3DatasetLoader, checkpoint_s3_key: str | None
) -> dict[str, Any]:
    if not checkpoint_s3_key:
        return {}
    state_path = None
    try:
        if not loader.object_exists(checkpoint_s3_key):
            return {}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            state_path = tmp.name
        loader.download_file(checkpoint_s3_key, state_path)
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load checkpoint state from %s.", checkpoint_s3_key)
        return {}
    finally:
        if state_path:
            os.remove(state_path)


def _save_checkpoint(
    loader: S3DatasetLoader,
    checkpoint_s3_key: str | None,
    state: dict[str, Any],
) -> None:
    if not checkpoint_s3_key:
        return
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(state, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    try:
        logger.info("Uploading checkpoint to s3://%s", checkpoint_s3_key)
        loader.upload_file(tmp_path, checkpoint_s3_key)
    finally:
        os.remove(tmp_path)


def _load_output_state(
    output_s3_key: str, loader: S3DatasetLoader
) -> tuple[int, set[str], dict[str, int]]:
    output_written = 0
    seen_hashes: set[str] = set()
    persona_counts: dict[str, int] = {}

    if not loader.object_exists(output_s3_key):
        return output_written, seen_hashes, persona_counts

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        output_path = tmp.name
    try:
        loader.download_file(output_s3_key, output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                output_written += 1
                assistant_text = last_assistant_content(rec)
                if assistant_text:
                    seen_hashes.add(_stable_message_hash(assistant_text))
                gs = (rec.get("metadata") or {}).get("gestalt_simulation") or {}
                persona_id = gs.get("persona_id") or "unknown"
                persona_counts[persona_id] = persona_counts.get(persona_id, 0) + 1
        return output_written, seen_hashes, persona_counts
    finally:
        os.remove(output_path)


def _select_persona_id(
    persona_counts: Counter[str],
    total_written: int,
    all_personas: list[str],
    max_fraction: float,
) -> str | None:
    if max_fraction <= 0:
        return None
    if total_written == 0:
        return None

    overrepresented = {
        pid for pid, count in persona_counts.items() if count / total_written > max_fraction
    }
    if len(overrepresented) >= len(all_personas):
        return None

    candidates = [pid for pid in all_personas if pid not in overrepresented]
    if not candidates:
        return None
    return random.choice(candidates)


def _make_checkpoint_payload(
    args: argparse.Namespace,
    written_count: int,
    skipped_count: int,
    retry_count: int,
    duplicate_count: int,
    source_file_index: int,
    source_line_index: int,
    seen_hashes: set[str],
    persona_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "target_records": args.max_records,
        "written_count": written_count,
        "skipped_count": skipped_count,
        "retry_count": retry_count,
        "duplicate_count": duplicate_count,
        "source_file_index": source_file_index,
        "source_line_index": source_line_index,
        "persona_counts": dict(persona_counts),
        "seen_hashes": list(sorted(seen_hashes)),
        "output_s3_key": args.output_s3_key,
        "checkpoint_job_name": args.checkpoint_job_name,
    }


def _process_records(
    args: argparse.Namespace,
    loader: S3DatasetLoader,
    simulator: GestaltSimulator,
    input_files: list[str],
) -> tuple[str, int]:
    """Process records from input files and write valid results to a temp file."""
    max_retries = max(0, args.retry_on_validation_failure)
    checkpoint_key = _resolve_checkpoint_s3_key(args)
    checkpoint_state = _load_checkpoint_state(loader, checkpoint_key) if args.resume else {}

    resume_written, resume_hashes, resume_persona_counts = (
        _load_output_state(args.output_s3_key, loader) if args.resume else (0, set(), {})
    )

    written_count = max(int(checkpoint_state.get("written_count", 0)), resume_written)
    skipped_count = int(checkpoint_state.get("skipped_count", 0))
    retry_count = int(checkpoint_state.get("retry_count", 0))
    duplicate_count = int(checkpoint_state.get("duplicate_count", 0))
    start_file_index = int(checkpoint_state.get("source_file_index", 0)) if args.resume else 0
    start_line_index = int(checkpoint_state.get("source_line_index", 0)) if args.resume else 0

    seen_hashes = set(resume_hashes)
    seen_hashes.update(checkpoint_state.get("seen_hashes", []))

    persona_counts = Counter(resume_persona_counts)
    for pid, cnt in checkpoint_state.get("persona_counts", {}).items():
        persona_counts[pid] = max(persona_counts.get(pid, 0), int(cnt))

    all_personas = simulator.persona_manager.get_available_archetypes()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as temp_out:
        temp_out_path = temp_out.name

    seed_written = 0
    if args.resume and loader.object_exists(args.output_s3_key):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as seed_file:
            seed_output_path = seed_file.name
        loader.download_file(args.output_s3_key, seed_output_path)
        with open(seed_output_path, "r", encoding="utf-8") as src, open(
            temp_out_path, "w", encoding="utf-8"
        ) as dst:
            for line in src:
                dst.write(line)
        seed_written = resume_written
        os.remove(seed_output_path)

    source_file_index = 0
    source_line_index = 0

    # Keep this script simple and safe for checkpointed restarts by using sequential
    # processing. This preserves exact write semantics for dedupe and cursor state.
    with open(temp_out_path, "a", encoding="utf-8") as out_fh:
        for file_idx, s3_file in enumerate(input_files):
            if written_count >= args.max_records:
                break
            if start_file_index and file_idx < start_file_index:
                continue

            if not s3_file.endswith(".jsonl"):
                logger.warning("Skipping non-JSONL file: %s", s3_file)
                continue

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_in:
                tmp_in_path = tmp_in.name
                loader.download_file(s3_file, tmp_in_path)

            try:
                logger.info("Processing source file: %s", s3_file)
                with open(tmp_in_path, "r", encoding="utf-8") as infile:
                    for line_no, line in enumerate(infile, start=1):
                        source_file_index = file_idx
                        source_line_index = line_no + 1

                        if file_idx == start_file_index and line_no <= start_line_index:
                            continue

                        if written_count >= args.max_records:
                            break
                        if not line.strip():
                            continue

                        try:
                            record = json.loads(line)
                        except Exception as exc:
                            skipped_count += 1
                            logger.error(
                                "Error parsing source record %s:%s: %s",
                                s3_file,
                                line_no,
                                exc,
                            )
                            continue

                        selected_persona_id = _select_persona_id(
                            persona_counts,
                            written_count,
                            all_personas,
                            args.persona_max_fraction,
                        )

                        result = process_single_record(
                            record,
                            s3_file,
                            simulator,
                            line_number=line_no,
                            max_retries=max_retries,
                            persona_id=selected_persona_id,
                        )
                        retry_count += max(0, result.attempts - 1)

                        if not result.record:
                            skipped_count += 1
                            logger.debug(
                                "Skipping record from %s:%s: %s",
                                s3_file,
                                line_no,
                                result.reason,
                            )
                            continue

                        if result.assistant_hash in seen_hashes:
                            duplicate_count += 1
                            logger.debug(
                                "Skipping duplicate assistant message hash for %s:%s",
                                s3_file,
                                line_no,
                            )
                            continue

                        out_fh.write(
                            json.dumps(result.record, ensure_ascii=False) + "\n"
                        )
                        seen_hashes.add(result.assistant_hash)
                        written_count += 1
                        if result.persona_id:
                            persona_counts[result.persona_id] += 1

                        if (
                            args.checkpoint_frequency > 0
                            and written_count % args.checkpoint_frequency == 0
                        ):
                            _save_checkpoint(
                                loader,
                                checkpoint_key,
                                _make_checkpoint_payload(
                                    args,
                                    written_count,
                                    skipped_count,
                                    retry_count,
                                    duplicate_count,
                                    source_file_index,
                                    source_line_index,
                                    seen_hashes,
                                    persona_counts,
                                ),
                            )

                        if written_count % 100 == 0:
                            logger.info(
                                "Processed %d/%d records...",
                                written_count,
                                args.max_records,
                            )
            finally:
                os.remove(tmp_in_path)

        _save_checkpoint(
            loader,
            checkpoint_key,
            _make_checkpoint_payload(
                args,
                written_count,
                skipped_count,
                retry_count,
                duplicate_count,
                source_file_index,
                0 if written_count >= args.max_records else source_line_index,
                seen_hashes,
                persona_counts,
            ),
        )

    if written_count < args.max_records:
        logger.warning(
            "Input data exhausted after producing %d valid records. Target was %d.",
            written_count,
            args.max_records,
        )

    logger.info(
        "Finished generation stats: written=%s skipped=%s retries=%s duplicates=%s seeded=%s",
        written_count - seed_written,
        skipped_count,
        retry_count,
        duplicate_count,
        seed_written,
    )
    return temp_out_path, written_count


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
        default="final_dataset/shards/curriculum/stage2/synthetic_persona_batch_10000.jsonl",
        help="S3 key where the augmented dataset will be saved.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10000,
        help="Target number of valid records to produce.",
    )
    parser.add_argument(
        "--retry-on-validation-fail",
        "--retry-on-validation-failure",
        dest="retry_on_validation_failure",
        type=int,
        default=1,
        help="Number of retries when a generated response fails validation.",
    )
    parser.add_argument(
        "--persona-max-fraction",
        type=float,
        default=0.0,
        help="Enable persona balancing if > 0 by capping persona share.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from checkpoint state if available.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoint and start fresh.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="checkpoints/persona-regeneration",
        help="S3 prefix for checkpoint state files.",
    )
    parser.add_argument(
        "--checkpoint-job-name",
        type=str,
        default=None,
        help="Checkpoint job name.",
    )
    parser.add_argument(
        "--checkpoint-s3-key",
        type=str,
        default=None,
        help="Full checkpoint S3 key path.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=250,
        help="Write checkpoint every N valid records.",
    )
    parser.add_argument(
        "--defense-model-path",
        type=str,
        default="/tmp/model.ckpt",
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
    parser.add_argument(
        "--nim-only",
        action="store_true",
        help="Use NIM-only mode when compatible with host command wrappers.",
    )

    args = parser.parse_args()
    args.checkpoint_job_name = args.checkpoint_job_name or os.environ.get(
        "CHECKPOINT_JOB_NAME", "persona-regen"
    )
    args.checkpoint_s3_key = _resolve_checkpoint_s3_key(args)

    if args.checkpoint_frequency < 0:
        args.checkpoint_frequency = 0

    if not os.environ.get("OVH_S3_ACCESS_KEY") or not os.environ.get(
        "OVH_S3_SECRET_KEY"
    ):
        logger.warning(
            "OVH_S3_ACCESS_KEY / OVH_S3_SECRET_KEY not set. "
            "S3 operations will fail if not using instance roles."
        )

    if args.max_records <= 0:
        logger.error("--max-records must be a positive integer.")
        sys.exit(1)

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
        "Found %d dataset file(s). Will run until %d valid records are produced.",
        len(input_files),
        args.max_records,
    )

    device = (
        "cuda"
        if os.environ.get("CUDA_VISIBLE_DEVICES") or os.path.exists("/dev/nvidia0")
        else "cpu"
    )

    simulator = _setup_simulator(args, loader, device)
    temp_out_path, records_written = _process_records(
        args,
        loader,
        simulator,
        input_files,
    )

    logger.info(
        "Uploading %d records to s3://%s/%s",
        records_written,
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
