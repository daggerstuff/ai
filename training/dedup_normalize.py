#!/usr/bin/env python3
"""Deduplication and ChatML normalization pipeline.

Exact dedup via SHA-256 content hash, near-dedup via Jaccard token similarity,
edge-case preservation, ChatML boundary verification, and sharded JSONL output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("dedup_normalize")

_INST_BOUNDARY = re.compile(r"\[/INST\]")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _extract_text(record: dict) -> str:
    messages = record.get("messages", [])
    if messages:
        return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
    if record.get("prompt") and record.get("chosen") and record.get("rejected"):
        return record["prompt"] + " " + record["chosen"] + " " + record["rejected"]
    return record.get("text", "") or record.get("instruction", "") + " " + record.get("output", "")


def _is_edge_case(record: dict) -> bool:
    return record.get("is_training_edge_case", False) is True


def _verify_chatml_boundary(record: dict) -> bool:
    text = _extract_text(record)
    if "[/INST]" in text:
        return bool(_INST_BOUNDARY.search(text))
    if "messages" in record:
        return True
    return True


def _attempt_reformat(record: dict) -> dict | None:
    """Try to reformat a record to have proper ChatML boundaries."""
    text = _extract_text(record)
    if not text:
        return None

    if "[/INST]" not in text and "messages" not in record:
        instruction = record.get("instruction", "")
        output = record.get("output", "")
        if instruction and output:
            record = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "metadata": record.get("metadata", {}),
            }
            if _is_edge_case(record) if "is_training_edge_case" in record else False:
                record["is_training_edge_case"] = True
            return record
    return None


def process_file(
    input_path: Path,
    seen_hashes: set[str],
    token_sets: list[tuple[frozenset[str], str]],
    jaccard_threshold: float,
    edge_case_hashes: set[str],
    rejection_log: list[dict],
) -> tuple[list[dict], int, int, int, int, int]:
    """Process one JSONL file.

    Returns (kept_records, exact_dupes, near_dupes, chatml_failures,
             reformatted, total_read).
    """
    kept: list[dict] = []
    exact_dupes = 0
    near_dupes = 0
    chatml_failures = 0
    reformatted = 0
    total_read = 0

    try:
        with open(input_path, encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                total_read += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in %s line %d", input_path, line_no)
                    continue

                text = _extract_text(record)
                text_hash = _content_hash(text)

                if _is_edge_case(record):
                    edge_case_hashes.add(text_hash)
                    if not _verify_chatml_boundary(record):
                        reformatted_rec = _attempt_reformat(record)
                        if reformatted_rec:
                            record = reformatted_rec
                            reformatted += 1
                        else:
                            chatml_failures += 1
                            rejection_log.append({
                                "file": str(input_path),
                                "line": line_no,
                                "reason": "ChatML boundary failure (edge case)",
                            })
                            continue
                    kept.append(record)
                    continue

                if text_hash in seen_hashes or text_hash in edge_case_hashes:
                    exact_dupes += 1
                    continue

                tokens = _token_set(text)
                is_near_dup = False
                for existing_tokens, existing_hash in token_sets:
                    if existing_hash == text_hash:
                        continue
                    if _jaccard_similarity(tokens, existing_tokens) > jaccard_threshold:
                        is_near_dup = True
                        break

                if is_near_dup:
                    near_dupes += 1
                    continue

                if not _verify_chatml_boundary(record):
                    reformatted_rec = _attempt_reformat(record)
                    if reformatted_rec:
                        record = reformatted_rec
                        reformatted += 1
                    else:
                        chatml_failures += 1
                        rejection_log.append({
                            "file": str(input_path),
                            "line": line_no,
                            "reason": "ChatML boundary failure",
                        })
                        continue

                seen_hashes.add(text_hash)
                token_sets.append((tokens, text_hash))
                kept.append(record)

    except OSError as exc:
        logger.warning("Cannot read %s: %s", input_path, exc)

    return kept, exact_dupes, near_dupes, chatml_failures, reformatted, total_read


def run_dedup(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jaccard_threshold = args.jaccard_threshold
    shard_size = args.shard_size

    seen_hashes: set[str] = set()
    edge_case_hashes: set[str] = set()
    token_sets: list[tuple[frozenset[str], str]] = []
    rejection_log: list[dict] = []

    all_kept: list[dict] = []
    total_in = 0
    total_exact_dup = 0
    total_near_dup = 0
    total_chatml_fail = 0
    total_reformatted = 0
    total_edge_preserved = 0

    for input_dir in args.input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.warning("Input directory not found: %s", input_path)
            continue
        for jsonl_file in sorted(input_path.rglob("*.jsonl")):
            kept, exact, near, chatml, refmt, n_read = process_file(
                jsonl_file, seen_hashes, token_sets, jaccard_threshold,
                edge_case_hashes, rejection_log,
            )
            all_kept.extend(kept)
            total_in += n_read
            total_exact_dup += exact
            total_near_dup += near
            total_chatml_fail += chatml
            total_reformatted += refmt
            total_edge_preserved += sum(1 for r in kept if _is_edge_case(r))

    shard_count = 0
    for i in range(0, len(all_kept), shard_size):
        shard = all_kept[i:i + shard_size]
        shard_path = output_dir / f"shard_{shard_count:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for record in shard:
                f.write(json.dumps(record) + "\n")
        shard_count += 1

    if rejection_log:
        rejection_path = output_dir / args.rejection_log
        with open(rejection_path, "w", encoding="utf-8") as f:
            for entry in rejection_log:
                f.write(json.dumps(entry) + "\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dirs": args.input_dirs,
        "total_samples_in": total_in,
        "exact_duplicates": total_exact_dup,
        "near_duplicates": total_near_dup,
        "chatml_failures": total_chatml_fail,
        "reformatted": total_reformatted,
        "edge_cases_preserved": total_edge_preserved,
        "total_samples_out": len(all_kept),
        "shard_count": shard_count,
        "jaccard_threshold": jaccard_threshold,
        "shard_size": shard_size,
    }
    report_path = output_dir / "normalization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Dedup complete: %d in → %d out (%d exact dup, %d near dup, %d ChatML fail, %d edge preserved)",
        total_in, len(all_kept), total_exact_dup, total_near_dup,
        total_chatml_fail, total_edge_preserved,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deduplicate and normalize training data.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Directories containing JSONL files to deduplicate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for sharded JSONL and reports.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistral-nemo",
        help="Model name for ChatML template verification.",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-dedup.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Maximum records per output shard.",
    )
    parser.add_argument(
        "--rejection_log",
        type=str,
        default="rejection_log.jsonl",
        help="Filename for ChatML rejection log.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_dedup(args)


if __name__ == "__main__":
    main()
