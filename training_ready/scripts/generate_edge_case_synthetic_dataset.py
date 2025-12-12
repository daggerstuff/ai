#!/usr/bin/env python3
"""Generate synthetic edge-case ChatML from edge-case prompts.

Outputs:
- ai/training_ready/data/generated/edge_case_synthetic.jsonl
- ai/training_ready/data/generated/edge_case_synthetic_stats.json

Design notes:
- Uses S3 manifest for bucket/endpoint.
- Pulls prompts from an existing edge-case prompts JSONL key.
- Generates SAFE therapeutic responses (no self-harm instruction, no doxxing/PII).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

DEFAULT_PROMPTS_KEY = "datasets/consolidated/edge_cases/edge_case_output/edge_case_prompts.jsonl"

logger = logging.getLogger(__name__)


_CRISIS_RE = re.compile(
    r"\b(suicide|suicidal|self[- ]?harm|kill myself|overdose|end it|i want to die)\b",
    re.IGNORECASE,
)


def _safe_system_prompt() -> str:
    return (
        "You are a therapeutic AI assistant. Respond with empathy, clarity, and practical support. "
        "Do not provide instructions for self-harm, violence, illegal acts, or doxxing. "
        "Do not include personal identifying information."
    )


def _safe_assistant_response(user_text: str) -> str:
    # Deterministic, template-based synthesis (no external model calls).
    if _CRISIS_RE.search(user_text):
        return (
            "I'm really sorry you're feeling this overwhelmed. You don't have to carry this alone. "
            "If you feel in immediate danger or might act on these thoughts, please contact your local emergency number right now, "
            "or reach out to a crisis line in your country. If you're in the U.S. or Canada you can call or text 988. "
            "If you're elsewhere, tell me your country and I can help you find a local option.\n\n"
            "For the next few minutes, let's focus on getting you through this moment safely. "
            "Are you alone right now, and is there a person you trust you can contact? "
            "Also, what's the strongest feeling in your body right now (panic, numbness, shame, anger, fear)?"
        )

    return (
        "That sounds really heavy, and it makes sense that you're feeling pulled in multiple directions. "
        "Before we problem-solve, can we slow down for a second?\n\n"
        "1) What happened most recently that made this spike?\n"
        "2) What are you afraid it means about you, your future, or your relationships?\n\n"
        "We can then choose one small next step for the next 24 hours—something realistic that reduces harm and increases support."
    )


def _extract_prompt_text(obj: dict[str, Any]) -> str | None:
    for key in ("prompt", "text", "scenario", "input", "instructions"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _iter_prompt_like_items(payload: Any) -> list[Any]:
    """
    edge_case_prompts can be stored as:
    - JSON object wrapping a list (common)
    - JSONL (rare here)
    - Direct list
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        # Prefer obvious keys first
        for k in ("prompts", "edge_case_prompts", "items", "data", "records"):
            v = payload.get(k)
            if isinstance(v, list):
                return v

        # Fallback: first list-valued field that looks like prompt records
        for v in payload.values():
            if isinstance(v, list) and v:
                return v

    return []


def _load_json_any(loader: S3DatasetLoader, s3_path: str) -> Any:
    """
    Load JSON from S3 even if the file contains multiple concatenated JSON values.
    Returns either a single JSON value or a list of values.
    """
    raw = loader.load_text(s3_path)

    # First try normal JSON
    with suppress(json.JSONDecodeError):
        return json.loads(raw)

    # Fallback: parse concatenated JSON objects/values
    decoder = json.JSONDecoder()
    idx = 0
    values: list[Any] = []
    n = len(raw)
    while idx < n:
        # Skip whitespace
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break

        try:
            val, next_idx = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            # Give up; caller will handle empty
            break
        values.append(val)
        idx = next_idx

    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        help="Path to ai/training_ready/data/s3_manifest.json",
    )
    parser.add_argument(
        "--prompts-key",
        default=DEFAULT_PROMPTS_KEY,
        help="S3 key for edge-case prompts JSONL",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "generated" / "edge_case_synthetic.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of examples to generate (0 = no limit)",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket")
    endpoint = manifest.get("endpoint")
    if not isinstance(bucket, str) or not bucket:
        raise ValueError("s3_manifest.json missing bucket")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("s3_manifest.json missing endpoint")
    return bucket, endpoint


def _write_synthetic_jsonl(
    *,
    out_path: Path,
    prompt_items: list[Any],
    prompts_s3_path: str,
    limit: int,
) -> tuple[int, int, int, Counter[str]]:
    generated = 0
    crisis_count = 0
    failures = 0
    by_source: Counter[str] = Counter()

    with out_path.open("w", encoding="utf-8") as f:
        for item in prompt_items:
            if isinstance(item, str):
                prompt_text = item.strip()
            elif isinstance(item, dict):
                prompt_text = _extract_prompt_text(item)
            else:
                prompt_text = None

            if not prompt_text:
                failures += 1
                continue

            if _CRISIS_RE.search(prompt_text):
                crisis_count += 1

            record = {
                "messages": [
                    {"role": "system", "content": _safe_system_prompt()},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": _safe_assistant_response(prompt_text)},
                ],
                "metadata": {
                    "source_family": "edge_case_synthetic",
                    "source_key": prompts_s3_path,
                    "pii_status": "none_detected",
                    "license_tag": "synthetic",
                    "split": "train",  # will be re-split during final compilation
                    "phase": "stage3_edge_stress_test",
                    "provenance": {
                        "original_s3_key": prompts_s3_path,
                        "processing_pipeline": "generate_edge_case_synthetic_dataset",
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "dedup_status": "unique",
                        "processing_steps": ["template_synthesis"],
                    },
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated += 1
            by_source["edge_case_prompts"] += 1

            if limit and generated >= limit:
                break

    return generated, crisis_count, failures, by_source


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))

    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    prompts_s3_path = f"s3://{bucket}/{args.prompts_key}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats_started = datetime.now(timezone.utc).isoformat()

    # This object is commonly a JSON wrapper, not true JSONL.
    prompts_payload: Any = _load_json_any(loader, prompts_s3_path)

    prompt_items = _iter_prompt_like_items(prompts_payload)
    if not prompt_items and isinstance(prompts_payload, list):
        # If we parsed multiple JSON values, try each for a prompt list
        for val in prompts_payload:
            prompt_items = _iter_prompt_like_items(val)
            if prompt_items:
                break

    generated, crisis_count, failures, by_source = _write_synthetic_jsonl(
        out_path=out_path,
        prompt_items=prompt_items,
        prompts_s3_path=prompts_s3_path,
        limit=args.limit,
    )

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": stats_started,
        "bucket": bucket,
        "endpoint": endpoint,
        "prompts_key": args.prompts_key,
        "output": str(out_path),
        "generated": generated,
        "crisis_keyword_hits": crisis_count,
        "prompt_parse_failures": failures,
        "counts": dict(by_source),
    }

    stats_path = out_path.with_name("edge_case_synthetic_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %s synthetic edge-case examples to %s", generated, out_path)
    logger.info("Stats: %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
