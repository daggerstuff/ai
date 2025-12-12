#!/usr/bin/env python3
"""Extract long-running therapy sessions from multi-turn professional datasets.

Outputs:
- ai/training_ready/data/generated/long_running_therapy.jsonl
- ai/training_ready/data/generated/long_running_therapy_stats.json

Default sources (from s3_manifest.json):
- datasets/gdrive/tier2_professional/soulchat_2_0_complete_no_limits.jsonl
- datasets/gdrive/tier2_professional/additional_specialized_conversations.jsonl

Notes:
- Uses the s3_manifest.json bucket/endpoint for S3 access.
- Extracts conversations with >= min_turns.
- Converts to ChatML (messages[]) and tags metadata.source_family=long_running_therapy.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

DEFAULT_SOURCE_KEYS = [
    "datasets/gdrive/tier2_professional/soulchat_2_0_complete_no_limits.jsonl",
    "datasets/gdrive/tier2_professional/additional_specialized_conversations.jsonl",
    "datasets/gdrive/raw/SoulChat2.0/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json",
]

logger = logging.getLogger(__name__)


def _extract_turn_content(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if not isinstance(turn, dict):
        return ""
    for k in ("content", "text", "value", "message", "utterance"):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _role_from_turn(turn: Any) -> str | None:
    if not isinstance(turn, dict):
        return None
    for k in ("role", "from", "speaker", "author"):
        v = turn.get(k)
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in ("system",):
                return "system"
            if vv in ("user", "human", "client", "patient"):
                return "user"
            if vv in ("assistant", "gpt", "bot", "therapist", "counselor"):
                return "assistant"
    return None


def _to_chatml_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    # Try common structures
    turns: list[Any] = []

    if isinstance(record.get("conversations"), list):
        turns = record["conversations"]
    elif isinstance(record.get("conversation"), list):
        turns = record["conversation"]
    elif isinstance(record.get("messages"), list):
        # Already ChatML-ish
        msgs = []
        for m in record["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                msgs.append({"role": role, "content": content})
        return msgs

    if not turns:
        return []

    # Some corpora store conversations as a list-of-lists
    # (each inner list is a dialogue).
    # If so, flatten one level when elements are lists of turns.
    if isinstance(turns[0], list):
        # Prefer the longest inner dialogue as the "session"
        inner = max((t for t in turns if isinstance(t, list)), key=len, default=[])
        turns = inner

    # Add a stable system message to support long-session continuity
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a therapeutic AI assistant. "
                "Maintain continuity across a long session, "
                "track context, and respond with empathy and practical support."
            ),
        }
    ]

    # Build roles; if unknown, alternate starting with user
    next_role = "user"
    for t in turns:
        content = _extract_turn_content(t)
        if not content:
            continue

        role = _role_from_turn(t) or next_role
        if role == "system":
            # skip extra system turns; keep single system header
            continue
        messages.append({"role": role, "content": content})

        next_role = "assistant" if role == "user" else "user"

    # Need at least one user+assistant exchange
    roles = {m["role"] for m in messages}
    return messages if "user" in roles and "assistant" in roles else []


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
    )
    parser.add_argument(
        "--source-key",
        action="append",
        default=[],
        help="S3 key to stream JSONL from (repeatable). If omitted, uses defaults.",
    )
    parser.add_argument("--min-turns", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parents[1] / "data" / "generated" / "long_running_therapy.jsonl"
        ),
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


def _iter_payload_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return iter(payload)
    if isinstance(payload, dict):
        return iter(
            payload.get("data") or payload.get("records") or payload.get("conversations") or []
        )
    return iter([])


def _iter_source_records(
    loader: S3DatasetLoader,
    *,
    bucket: str,
    key: str,
) -> Any:
    s3_path = f"s3://{bucket}/{key}"
    lower = key.lower()
    if lower.endswith(".jsonl"):
        return loader.stream_jsonl(s3_path)
    if lower.endswith(".json"):
        return _iter_payload_records(loader.load_json(s3_path))
    return iter([])


def _count_user_assistant_turns(messages: list[dict[str, str]]) -> int:
    return sum(m["role"] in ("user", "assistant") for m in messages)


def _build_output_record(
    *,
    messages: list[dict[str, str]],
    s3_path: str,
    turns: int,
) -> dict[str, Any]:
    return {
        "messages": messages,
        "metadata": {
            "source_family": "long_running_therapy",
            "source_key": s3_path,
            "pii_status": "none_detected",
            "license_tag": "therapeutic_license",
            "split": "test",  # hard holdout by contract
            "phase": "stage5_long_running_therapy",
            "conversation_length": turns,
            "provenance": {
                "original_s3_key": s3_path,
                "processing_pipeline": "extract_long_running_therapy",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "dedup_status": "unique",
                "processing_steps": ["chatml_convert", "length_filter"],
            },
        },
    }


def _limit_reached(*, kept: int, limit: int) -> bool:
    return limit > 0 and kept >= limit


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = _build_arg_parser()
    args = parser.parse_args()

    bucket, endpoint = _load_s3_manifest(Path(args.manifest))
    source_keys = args.source_key or DEFAULT_SOURCE_KEYS

    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    failures = 0
    turn_hist = Counter()
    sources_used = Counter()

    started_at = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", encoding="utf-8") as f:
        for key in source_keys:
            s3_path = f"s3://{bucket}/{key}"
            sources_used[key] += 1

            iterator = _iter_source_records(loader, bucket=bucket, key=key)

            for rec in iterator:
                seen += 1
                if not isinstance(rec, dict):
                    failures += 1
                    continue
                messages = _to_chatml_messages(rec)
                if not messages:
                    failures += 1
                    continue

                turns = _count_user_assistant_turns(messages)
                turn_hist[str(min(turns, 200))] += 1

                if turns < args.min_turns:
                    continue

                out = _build_output_record(
                    messages=messages,
                    s3_path=s3_path,
                    turns=turns,
                )
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept += 1

                if _limit_reached(kept=kept, limit=args.limit):
                    break

            if _limit_reached(kept=kept, limit=args.limit):
                break

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "bucket": bucket,
        "endpoint": endpoint,
        "source_keys": source_keys,
        "min_turns": args.min_turns,
        "seen_records": seen,
        "kept_records": kept,
        "parse_failures": failures,
        "sources_used": dict(sources_used),
        "turn_histogram_capped_200": dict(turn_hist),
    }

    stats_path = out_path.with_name("long_running_therapy_stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Wrote %s long-running therapy sessions to %s", kept, out_path)
    logger.info("Stats: %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
