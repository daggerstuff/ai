"""Build the unified mixed-task SFT dataset (Phase 2.3 — Dry-Run SFT Validation).

PIX-4070 Task 2.3 — combine the Phase 2.1 persona-selection records and Phase
2.2 persona-enhanced dialogue records into a single ChatML JSONL file, tagged
with ``task_type`` in metadata, then validate every record against strict
ChatML formatting requirements.

AC (from PIX-4070 Phase 2.3):
  * Generate a unified 10,000-record mixed-task JSONL file.
  * Validate output against strict ChatML formatting requirements.

This module is intentionally tokenizer-free — it reuses the same char-based
proxy (CHARS_PER_TOKEN = 4) as ``generate_sft_dialogue`` so the dry-run does
not require a HF tokenizer download. Phase 4 wires the real tokenizer in.

Input formats (both already produced by Phase 2.1 / 2.2 generators):

  selection record  : {"messages": [...], "metadata": {...}}
  dialogue record   : {"messages": [...], "metadata": {...}}

Output format (unified):

  {"messages": [...], "metadata": {..., "task_type": "selection"|"dialogue"}}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_ROLES = ("system", "user", "assistant", "tool")
# Match Phase 2.2's proxy so token estimates are comparable across tasks.
CHARS_PER_TOKEN = 4
DEFAULT_TARGET_RECORDS = 10_000
DEFAULT_MAX_TOKENS = 1024
TASK_SELECTION = "selection"
TASK_DIALOGUE = "dialogue"
VALID_TASK_TYPES = (TASK_SELECTION, TASK_DIALOGUE)


def is_chatml_compliant(messages: list[dict[str, Any]]) -> bool:
    """Strict ChatML shape check: non-empty list of {role, content: str} dicts."""
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES:
            return False
        if not isinstance(m.get("content"), str) or not m["content"]:
            return False
    # ChatML requires the first turn to be ``system`` per the PAL paper template
    # (both Phase 2.1 and Phase 2.2 emit system-first messages).
    return messages[0].get("role") == "system"


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative token estimate: total chars / CHARS_PER_TOKEN."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return (total_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def validate_token_bounds(messages: list[dict[str, str]], max_tokens: int) -> bool:
    return estimate_tokens(messages) <= max_tokens


def _has_json_leakage(s: str) -> bool:
    """Return True if the string contains JSON-structural characters that
    indicate raw JSON bled through into the natural-language text.

    The PAL paper requires personae be expressed in natural language only;
    any ``{`` ``}`` or ``"`` strongly suggests JSON formatting leaked.
    Single quotes (``'``) are NOT flagged — they are legitimate natural-language
    punctuation (apostrophes in contractions like "patient's" or possessives).
    """
    return any(ch in s for ch in ("{", "}", '"'))


def _messages_have_json_leakage(messages: list[dict[str, str]]) -> bool:
    return any(_has_json_leakage(m.get("content", "")) for m in messages)


def validate_record(record: dict[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS) -> bool:
    """Strict validation: ChatML shape + token bound + no JSON leakage."""
    if not isinstance(record, dict):
        return False
    messages = record.get("messages")
    if not isinstance(messages, list):
        return False
    if not is_chatml_compliant(messages):
        return False
    if not validate_token_bounds(messages, max_tokens):
        return False
    return not _messages_have_json_leakage(messages)


def load_records(
    path: Path,
    task_type: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Load a Phase 2.1/2.2 output JSONL, tagging each record's metadata with ``task_type``.

    Records failing ``validate_record`` are dropped and counted.
    """
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}, got {task_type!r}")
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    out: list[dict[str, Any]] = []
    dropped = 0
    with path.open(encoding="utf-8") as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                dropped += 1
                continue
            if not validate_record(record, max_tokens=max_tokens):
                dropped += 1
                continue
            metadata = dict(record.get("metadata", {}))
            metadata["task_type"] = task_type
            metadata["estimated_tokens"] = estimate_tokens(record["messages"])
            out.append({"messages": record["messages"], "metadata": metadata})
    if dropped:
        sys.stderr.write(f"dropped {dropped} invalid records from {path}\n")
    return out


@dataclass
class UnifiedStats:
    total: int
    selection: int
    dialogue: int
    dropped: int


def build_unified_dataset(  # noqa: PLR0913 — all 6 params are necessary for the public API
    selection_path: Path,
    dialogue_path: Path,
    output_path: Path,
    target_records: int = DEFAULT_TARGET_RECORDS,
    seed: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> UnifiedStats:
    """Combine selection + dialogue JSONL into a unified mixed-task file.

    Interleaves the two task types uniformly, capping at ``target_records``
    (PIX-4070 Phase 2.3 AC: 10,000 records). If one source is shorter than
    half the target, the other source is sampled with replacement to fill.

    Returns ``UnifiedStats`` so callers (CLI / tests) can assert mixing ratios.
    """
    if target_records < 1:
        raise ValueError("target_records must be >= 1")
    selection = load_records(selection_path, TASK_SELECTION, max_tokens=max_tokens)
    dialogue = load_records(dialogue_path, TASK_DIALOGUE, max_tokens=max_tokens)
    rng = random.Random(seed)

    # Interleave: alternating selection/dialogue, sampling from each source.
    # If one side runs out first, fill the remainder from the other side.
    sel_pool = list(selection)
    dia_pool = list(dialogue)
    rng.shuffle(sel_pool)
    rng.shuffle(dia_pool)

    unified: list[dict[str, Any]] = []
    sel_idx = dia_idx = 0
    while len(unified) < target_records:
        want_sel = len(unified) % 2 == 0
        if want_sel and sel_idx < len(sel_pool):
            unified.append(sel_pool[sel_idx])
            sel_idx += 1
        elif not want_sel and dia_idx < len(dia_pool):
            unified.append(dia_pool[dia_idx])
            dia_idx += 1
        elif sel_idx < len(sel_pool):
            unified.append(sel_pool[sel_idx])
            sel_idx += 1
        elif dia_idx < len(dia_pool):
            unified.append(dia_pool[dia_idx])
            dia_idx += 1
        else:
            # Both pools exhausted before hitting target — sample with replacement.
            pool = sel_pool + dia_pool
            if not pool:
                break
            unified.append(rng.choice(pool))

    dropped = (len(selection) - sel_idx) + (len(dialogue) - dia_idx)
    final_sel = sum(1 for r in unified if r["metadata"].get("task_type") == TASK_SELECTION)
    final_dia = sum(1 for r in unified if r["metadata"].get("task_type") == TASK_DIALOGUE)

    with output_path.open("w", encoding="utf-8") as fout:
        for record in unified:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return UnifiedStats(
        total=len(unified),
        selection=final_sel,
        dialogue=final_dia,
        dropped=dropped,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build unified mixed-task SFT dataset (Phase 2.3 dry-run validation).",
    )
    parser.add_argument(
        "selection_input",
        type=Path,
        help="Phase 2.1 output JSONL: {messages, metadata}",
    )
    parser.add_argument(
        "dialogue_input",
        type=Path,
        help="Phase 2.2 output JSONL: {messages, metadata}",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Unified output JSONL with task_type-tagged metadata",
    )
    parser.add_argument(
        "--target-records",
        type=int,
        default=DEFAULT_TARGET_RECORDS,
        help="Target number of records in the unified file (PIX-4070 AC: 10000).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max estimated tokens per record (char proxy).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    for p, label in (
        (args.selection_input, "selection_input"),
        (args.dialogue_input, "dialogue_input"),
    ):
        if not p.exists():
            print(f"{label} not found: {p}", file=sys.stderr)  # noqa: T201 — CLI entry point
            return 1

    stats = build_unified_dataset(
        args.selection_input,
        args.dialogue_input,
        args.output,
        target_records=args.target_records,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
    print(  # noqa: T201 — CLI entry point
        f"wrote {stats.total} records to {args.output} "
        f"(selection={stats.selection}, dialogue={stats.dialogue}, dropped={stats.dropped})"
    )
    if stats.total < args.target_records:
        print(  # noqa: T201 — CLI entry point
            f"WARNING: wrote {stats.total} < target {args.target_records}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
