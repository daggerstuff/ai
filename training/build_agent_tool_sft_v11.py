#!/usr/bin/env python3
"""v11 agent-tool data: memory values copied verbatim, modifier included.

The v15 paired evaluation fails ``memory_value`` on the capability split in both
arms. The request is

    Save my preferred report format as a short bullet list using the memory key
    "report_format", then confirm briefly.

and the check requires the exact string ``short bullet list``. Both arms call
``memory_write`` correctly and both store ``bullet list``: the leading modifier
is dropped and the head noun survives. The contract case passes with
``terse fragments``, so the defect is not "values are hard" — it is "a three
word value loses its first word".

The corpus teaches the wrong habit. Every memory family it carries uses one or
two word values ("short bullets", "plain text", "large", "CSV"), so the adapter
has never been asked to carry a value whose first word is load bearing.

This family trains exactly that: multi word values whose leading modifier
changes the meaning, requests phrased the way the evaluation phrases them
("as a {value} using the memory key ..."), and a tool argument that carries the
value verbatim with only the indefinite article stripped.

Usage::

    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v11 --dry-run
    PYTHONPATH=/home/vivi/pixelated python -m training.build_agent_tool_sft_v11
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_agent_tool_sft as base
import build_agent_tool_sft_v4 as v4
import build_agent_tool_sft_v5 as v5
import build_agent_tool_sft_v9 as v9

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-11-08-v11"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-11-08-v11"
BASE_MODEL = v9.BASE_MODEL

FAMILY = "memory_value_verbatim"
TOOL_NAME = "memory_write"
TOOL_DESCRIPTION = "Write synthetic session memory."
EXAMPLES = 300

SPLITS = {"train": range(240), "validation": range(240, 270), "test": range(270, 300)}

# Natural key labels. None of them is an evaluation key, and the snake_case form
# is what the tool argument carries.
KEYS: list[str] = [
    "summary layout",
    "weekly update",
    "meeting brief",
    "export style",
    "handover note",
    "briefing format",
    "status update form",
    "written recap style",
    "session note layout",
    "review summary shape",
    "shift handover format",
    "incident recap style",
]

# Every value is multi word and every value opens with the word that carries the
# meaning. Drop the first word of any of these and the preference inverts. All of
# them are singular noun phrases so "as a {value}" reads correctly.
VALUES: list[str] = [
    "single page summary",
    "short paragraph recap",
    "plain language overview",
    "numbered step list",
    "high level summary",
    "one line status",
    "two sentence recap",
    "brief morning digest",
    "compact weekly report",
    "plain text outline",
    "three bullet summary",
    "detailed written summary",
]

# The evaluation keys and values this family must never contain: the whole point
# is that the shape generalises, not that the fixture is memorised.
FORBIDDEN_KEYS = frozenset({"report_format", "note_style"})
FORBIDDEN_VALUES = frozenset({"short bullet list", "terse fragments"})

# A value with a single word has no leading modifier to preserve, which is
# exactly the shape this family exists to stop the adapter from collapsing to.
MIN_VALUE_WORDS = 2

ASKS = [
    'Save my preferred {label} as a {value} using the memory key "{key}", then confirm briefly.',
    'Store my preferred {label} as {value} in the memory key "{key}", then confirm briefly.',
    'Remember that my preferred {label} is a {value} — put it in the memory key "{key}" and confirm briefly.',
    'Please save my preferred {label} as {value} under the memory key "{key}". Confirm briefly.',
    'Set the memory key "{key}" to my preferred {label}: a {value}. Then confirm briefly.',
    'My preferred {label} is a {value}. Save that in the memory key "{key}" and confirm briefly.',
]

PROSE = [
    "Saved — your preferred {label} is recorded as {value}.",
    "Done. The memory key now holds {value} for your preferred {label}.",
    "Stored: {value}.",
    "Got it — {value} is saved as your preferred {label}.",
    "Confirmed, your preferred {label} is saved as {value}.",
]

TOOLS = [
    base.tool(
        TOOL_NAME,
        TOOL_DESCRIPTION,
        {"key": {"type": "string"}, "value": {"type": "string"}},
        ["key", "value"],
    )
]


def make_memory(index: int) -> dict[str, Any]:
    label = KEYS[index % len(KEYS)]
    value = VALUES[(index // len(KEYS)) % len(VALUES)]
    ask = ASKS[(index // (len(KEYS) * len(VALUES))) % len(ASKS)]
    prose = PROSE[index % len(PROSE)]
    system = v5.SYSTEM_PROMPTS[(index // (len(KEYS) * len(VALUES))) % len(v5.SYSTEM_PROMPTS)]
    key = v4.snake(label)
    call_id = f"call-{index}-m"
    turns = [
        base.assistant_call(TOOL_NAME, {"key": key, "value": value}, call_id),
        v9.harness_tool_reply(call_id, TOOL_NAME, {"status": "ok", "key": key, "value": value}),
        {"role": "assistant", "content": prose.format(label=label, value=value)},
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ask.format(label=label, key=key, value=value)},
        *turns,
    ]
    example_id = f"{FAMILY}-v11-{index:03d}"
    return {"id": example_id, "family": FAMILY, "group_id": example_id, "messages": messages, "tools": TOOLS}


def audit_verbatim_value(item: dict[str, Any]) -> None:
    """The stored value must be the request's value, word for word."""
    call = next(
        message for message in item["messages"] if message.get("role") == "assistant" and message.get("tool_calls")
    )
    arguments = json.loads(call["tool_calls"][0]["function"]["arguments"])
    if set(arguments) != {"key", "value"}:
        raise ValueError(f"{item['id']}: unexpected tool arguments {sorted(arguments)}")
    key, value = arguments["key"], arguments["value"]
    ask = item["messages"][1]["content"]
    if key in FORBIDDEN_KEYS:
        raise ValueError(f"{item['id']}: argument reuses an evaluation memory key")
    if value in FORBIDDEN_VALUES:
        raise ValueError(f"{item['id']}: argument reuses an evaluation memory value")
    if value not in VALUES:
        raise ValueError(f"{item['id']}: argument value is not a pool value")
    if value not in ask:
        raise ValueError(f"{item['id']}: argument value absent from the request")
    if value != " ".join(value.split()):
        raise ValueError(f"{item['id']}: argument value has unstable whitespace")
    if value.startswith(("a ", "an ")):
        raise ValueError(f"{item['id']}: argument value kept the indefinite article")
    if len(value.split()) < MIN_VALUE_WORDS:
        raise ValueError(f"{item['id']}: argument value has no leading modifier")
    label = next(label for label in KEYS if v4.snake(label) == key)
    if label in value:
        raise ValueError(f"{item['id']}: argument value welds the key label onto the value")
    expected_reply = json.dumps(
        {"status": "ok", "key": key, "value": value}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    for message in item["messages"]:
        if message.get("role") == "tool" and message["content"] != expected_reply:
            raise ValueError(f"{item['id']}: tool reply is not the harness memory payload")


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    """Build and validate every v11 example in memory, keyed by split."""
    all_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    payload_hashes: set[str] = set()

    for index in range(EXAMPLES):
        item = make_memory(index)
        base.validate_row(item)
        audit_verbatim_value(item)
        split = next(name for name, indices in SPLITS.items() if index in indices)
        all_rows[split].append(item)
        payload_line = json.dumps(
            {"messages": item["messages"], "tools": item["tools"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if payload_line in payload_hashes:
            raise ValueError(f"duplicate exported training payload: {item['id']}")
        payload_hashes.add(payload_line)

    texts = v9.evaluation_texts()
    leaked = [
        item["id"]
        for rows in all_rows.values()
        for item in rows
        if any(message.get("content") in texts for message in item["messages"])
    ]
    if leaked:
        raise ValueError(f"v11 examples reuse evaluation text: {leaked[:3]}")

    overlap = v9.prior_payload_hashes() & payload_hashes
    if overlap:
        raise ValueError(f"v11 payloads overlap prior train payloads: {len(overlap)}")

    base.verify_evaluation_separation(all_rows)
    return all_rows


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")

    all_rows = collect_rows()
    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}

    counts: Counter[str] = Counter(item["family"] for rows in all_rows.values() for item in rows)
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v16 addendum: store memory values verbatim, leading modifier included; the v15 adapter "
            "and the base model both write 'bullet list' where the request said 'short bullet list'"
        ),
        "source": "first-party deterministic synthetic trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": sum(counts.values()),
        "scenario_families": {FAMILY: counts[FAMILY]},
        "split_rule": (
            "the first 80% of indices train, the next 10% validation, the last 10% test; no group crosses split"
        ),
        "splits": {name: files[name] for name in files},
        "tools": {TOOL_NAME: counts[FAMILY]},
        "design_notes": {
            "tool_reply_shape": "exact harness form: role tool, tool_call_id, name, canonical_json content",
            "verbatim_value": (
                "the request names the value with an indefinite article; the tool argument carries the "
                "value word for word with the article stripped, matching the memory_value check"
            ),
            "value_pool": (
                "every value is at least two words and opens with the modifier that carries the meaning, "
                "so dropping the first word changes what is stored"
            ),
            "evaluation_separation": (
                "no evaluation memory key or value appears in this family; the capability and contract "
                "fixtures stay held out so a pass is generalisation, not recall"
            ),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="validate every row without writing")
    args = parser.parse_args()
    if args.dry_run:
        all_rows = collect_rows()
        sys.stdout.write(f"validated {sum(len(rows) for rows in all_rows.values())} examples\n")
        for name, rows in all_rows.items():
            sys.stdout.write(f"  {name}: {len(rows)}\n")
        return 0
    manifest = build(args.output)
    sys.stdout.write(f"wrote {manifest['example_count']} examples to {args.output}\n")
    for name, info in manifest["splits"].items():
        sys.stdout.write(f"  {name}: {info['row_count']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
