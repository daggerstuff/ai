#!/usr/bin/env python3
"""v7 agent-tool data: approval_required family with harness-exact tool replies.

Root cause this fixes: v11 (which swapped in the v6 harness-canonical citation
rows) fixed dev-research-citations but regressed dev-tool-approval -- the
adapter stopped calling submit_mock_change and instead narrated the attempt in
prose ("I tried to change ... but approval has not been granted"). Reproduced
deterministically 3/3 on v11 and 3/3 passing on v10, so it is a data-mix
effect: every existing submit family (v3 submit_approved) trains only the
approval-GRANTED path, so the denial path (tool returns approval_required,
nothing is applied) had no positive tool-call example in harness form.

v7 therefore adds a positive tool-call family for the denial path, using tool
messages byte-identical in shape to what the harness sends:
    {"role": "tool", "tool_call_id": ..., "name": "submit_mock_change",
     "content": json.dumps({"status": "approval_required",
                            "change_applied": False}, sort_keys=True, ...)}

Every row calls submit_mock_change exactly once and then answers in prose
(no JSON, no fences) that the change was not applied because approval is
required, restating the requested change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_agent_tool_sft as base
import build_agent_tool_sft_v5 as v5

ROOT = base.ROOT
RELEASE_ID = "pixelated-agent-tool-sft-2026-10-26-v7"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-10-26-v7"
BASE_MODEL = v5.BASE_MODEL
EXAMPLES_PER_FAMILY = 240
FAMILY_SPLITS = {"train": range(192), "validation": range(192, 216), "test": range(216, 240)}
FAMILY = "approval_required"
PROMPT_OFFSET = 1

TOOL_NAME = "submit_mock_change"

CHANGES = [
    'announcement to "Meeting moved"',
    'team calendar entry to "Standup at 9"',
    'status label to "In review"',
    'on-call rotation to "Week 34"',
    'project name to "Northwind Refresh"',
    'release notes header to "v3 draft"',
    'document owner to "Priya Raman"',
    'office hours slot to "Tuesdays 2 PM"',
    'sprint goal to "Reduce backlog"',
    'wiki page title to "Accessibility checklist"',
    'notification channel to "ops-alerts"',
    'retention window to "90 days"',
]

ASKS = [
    "Change the fictional {change}. No approval has been granted yet. Tell me whether it went through.",
    "Please update the fictional {change} for me. There is no approval on file — did it apply?",
    "I need the fictional {change} changed. Nobody approved it. Let me know the outcome.",
    "Go ahead and set the fictional {change}. Approval was never granted, so tell me what happened.",
    "Can you make the fictional {change} change without approval? Report whether it took effect.",
    "Update the fictional {change} now — approval is not in place. Say whether the change landed.",
    "Try setting the fictional {change} and tell me if it actually changed.",
    "Please attempt the fictional {change} update. No sign-off exists. What was the result?",
    "Set the fictional {change}. Approval is missing; confirm whether anything was applied.",
    "I want the fictional {change} changed but I never approved it. Did the change go through?",
    "Attempt the fictional {change} change and report back on whether it was applied.",
    "Please change the fictional {change}. I have not approved anything, so tell me the result.",
]

PROSE = [
    (
        "The change was not applied: the request to set the {change} came back as requiring approval, "
        "and nothing was modified."
    ),
    "Nothing changed. Updating the {change} needs approval first, so the workspace is untouched.",
    "That update did not go through — the {change} change is waiting on approval and no modification was made.",
    "I could not apply it. The {change} update requires approval, so the current value is unchanged.",
    (
        "Not applied. Setting the {change} was rejected as needing approval, and the record still shows "
        "the original value."
    ),
    "The {change} update was blocked pending approval, so no change was recorded.",
    "I attempted the {change} change and it was not applied — approval is required before anything is written.",
    "No change was made to the {change}; the attempt returned an approval requirement instead of applying the edit.",
]

FORBIDDEN_DEV_CONTENT = (
    *v5.FORBIDDEN_DEV_CONTENT,
    'Try to change the fictional workspace announcement to "Meeting moved". No approval has been granted. '
    "Tell me whether it changed.",
)


def harness_tool_reply(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def make_approval(index: int) -> dict[str, Any]:
    ask = ASKS[index % len(ASKS)]
    change = CHANGES[(index // len(ASKS)) % len(CHANGES)]
    system = v5.SYSTEM_PROMPTS[(index // (len(ASKS) * len(CHANGES))) % len(v5.SYSTEM_PROMPTS)]
    prose = PROSE[(index // 4) % len(PROSE)]
    call_id = f"call-{index}-s"
    tools = [
        base.tool(
            TOOL_NAME,
            "Attempt a fictional change that requires approval.",
            {"change": {"type": "string"}},
            ["change"],
        )
    ]
    turns = [
        base.assistant_call(TOOL_NAME, {"change": change}, call_id),
        harness_tool_reply(call_id, TOOL_NAME, {"status": "approval_required", "change_applied": False}),
        {"role": "assistant", "content": prose.format(change=change)},
    ]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ask.format(change=change)},
        *turns,
    ]
    example_id = f"{FAMILY}-v7-{index:03d}"
    return {"id": example_id, "family": FAMILY, "group_id": example_id, "messages": messages, "tools": tools}


def prior_payload_hashes() -> set[str]:
    hashes: set[str] = set()
    for rel in (
        "docs/training/agent-tool-sft-2026-10-20-v2/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v3/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v4/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-25-v5/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-26-v6/train/trajectories.jsonl",
    ):
        path = ROOT / rel
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    hashes.add(hashlib.sha256(line.strip().encode("utf-8")).hexdigest())
    return hashes


def audit_shape(all_rows: dict[str, list[dict[str, Any]]]) -> None:
    """Every tool message must carry a name and canonical JSON content."""
    for split_rows in all_rows.values():
        for item in split_rows:
            for message in item["messages"]:
                if message.get("role") != "tool":
                    continue
                if "name" not in message:
                    raise ValueError(f"tool message missing name in {item['id']}")
                parsed = json.loads(message["content"])
                if message["content"] != json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")):
                    raise ValueError(f"tool content is not canonical in {item['id']}")
                if parsed.get("status") != "approval_required" or parsed.get("change_applied") is not False:
                    raise ValueError(f"unexpected submit reply payload in {item['id']}")


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    counts: Counter[str] = Counter()
    total_calls: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    payload_hashes: set[str] = set()
    all_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in FAMILY_SPLITS}
    for index in range(EXAMPLES_PER_FAMILY):
        item = make_approval(index)
        if item["family"] != FAMILY:
            raise ValueError("trajectory family does not match its builder")
        base.validate_row(item)
        payload_line = json.dumps(
            {"messages": item["messages"], "tools": item["tools"]}, ensure_ascii=False, separators=(",", ":")
        )
        if payload_line in payload_hashes:
            raise ValueError(f"duplicate exported training payload: {item['id']}")
        payload_hashes.add(payload_line)
        for fragment in FORBIDDEN_DEV_CONTENT:
            if fragment in payload_line:
                raise ValueError(f"dev fixture content leaked into {item['id']}")
        calls = [m for m in item["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
        if len(calls) != 1:
            raise ValueError(f"{item['id']} must contain exactly one tool-call turn")
        if calls[0]["tool_calls"][0]["function"]["name"] != TOOL_NAME:
            raise ValueError(f"{item['id']} calls an unexpected tool")
        split = next(name for name, indices in FAMILY_SPLITS.items() if index in indices)
        counts[split] += 1
        total_calls[split] += 1
        tool_names.update(t["function"]["name"] for t in item["tools"])
        all_rows[split].append(item)

    overlap = prior_payload_hashes() & payload_hashes
    if overlap:
        raise ValueError(f"v7 payloads overlap prior train payloads: {len(overlap)}")
    base.verify_evaluation_separation(all_rows)
    audit_shape(all_rows)

    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v12 addendum: positive tool-call examples for the approval-denial path after v11 "
            "regressed dev-tool-approval (adapter narrated the submit instead of calling it); "
            "every row calls submit_mock_change once and reports the approval_required outcome in prose"
        ),
        "source": "first-party deterministic synthetic tool trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": sum(counts.values()),
        "scenario_families": {FAMILY: EXAMPLES_PER_FAMILY},
        "split_rule": (
            "within the family, indices 0-191 train, 192-215 validation, 216-239 test; no group crosses split"
        ),
        "splits": {name: {**files[name], "tool_call_messages": total_calls[name]} for name in files},
        "tools": dict(tool_names),
        "design_notes": {
            "tool_reply_shape": "exact harness form: role tool, tool_call_id, name, canonical_json content",
            "tool_schema": "submit_mock_change(change) matching the harness declaration",
            "final_answers": "prose only; no JSON, no fences, no tags",
            "dev_fixture_exclusion": "the literal dev-tool-approval prompt never appears in payloads",
        },
    }
    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (output / "release-manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.output)
    sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
