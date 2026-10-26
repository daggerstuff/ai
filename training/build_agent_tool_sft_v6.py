#!/usr/bin/env python3
"""v6 agent-tool data: citation_json family with harness-exact tool replies.

Root cause this fixes: the dev-research-citations eval failure is caused by the
harness serializing tool results with canonical_json (sorted keys, compact
separators) and adding a `name` field to tool messages, while every prior
agent-tool payload serialized tool results with default json.dumps (spaced,
insertion order) and no `name` field. A controlled replay (3 reps x 3 variants,
2026-10-26) showed the adapter answers bare JSON 3/3 under training-style tool
results but only 1/3 under harness-style results; the fixture source title is
irrelevant.

v6 therefore regenerates the v5 citation_json family with tool messages that
are byte-identical in shape to what the harness sends:
    {"role": "tool", "tool_call_id": ..., "name": <tool name>,
     "content": json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))}

The final assistant answers stay bare json.dumps objects (no fences, no tags),
and the v5 citation_json train rows are replaced (not supplemented) by these
rows in the v11 release, so the harness-shaped citation pattern exists only in
its harness-exact form.
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
RELEASE_ID = "pixelated-agent-tool-sft-20261026-v6"
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-10-26-v6"
BASE_MODEL = v5.BASE_MODEL
EXAMPLES_PER_FAMILY = 240
FAMILY_SPLITS = {"train": range(0, 192), "validation": range(192, 216), "test": range(216, 240)}
FAMILY = "citation_json"
PROMPT_OFFSET = v5.PROMPT_OFFSETS[FAMILY]

FORBIDDEN_DEV_CONTENT = v5.FORBIDDEN_DEV_CONTENT


def harness_tool_reply(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def row(index: int, user_text: str, tools: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
    example_id = f"{FAMILY}-v6-{index:03d}"
    system = v5.SYSTEM_PROMPTS[(index + PROMPT_OFFSET) % len(v5.SYSTEM_PROMPTS)]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}, *turns]
    return {"id": example_id, "family": FAMILY, "group_id": example_id, "messages": messages, "tools": tools}


def make_citation(index: int) -> dict[str, Any]:
    prefix, kind = v5.CITATION_SETS[index % len(v5.CITATION_SETS)]
    suffix_pair = v5.CITATION_SUFFIXES[(index // len(v5.CITATION_SETS)) % len(v5.CITATION_SUFFIXES)]
    topic = v5.CITATION_TOPICS[(index // 3) % len(v5.CITATION_TOPICS)]
    ask = v5.CITATION_ASKS[(index // 4) % len(v5.CITATION_ASKS)].format(kind=kind, topic=topic)
    variant = v5.CITATION_VARIANTS[index % len(v5.CITATION_VARIANTS)]
    user = ask
    search_id, r1_id, r2_id = f"call-{index}-s", f"call-{index}-r1", f"call-{index}-r2"
    tools = [
        base.tool("search_mock_sources", "Search fictional source cards.", {}, []),
        base.tool(
            "read_mock_document",
            "Read a fictional source card.",
            {"document_id": {"type": "string"}},
            ["document_id"],
        ),
    ]

    def src(i: int, suffix: str) -> dict[str, str]:
        sid = f"{prefix}-{suffix}"
        return {
            "id": sid,
            "title": v5.CITATION_TITLES[(index + i) % len(v5.CITATION_TITLES)],
            "topic": topic,
            "content": f"{v5.CITATION_CONTENT_TEMPLATES[(index + i * 5) % len(v5.CITATION_CONTENT_TEMPLATES)].format(topic=topic)} "
            f"{v5.CITATION_DETAILS[(index + i * 3) % len(v5.CITATION_DETAILS)]}",
        }

    def read_turns(cited: list[dict[str, str]]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for i, call_id in enumerate([r1_id, r2_id][: len(cited)]):
            s = cited[i]
            turns.append(base.assistant_call("read_mock_document", {"document_id": s["id"]}, call_id))
            turns.append(
                harness_tool_reply(
                    call_id,
                    "read_mock_document",
                    {"status": "ok", "document_id": s["id"], "document": s["content"]},
                )
            )
        summary = " ".join(s["content"] for s in cited)
        turns.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"topic": topic, "summary": summary, "citations": [s["id"] for s in cited]},
                    ensure_ascii=False,
                ),
            }
        )
        return turns

    if variant == "single":
        s = src(0, suffix_pair[0])
        search_reply = {"status": "ok", "sources": [{"id": s["id"], "title": s["title"], "topic": topic}]}
        cited = [s]
    elif variant == "distractor":
        s = src(0, suffix_pair[0])
        d_id = f"{prefix}-Z"
        d = {"id": d_id, "title": "Unrelated listing", "topic": "unrelated", "content": f"An unrelated entry, not about {topic}."}
        search_reply = {
            "status": "ok",
            "sources": [
                {"id": s["id"], "title": s["title"], "topic": topic},
                {"id": d["id"], "title": d["title"], "topic": d["topic"]},
            ],
        }
        cited = [s]
    else:
        s1 = src(0, suffix_pair[0])
        s2 = src(1, suffix_pair[1])
        search_reply = {
            "status": "ok",
            "sources": [
                {"id": s1["id"], "title": s1["title"], "topic": topic},
                {"id": s2["id"], "title": s2["title"], "topic": topic},
            ],
        }
        cited = [s1, s2]

    turns = [
        base.assistant_call("search_mock_sources", {}, search_id),
        harness_tool_reply(search_id, "search_mock_sources", search_reply),
        *read_turns(cited),
    ]
    return row(index, user, tools, turns)


def prior_payload_hashes() -> set[str]:
    hashes: set[str] = set()
    for rel in (
        "docs/training/agent-tool-sft-2026-10-20-v2/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v3/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v4/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-25-v5/train/trajectories.jsonl",
    ):
        path = ROOT / rel
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    hashes.add(hashlib.sha256(line.strip().encode("utf-8")).hexdigest())
    return hashes


def build(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    counts: Counter[str] = Counter()
    total_calls: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    payload_hashes: set[str] = set()
    all_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in FAMILY_SPLITS}
    for index in range(EXAMPLES_PER_FAMILY):
        item = make_citation(index)
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
        split = next(name for name, indices in FAMILY_SPLITS.items() if index in indices)
        counts[split] += 1
        total_calls[split] += sum(len(m.get("tool_calls", [])) for m in item["messages"])
        tool_names.update(t["function"]["name"] for t in item["tools"])
        all_rows[split].append(item)

    overlap = prior_payload_hashes() & payload_hashes
    if overlap:
        raise ValueError(f"v6 payloads overlap prior train payloads: {len(overlap)}")
    base.verify_evaluation_separation(all_rows)

    # Shape audit: every tool message must carry name and canonical content.
    for split_rows in all_rows.values():
        for item in split_rows:
            for message in item["messages"]:
                if message.get("role") != "tool":
                    continue
                if "name" not in message:
                    raise ValueError(f"tool message missing name in {item['id']}")
                parsed = json.loads(message["content"])
                if message["content"] != json.dumps(
                    parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ):
                    raise ValueError(f"tool content is not canonical in {item['id']}")

    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v11 eval-contract addendum: citation_json regenerated with harness-exact tool replies "
            "(canonical_json content, sorted keys, compact separators, name field present) after a "
            "controlled replay proved the v10 adapter flips from bare JSON to prose under the "
            "harness's canonical tool-result formatting"
        ),
        "source": "first-party deterministic synthetic tool trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": sum(counts.values()),
        "scenario_families": {FAMILY: EXAMPLES_PER_FAMILY},
        "split_rule": "within the family, indices 0-191 train, 192-215 validation, 216-239 test; no group crosses split",
        "splits": {name: {**files[name], "tool_call_messages": total_calls[name]} for name in files},
        "tools": dict(tool_names),
        "design_notes": {
            "tool_reply_shape": "exact harness form: role tool, tool_call_id, name, canonical_json content",
            "replaces": "v5 citation_json train rows are replaced by these rows in release v11 (not supplemented)",
            "final_answers": "bare json.dumps objects with no fences or tags",
            "dev_fixture_exclusion": "literal dev prompt contents and summaries never appear in payloads",
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
