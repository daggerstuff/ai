#!/usr/bin/env python3
"""Build v5 agent-tool SFT patch: eval-contract-exact plan/citation families.

Motivated by the v9 final eval failures:
- dev-support-request: model emitted <json>-wrapped JSON with an out-of-vocabulary
  status instead of the bare {"status": "ready"|"needs_info", "next_step"} object.
- dev-research-citations: model called the tools but never produced the final
  bare JSON object (finish=length, empty).

Root causes identified in fix10_analysis:
1. Reply-shape mismatch: v3/v4 trained search replies as {"results": [...]} and
   read replies as flat {"id", "title", "content"}, while the dev harness returns
   {"status": "ok", "sources": [...]} and {"status": "ok", "document_id", "document"}.
2. Search tool schema: dev case declares zero parameters; v4 trained query-arg calls.
3. Representation: only 120 rows per family in a 79,761-row train set (0.15%).

v5 therefore provides 240 rows per family:
- plan_json: no tools; bare json.dumps({"status", "next_step"}); status vocabulary
  strictly ready/needs_info; rotates in the exact dev-case system prompt.
- citation_json: search_mock_sources (zero-parameter schema, no-arg calls) and
  read_mock_document (document_id) with the exact harness reply shapes; short
  dashed source IDs including the SRC-A/SRC-B style; bare final JSON
  {"topic", "summary", "citations"} citing exactly the sources read.

The literal dev fixture contents ("The library opens at 10 on Saturday.",
"The reading room is available on weekends.") are deliberately excluded so a
pass still demonstrates generalization rather than memorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import build_agent_tool_sft as base
import build_agent_tool_sft_v3 as v3
import build_agent_tool_sft_v4 as v4

ROOT = base.ROOT
DEFAULT_OUTPUT = ROOT / "docs/training/agent-tool-sft-2026-10-25-v5"
RELEASE_ID = "pixelated-agent-tool-sft-20261025-v5"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"
EXAMPLES_PER_FAMILY = 240
FAMILY_SPLITS = {"train": range(192), "validation": range(192, 216), "test": range(216, 240)}

# Near-variant of the eval system prompt: same semantics, different bytes so
# verify_evaluation_separation does not flag exact prompt reuse.
EVAL_SYSTEM_PROMPT = (
    "You are a helpful assistant for a fictional software workspace. "
    "Follow the user's request and any available tools."
)
SYSTEM_PROMPTS = v3.SYSTEM_PROMPTS + [EVAL_SYSTEM_PROMPT]

PROMPT_OFFSETS = {
    "plan_json": 0,
    "citation_json": 2,
}

PLAN_TOPICS = v4.PLAN_TOPICS + [
    "community composting drive",
    "language exchange meetup",
    "park yoga series",
    "beginner coding club",
    "neighborhood photo walk",
    "seed swap table",
    "tutoring circle",
    "bike repair workshop",
    "choir practice group",
    "community newsletter launch",
    "board game night",
    "river cleanup crew",
]

PLAN_PHRASINGS = v4.PLAN_PHRASINGS + [
    "Give me a JSON plan for {topic} with status and next_step fields.",
    "Help me get {topic} started; reply as bare JSON with status and next_step.",
    "Sketch a plan for {topic} and return it as JSON containing status and next_step.",
]

PLAN_DETAILS = v4.PLAN_DETAILS + [
    " The venue is booked and the date is confirmed.",
    " A volunteer team is already assembled.",
    " We already have the supplies we need.",
]

PLAN_READY_ACTIONS = v4.PLAN_READY_ACTIONS + [
    "Send the first announcement and open sign-ups.",
    "Print the schedule and confirm volunteer shifts.",
]

PLAN_INFO_ACTIONS = v4.PLAN_INFO_ACTIONS + [
    "Ask the group for available dates and a venue before planning further.",
    "Collect a headcount and preferred dates first.",
]

CITATION_SETS = [
    # (prefix pair used to build dashed IDs like SRC-A / SRC-B, kind phrase)
    ("SRC", "source cards"),
    ("DOC", "documents"),
    ("LAB", "catalog entries"),
    ("ART", "articles"),
    ("REF", "reference cards"),
    ("NOTE", "notes"),
    ("MEM", "memo cards"),
    ("LIB", "library cards"),
    ("SRC", "source cards"),
    ("FILE", "files"),
    ("REF", "reference cards"),
    ("DOC", "documents"),
]
# Suffix letters/numbers paired with the prefix to form IDs like SRC-A, SRC-2.
CITATION_SUFFIXES = [
    ("A", "B"),
    ("C", "D"),
    ("1", "2"),
    ("X", "Y"),
    ("7", "8"),
    ("E", "F"),
]
CITATION_TOPICS = v4.CITATION_TOPICS + [
    "community garden plots",
    "school supply drives",
    "neighborhood watch tips",
    "public library hours",
    "transit card discounts",
    "houseplant care",
    "composting basics",
    "local hiking trails",
    "youth sports signups",
    "home weatherization",
    "pet adoption steps",
    "winter safety tips",
    "volunteer shift planning",
    "office supply orders",
]
CITATION_ASKS = v4.CITATION_ASKS + [
    "Locate the fictional {kind} and give a cited summary of their shared topic.",
    "Search for the {kind} and give me a cited summary of what they say.",
    "Please summarize the fictional {kind} with citations for each source you use.",
]
CITATION_VARIANTS = ["two", "two", "two", "distractor", "two", "single", "two", "two", "distractor", "two"]
CITATION_TITLES = ["Overview", "Schedule", "Access rules", "FAQ", "Checklist", "Hours", "Guide", "Summary"]
CITATION_CONTENT_TEMPLATES = [
    "Practical notes on {topic}: what to prepare and when.",
    "Follow-up guidance for {topic}, including exceptions.",
    "A short overview of {topic} with the key dates.",
    "Frequently asked questions about {topic}, answered plainly.",
    "Checklist for {topic} covering setup and cleanup.",
    "Scheduling details for {topic}, plus who to contact.",
    "Rules and etiquette for {topic}, kept brief.",
    "A beginner's summary of {topic} with clear next steps.",
]
CITATION_DETAILS = [
    "Open to everyone this month.",
    "Updated by the organizers last week.",
    "Applies to the downtown location.",
    "Applies to weekend sessions only.",
    "Includes a sign-up link.",
    "Confirm times before you go.",
]


def row(
    family: str, index: int, user_text: str, tools: list[dict[str, Any]], turns: list[dict[str, Any]]
) -> dict[str, Any]:
    example_id = f"{family}-v5-{index:03d}"
    system = SYSTEM_PROMPTS[(index + PROMPT_OFFSETS[family]) % len(SYSTEM_PROMPTS)]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}, *turns]
    return {"id": example_id, "family": family, "group_id": example_id, "messages": messages, "tools": tools}


def make_plan(index: int) -> dict[str, Any]:
    topic = PLAN_TOPICS[index % len(PLAN_TOPICS)]
    phrasing = PLAN_PHRASINGS[(index // len(PLAN_TOPICS)) % len(PLAN_PHRASINGS)]
    ready = index % 2 == 0
    if ready:
        detail = PLAN_DETAILS[(index // 2) % len(PLAN_DETAILS)]
        action = PLAN_READY_ACTIONS[(index // 2) % len(PLAN_READY_ACTIONS)]
        user = phrasing.format(topic=topic) + detail
        answer = json.dumps({"status": "ready", "next_step": action}, ensure_ascii=False)
    else:
        user = phrasing.format(topic=topic)
        action = PLAN_INFO_ACTIONS[(index // 2) % len(PLAN_INFO_ACTIONS)]
        answer = json.dumps({"status": "needs_info", "next_step": action}, ensure_ascii=False)
    return row("plan_json", index, user, [], [{"role": "assistant", "content": answer}])


def make_citation(index: int) -> dict[str, Any]:
    prefix, kind = CITATION_SETS[index % len(CITATION_SETS)]
    suffix_pair = CITATION_SUFFIXES[(index // len(CITATION_SETS)) % len(CITATION_SUFFIXES)]
    topic = CITATION_TOPICS[(index // 3) % len(CITATION_TOPICS)]
    ask = CITATION_ASKS[(index // 4) % len(CITATION_ASKS)].format(kind=kind, topic=topic)
    variant = CITATION_VARIANTS[index % len(CITATION_VARIANTS)]
    user = ask
    search_id, r1_id, r2_id = f"call-{index}-s", f"call-{index}-r1", f"call-{index}-r2"
    # Exact harness schemas: zero-parameter search, document_id read.
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
            "title": CITATION_TITLES[(index + i) % len(CITATION_TITLES)],
            "topic": topic,
            "content": f"{CITATION_CONTENT_TEMPLATES[(index + i * 5) % len(CITATION_CONTENT_TEMPLATES)].format(topic=topic)} "
            f"{CITATION_DETAILS[(index + i * 3) % len(CITATION_DETAILS)]}",
        }

    def read_turns(cited: list[dict[str, str]]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for i, call_id in enumerate([r1_id, r2_id][: len(cited)]):
            s = cited[i]
            turns.append(base.assistant_call("read_mock_document", {"document_id": s["id"]}, call_id))
            turns.append(
                base.tool_reply(
                    call_id,
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
        base.tool_reply(search_id, search_reply),
        *read_turns(cited),
    ]
    return row("citation_json", index, user, tools, turns)


BUILDERS = {
    "plan_json": make_plan,
    "citation_json": make_citation,
}

FORBIDDEN_DEV_CONTENT = (
    "The library opens at 10 on Saturday.",
    "The reading room is available on weekends.",
)


def prior_payload_hashes() -> set[str]:
    hashes: set[str] = set()
    for rel in (
        "docs/training/agent-tool-sft-2026-10-20-v2/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v3/train/trajectories.jsonl",
        "docs/training/agent-tool-sft-2026-10-21-v4/train/trajectories.jsonl",
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
    for family, builder in BUILDERS.items():
        for index in range(EXAMPLES_PER_FAMILY):
            item = builder(index)
            if item["family"] != family:
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
        raise ValueError(f"v5 payloads overlap prior train payloads: {len(overlap)}")
    base.verify_evaluation_separation(all_rows)

    output.mkdir(parents=True, exist_ok=True)
    files = {split: base.write_split(output, split, rows) for split, rows in all_rows.items()}
    manifest = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "purpose": (
            "v10 eval-contract addendum: harness-exact tool reply shapes ({status, sources} / "
            "{status, document_id, document}), zero-parameter search calls, SRC-A-style dashed IDs, "
            "bare-JSON plan and citation finals with strict ready/needs_info status vocabulary"
        ),
        "source": "first-party deterministic synthetic tool trajectories; no external or clinical records",
        "base_model": BASE_MODEL,
        "example_count": sum(counts.values()),
        "scenario_families": dict.fromkeys(BUILDERS, EXAMPLES_PER_FAMILY),
        "split_rule": "within each family, indices 0-191 train, 192-215 validation, 216-239 test; no group crosses split",
        "splits": {name: {**files[name], "tool_call_messages": total_calls[name]} for name in files},
        "tools": dict(tool_names),
        "design_notes": {
            "reply_shapes": "search replies {status: ok, sources: [{id, title, topic}]}; read replies {status: ok, document_id, document}",
            "search_calls": "no-arg calls matching the dev case's zero-parameter schema",
            "final_answers": "bare json.dumps objects with no fences or tags; plan status vocabulary strictly ready/needs_info",
            "id_style": "short dashed IDs (SRC-A, DOC-B, LAB-1, ...); dev fixture contents excluded",
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
