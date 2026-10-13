"""Prepare an immutable, local-only paired evaluation for the Glimmer SFT pilot.

This command never calls a model or uploads data. Each held-out conversation
contributes its final assistant turn, with the reference excluded from the
rendered prompt. Golden controls are separate from the corpus metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
AI_ROOT = Path(__file__).resolve().parents[3]
PILOT = AI_ROOT / "training/output/arc_corpus/export/fireworks_pilot_v1"
TEMPLATE = Path(__file__).with_name("muse_glimmer_template.json")
QUESTIONS = AI_ROOT / "research/experiments/evals/golden_questions.json"
SEED_MAP = AI_ROOT / "training/arc_plans/seed_map.jsonl"
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    """Hash a source without loading the complete file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL strictly; a malformed row must not disappear silently."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_pilot(pilot: Path) -> dict[str, list[dict[str, Any]]]:
    """Check frozen hashes, counts, provenance order, and split isolation."""
    identity = json.loads((pilot / "identity.json").read_text(encoding="utf-8"))
    names = [f"{split}.jsonl" for split in SPLITS] + ["manifest_internal.jsonl"]
    for name in names:
        if sha256(pilot / name) != identity["files"][name]["sha256"]:
            raise ValueError(f"Frozen pilot hash mismatch: {name}")
    rows = {split: read_jsonl(pilot / f"{split}.jsonl") for split in SPLITS}
    manifest = read_jsonl(pilot / "manifest_internal.jsonl")
    provenance: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    groups: dict[tuple[str, str], str] = {}
    arcs: dict[str, str] = {}
    for record in manifest:
        split = record["split"]
        group = (record["source_family"], record["source_id"])
        if groups.setdefault(group, split) != split or arcs.setdefault(record["arc_id"], split) != split:
            raise ValueError("Pilot seed or arc crosses splits")
        provenance[split].append(record)
    if len(manifest) != identity["files"]["manifest_internal.jsonl"]["rows"]:
        raise ValueError("Pilot provenance count mismatch")
    for split in SPLITS:
        if len(rows[split]) != identity["files"][f"{split}.jsonl"]["rows"]:
            raise ValueError(f"Pilot row count mismatch: {split}")
        if len(provenance[split]) != len(rows[split]):
            raise ValueError(f"Pilot provenance count mismatch: {split}")
    return provenance


def render_prompt(messages: list[dict[str, Any]], template: dict[str, Any]) -> str:
    """Use the provider's pinned template/date, never a dynamic HF template."""
    system = template["system_prompt"].format(date=template["system_prompt_date_seen_during_training"])
    rendered = template["bos"] + template["turns"]["system"].format(content=system)
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected or not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Expected nonempty alternating user/assistant turns")
        rendered += template["turns"][expected].format(content=message["content"])
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Evaluation prompt must end with a user turn")
    return rendered + template["generation_prefix"]


def audit_lineage(provenance: dict[str, list[dict[str, Any]]], seed_map: Path) -> dict[str, Any]:
    """Resolve arc fallback IDs through actual seed reuse, including variation passes."""
    mappings: dict[str, set[str]] = {}
    for row in read_jsonl(seed_map):
        if row["status"] in {"written", "skipped"}:
            mappings.setdefault(row["arc_id"], set()).add(re.sub(r":v\d+$", "", row["key"]))
    keys: dict[str, str | None] = {}
    splits: dict[str, set[str]] = {}
    for split, rows in provenance.items():
        for row in rows:
            arc = row["arc_id"]
            options = mappings.get(arc, set())
            key = None
            if row["source_id"] != arc:
                key = f"{row['source_family']}:{row['source_id']}"
            elif len(options) == 1:
                key = f"seed_map:{next(iter(options))}"
            keys[arc] = key
            if key is not None:
                splits.setdefault(key, set()).add(split)
    excluded = {}
    for row in provenance["test"]:
        arc = row["arc_id"]
        key = keys[arc]
        if key is None:
            excluded[arc] = "unresolved_or_ambiguous_seed"
        elif splits[key] & {"train", "val"}:
            excluded[arc] = "seed_seen_in_train_or_validation"
    return {
        "excluded_test_arcs": excluded,
        "cross_split_seed_groups": {key: sorted(value) for key, value in splits.items() if len(value) > 1},
        "unresolved_pilot_arcs": sorted(arc for arc, key in keys.items() if key is None),
        "scope": "Generator seed identities; semantic duplicates and external control exposure are not established",
    }


def make_cases(pilot: Path, template: dict[str, Any], questions: Path, lineage: dict[str, Any]) -> list[dict[str, Any]]:
    """Create paired inference inputs; reference material is explicitly separate."""
    provenance = validate_pilot(pilot)
    cases = []
    for row, source in zip(read_jsonl(pilot / "test.jsonl"), provenance["test"], strict=True):
        if source["arc_id"] in lineage["excluded_test_arcs"]:
            continue
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise ValueError("Held-out conversation must end with a nonempty assistant reference")
        cases.append(
            {
                "id": f"heldout:{source['arc_id']}:s{source['session_n']}",
                "suite": "heldout_final_turn",
                "category": source["difficulty"],
                "prompt": render_prompt(messages[:-1], template),
                "reference": messages[-1]["content"],
                "source_family": source["source_family"],
                "source_id": source["source_id"],
            }
        )
    for question in json.loads(questions.read_text(encoding="utf-8")):
        cases.append(
            {
                "id": f"control:{question['id']}",
                "suite": "golden_controls",
                "category": question["category"],
                "prompt": render_prompt([{"role": "user", "content": question["prompt"]}], template),
                "expected_behavior": question["expected_behavior"],
            }
        )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate evaluation case ID")
    for case in cases:
        case["prompt_sha256"] = hashlib.sha256(case["prompt"].encode()).hexdigest()
    return cases


def atomic_write(path: Path, content: str) -> None:
    """Replace a generated file only after all its bytes reach disk."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(
    pilot: Path, template_path: Path, questions: Path, output: Path, seed_map: Path = SEED_MAP
) -> dict[str, Any]:
    """Build in a new directory, refusing to overwrite an earlier experiment."""
    names = ("identity.json", "manifest_internal.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")
    sources = [pilot / name for name in names]
    sources.extend([template_path, questions, seed_map])
    before = {str(path): sha256(path) for path in sources}
    template = json.loads(template_path.read_text(encoding="utf-8"))
    lineage = audit_lineage(validate_pilot(pilot), seed_map)
    cases = make_cases(pilot, template, questions, lineage)
    if not any(case["suite"] == "heldout_final_turn" for case in cases):
        raise ValueError("No seed-isolated held-out cases remain")
    if before != {str(path): sha256(path) for path in sources}:
        raise ValueError("Source changed during evaluation preparation")
    output.mkdir(parents=True, exist_ok=False)
    serialized = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    atomic_write(output / "cases.jsonl", serialized)
    manifest = {
        "status": "prepared_not_evaluated",
        "source_sha256": before,
        "cases_sha256": sha256(output / "cases.jsonl"),
        "case_count": len(cases),
        "suites": dict(Counter(case["suite"] for case in cases)),
        "lineage": lineage,
        "system_date": template["system_prompt_date_seen_during_training"],
        "decoding": {"temperature": 0, "max_tokens": 1024, "stop": [template["stop_token"]]},
        "comparison": {
            "base": "meta-models/Muse-Glimmer-30B",
            "sft": "accounts/linencloset/models/arc-pilot-v1-therapist-sft",
            "job": "wwopccov",
            "required": "Same prompts, tokenizer revision, precision, runtime and decoding in both arms",
        },
        "execution": {
            "input_field": "prompt",
            "never_send": ["reference", "expected_behavior"],
            "smoke": "Five case IDs spanning both suites, identical in both arms",
            "checkpoint": "Persist each arm/case/prompt-hash response; reject resume config mismatches",
            "context": "Count tokens before inference; reject overflow instead of silently truncating",
            "length_finish": "Flag as truncated; do not score as a complete response",
        },
        "review": {
            "heldout": ["ledger JSON/schema", "ledger grounding", "continuity", "response quality"],
            "controls": ["risk handling", "unnecessary crisis escalation", "response quality"],
            "report": "Paired wins/ties/losses by suite and seed; separate malformed/truncated outputs",
            "limitations": "Mechanical or keyword scores do not establish clinical quality or safety",
        },
    }
    atomic_write(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """CLI entry point; no credentials or model libraries are needed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=PILOT)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--seed-map", type=Path, default=SEED_MAP)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest = prepare(args.pilot_dir, args.template, args.questions, args.output, args.seed_map)
    LOGGER.info("Prepared %s cases in %s; no inference performed", manifest["case_count"], args.output)


if __name__ == "__main__":
    main()
