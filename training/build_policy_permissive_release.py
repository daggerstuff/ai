"""Build a local W&B research release from sources the project classifies permissive-use."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "ai/data/curated/recovery/gold-20261017/train_master_gold.recovered_candidate.jsonl"
LEDGER = ROOT / "docs/training/execution-2026-10-17/source-decision-ledger.jsonl"
OUT = ROOT / "ai/data/curated/recovery/gold-20261017/wandb-policy-permissive-20261018"
MANIFEST = ROOT / "docs/training/execution-2026-10-18/wandb-policy-permissive-sources-release.json"
EXPECTED_CANDIDATE_SHA = "4f0fd4c5fcfdffeb5434057ff4e122356256f9775540a3b41c1421a75e507663"
EXPECTED_CANDIDATE_ROWS = 190139
SOURCES = {"psycheval", "dsm_vector_space"}
EXPECTED_COUNTS = {"psycheval": 1977, "dsm_vector_space": 116}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"),
    "card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "url": re.compile(r"https?://\S+"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip("`\"'").lower())


def policy_lookup() -> tuple[dict[str, tuple[dict[str, Any], Path]], dict[str, str]]:
    lookup: dict[str, tuple[dict[str, Any], Path]] = {}
    hashes: dict[str, str] = {}
    for path in (ROOT / "ai/data").glob("source_family_*_manifest.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for source in data.get("sources", []):
            aliases = [
                source.get("source_id"),
                source.get("title"),
                source.get("ds_alias"),
                *(source.get("aliases") or []),
            ]
            for alias in aliases:
                for item in alias if isinstance(alias, list) else [alias]:
                    if item:
                        lookup[normalize(item)] = (source, path)
    for label in SOURCES:
        record, path = lookup[normalize(label)]
        if record.get("rights_decision", {}).get("decision") != "permissive-use":
            raise ValueError(f"Project policy does not classify {label} as permissive-use")
        if record.get("contribution_mode") != "supervised-finetuning":
            raise ValueError(f"Project policy does not allow SFT for {label}")
        hashes[label] = sha256(path.read_bytes())
    return lookup, hashes


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records))
    os.chmod(path, 0o600)


def load_references() -> dict[int, dict[str, Any]]:
    references: dict[int, dict[str, Any]] = {}
    for line in LEDGER.open(encoding="utf-8"):
        entry = json.loads(line)
        if entry["source_label"] in SOURCES:
            for reference in entry["candidate_rows"]:
                references[reference["row_number_1based"]] = reference
    return references


def select_rows(
    candidate_bytes: bytes, references: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    all_text: list[str] = []
    for row_number, raw_line in enumerate(candidate_bytes.splitlines(), start=1):
        row = json.loads(raw_line)
        if row.get("source") not in SOURCES:
            continue
        reference = references.get(row_number)
        if reference is None or sha256(raw_line) != reference["row_sha256"]:
            raise ValueError(f"ledger mismatch at candidate row {row_number}")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"invalid messages at candidate row {row_number}")
        projected = []
        for message in messages:
            if message.get("role") == "system":
                continue
            role, content = message.get("role"), message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                raise ValueError(f"invalid rendered message at candidate row {row_number}")
            projected.append({"role": role, "content": content, "weight": 0 if role == "user" else 1})
        roles = [message["role"] for message in projected]
        if not roles or roles[-1] != "assistant" or "user" not in roles or "assistant" not in roles:
            raise ValueError(f"invalid assistant-target sequence at candidate row {row_number}")
        all_text.extend(message["content"] for message in projected)
        group_hash = json_hash([row["source"], messages])
        selected.append(
            {
                "candidate_row_number": row_number,
                "candidate_row_sha256": reference["row_sha256"],
                "messages_sha256": reference["messages_sha256"],
                "content_group_sha256": group_hash,
                "source": row["source"],
                "payload": {"messages": projected},
            }
        )

    actual_counts = {source: sum(row["source"] == source for row in selected) for source in SOURCES}
    if len(selected) != EXPECTED_TOTAL or actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"source census changed: {actual_counts}")
    if len({row["content_group_sha256"] for row in selected}) != len(selected):
        raise ValueError("exact duplicate content exists; resolve before splitting")
    return selected, all_text


def make_splits(
    selected: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    split_by_row: dict[int, str] = {}
    for source in sorted(SOURCES):
        group = sorted(
            (row for row in selected if row["source"] == source), key=lambda row: row["content_group_sha256"]
        )
        validation_n = round(len(group) * 0.1)
        test_n = round(len(group) * 0.1)
        train_n = len(group) - validation_n - test_n
        for index, row in enumerate(group):
            split_by_row[row["candidate_row_number"]] = (
                "train" if index < train_n else "validation" if index < train_n + validation_n else "test"
            )

    split_payloads: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    internal_rows, rowrefs = [], []
    group_sets: dict[str, set[str]] = {name: set() for name in split_payloads}
    for row in selected:
        split = split_by_row[row["candidate_row_number"]]
        split_payloads[split].append(row["payload"])
        group_sets[split].add(row["content_group_sha256"])
        internal_rows.append(
            {
                "candidate_row_sha256": row["candidate_row_sha256"],
                "content_group_sha256": row["content_group_sha256"],
                "split": split,
                "source_label": row["source"],
            }
        )
        rowrefs.append(
            {
                "candidate_row_sha256": row["candidate_row_sha256"],
                "content_group_sha256": row["content_group_sha256"],
                "provisional_split": split,
            }
        )
    return split_payloads, group_sets, internal_rows, rowrefs


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes())


def build_manifest(
    selected: list[dict[str, Any]],
    policy_hashes: dict[str, str],
    candidate_sha: str,
    artifacts: dict[str, Any],
    pii_counts: dict[str, int],
) -> dict[str, Any]:
    paths = artifacts["paths"]
    counts = artifacts["counts"]
    group_sets = artifacts["group_sets"]
    rowref_path = artifacts["rowref_path"]
    heldout_path = artifacts["heldout_path"]
    internal_path = artifacts["internal_path"]
    return {
        "schema_version": "g04-release-manifest-v1",
        "status": "approved_for_research_training",
        "upload_gate": True,
        "release_id": "wandb-policy-permissive-sources-20261018-v1",
        "provider": "wandb",
        "scope": {
            "use": "research_fine_tuning_candidate_only",
            "clinical_or_production_readiness": False,
            "candidate_rows": len(selected),
            "source_groups": len(selected),
        },
        "source_candidate_sha256": candidate_sha,
        "source_candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": candidate_sha,
            "row_count": EXPECTED_CANDIDATE_ROWS,
        },
        "partitions": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
                "row_count": counts[name],
                "source_group_count": len(group_sets[name]),
                "format": "weighted_messages_jsonl_v1",
                "split": name,
            }
            for name, path in paths.items()
        },
        "payload_contract": {
            "format": "weighted_messages_jsonl_v1",
            "user_weight": 0,
            "assistant_weight": 1,
            "loss_mask": "assistant-turn-only",
            "projection": "drop source system messages; preserve user/assistant turns and weight assistant targets",
        },
        "evidence": {
            "group_disjointness": {
                "status": "verified",
                "group_key": "content_group_sha256",
                "train_group_count": len(group_sets["train"]),
                "validation_group_count": len(group_sets["validation"]),
                "test_group_count": len(group_sets["test"]),
                "pairwise_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                "evidence_path": str(rowref_path.relative_to(ROOT)),
                "evidence_sha256": file_hash(rowref_path),
                "reserved_eval_group_count": 0,
                "heldout_group_intersections": {"train": 0, "validation": 0, "test": 0},
                "heldout_exclusions_path": str(heldout_path.relative_to(ROOT)),
                "heldout_exclusions_sha256": file_hash(heldout_path),
            },
            "privacy": {
                "basic_pattern_screen": {"status": "passed", "counts": pii_counts},
                "clinical_ner_screen": "not run",
                "limitations": [
                    "Pattern screen does not certify absence of sensitive information.",
                    "clinical_reviewed source metadata is not treated as clinical validation.",
                ],
            },
            "source_origin": {
                "status": "existing_project_manifest_permissive_use_decisions",
                "policy_entries": {
                    source: {
                        "source_id": "SRC-036" if source == "psycheval" else "SRC-044",
                        "manifest_sha256": policy_hashes[source],
                    }
                    for source in sorted(SOURCES)
                },
                "decision_basis": (
                    "Existing project source-family manifest entries classify these two sources as "
                    "permissive-use and supervised-finetuning; research experiment only."
                ),
            },
        },
        "internal_manifest": {
            "path": str(internal_path.relative_to(ROOT)),
            "sha256": file_hash(internal_path),
            "row_count": len(artifacts["internal_rows"]),
        },
        "limitations": [
            "Research-only SFT experiment; not clinical or production readiness.",
            "Groups are exact-message hashes; no original seed lineage key exists for these source rows.",
            "No semantic near-duplicate certification.",
            "System messages were removed to satisfy the W&B weighted user/assistant payload contract.",
        ],
        "created_at_utc": "2026-10-18T15:35:00Z",
    }


def main() -> None:
    candidate_bytes = CANDIDATE.read_bytes()
    candidate_sha = sha256(candidate_bytes)
    if candidate_sha != EXPECTED_CANDIDATE_SHA:
        raise ValueError("candidate identity mismatch; refusing to build")
    _, policy_hashes = policy_lookup()
    selected, all_text = select_rows(candidate_bytes, load_references())
    split_payloads, group_sets, internal_rows, rowrefs = make_splits(selected)
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(OUT, 0o700)
    paths = {name: OUT / f"{name}.jsonl" for name in split_payloads}
    for name, records in split_payloads.items():
        write_jsonl(paths[name], records)
    internal_path, rowref_path, heldout_path = (
        OUT / "manifest_internal.jsonl",
        OUT / "rowrefs.jsonl",
        OUT / "heldout.jsonl",
    )
    write_jsonl(internal_path, internal_rows)
    write_jsonl(rowref_path, rowrefs)
    write_jsonl(heldout_path, [])
    pii_counts = {name: len(pattern.findall("\n".join(all_text))) for name, pattern in PATTERNS.items()}
    if any(pii_counts.values()):
        raise ValueError(f"identifier-pattern hits; do not upload: {pii_counts}")
    counts = {name: len(records) for name, records in split_payloads.items()}
    artifacts = {
        "paths": paths,
        "counts": counts,
        "group_sets": group_sets,
        "rowref_path": rowref_path,
        "heldout_path": heldout_path,
        "internal_path": internal_path,
        "internal_rows": internal_rows,
    }
    manifest = build_manifest(selected, policy_hashes, candidate_sha, artifacts, pii_counts)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.chmod(MANIFEST, 0o600)
    sys.stdout.write(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "counts": counts,
                "sources": {source: sum(row["source"] == source for row in selected) for source in SOURCES},
                "exact_duplicate_groups": 0,
                "identifier_pattern_counts": pii_counts,
                "candidate_sha256": candidate_sha,
                "release_train_sha256": manifest["partitions"]["train"]["sha256"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
