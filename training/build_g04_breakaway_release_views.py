"""Build a private, provider-neutral G04 release and local provider renderings."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/training/execution-2026-10-17"
SOURCE_CANDIDATE = ROOT / "ai/data/curated/recovery/gold-20261017/train_master_gold.recovered_candidate.jsonl"
SOURCE_PACKAGE = ROOT / "ai/data/curated/recovery/gold-20261017/g03-edge-review-package-20261017T1626Z"
FREEZE = DOCS / "g03-edge-case-breakaway-frozen-2026-10-19.json"
LEGACY_MANIFEST = DOCS / "g04-release-manifest.json"
RENDER_REPORT = DOCS / "g04-edge-render-report.json"
EXCLUSIONS = DOCS / "g03-edge-case-heldout-exclusions.jsonl"
EXPECTED_CANDIDATE_SHA256 = "4f0fd4c5fcfdffeb5434057ff4e122356256f9775540a3b41c1421a75e507663"
EXPECTED_SPLITS = {"train": 410, "validation": 72, "test": 40}
SPLIT_FILES = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
LOGGER = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def project_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("empty source conversation")
    result = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            raise ValueError("G03 source contains an unsupported or empty message")
        result.append({"role": role, "content": content, "weight": 0 if role == "user" else 1})
    return result


def payload_signature(messages: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json({"messages": messages}))


def load_pinned_source() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(RENDER_REPORT.read_text(encoding="utf-8"))
    if sha256_file(SOURCE_CANDIDATE) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("source candidate hash differs from the frozen G03 identity")
    if freeze["status"] != "frozen_local_research_only_provider_transfer_unqualified":
        raise ValueError("unexpected G03 transfer status; inspect before rendering")
    if freeze["candidate"]["sha256"] != EXPECTED_CANDIDATE_SHA256 or legacy.get("upload_gate") is not False:
        raise ValueError("frozen source identity or closed legacy upload gate did not match")
    internal = read_jsonl(SOURCE_PACKAGE / "manifest_internal.jsonl")
    if len(internal) != sum(EXPECTED_SPLITS.values()):
        raise ValueError("internal source manifest row count mismatch")
    requested = {int(row["candidate_row_number_1based"]) for row in internal}
    candidate_rows: dict[int, dict[str, Any]] = {}
    candidate_hashes: dict[int, str] = {}
    hasher = hashlib.sha256()
    with SOURCE_CANDIDATE.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            hasher.update(raw)
            if number in requested:
                candidate_rows[number] = json.loads(raw.decode("utf-8"))
                candidate_hashes[number] = sha256_bytes(raw.rstrip(b"\r\n"))
    if hasher.hexdigest() != EXPECTED_CANDIDATE_SHA256 or set(candidate_rows) != requested:
        raise ValueError("candidate changed or contains an invalid row reference")
    for reference in internal:
        number = int(reference["candidate_row_number_1based"])
        if candidate_hashes[number] != reference["candidate_row_sha256"]:
            raise ValueError("candidate row hash differs from its frozen lineage reference")
    return freeze, report, internal, candidate_rows


def make_reference_index(
    internal: list[dict[str, Any]], candidate_rows: dict[int, dict[str, Any]]
) -> dict[str, dict[str, deque[dict[str, Any]]]]:
    index: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(lambda: defaultdict(deque))
    for reference in internal:
        split = reference["split"]
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"unexpected split label: {split}")
        candidate = candidate_rows[int(reference["candidate_row_number_1based"])]
        signature = payload_signature(project_messages(candidate.get("messages") or []))
        index[split][signature].append(reference)
    return index


def render_split(
    source_split: str,
    references: dict[str, dict[str, deque[dict[str, Any]]]],
    candidate_rows: dict[int, dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    source_path = SOURCE_PACKAGE / SPLIT_FILES[source_split]
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if sha256_file(source_path) != freeze["package"]["files"][SPLIT_FILES[source_split]]["sha256"]:
        raise ValueError(f"frozen source payload hash mismatch: {source_path}")
    payload_rows = read_jsonl(source_path)
    if len(payload_rows) != EXPECTED_SPLITS[source_split]:
        raise ValueError(f"frozen source split count mismatch: {source_split}")
    canonical_rows = []
    fireworks_rows = []
    lineage = []
    mappings = []
    for payload in payload_rows:
        messages = payload.get("messages")
        signature = payload_signature(messages) if isinstance(messages, list) else ""
        queue = references[source_split].get(signature)
        if not queue:
            raise ValueError(f"rendered row has no lineage match: {source_split}")
        reference = queue.popleft()
        candidate = candidate_rows[int(reference["candidate_row_number_1based"])]
        if project_messages(candidate["messages"]) != messages:
            raise ValueError("frozen payload differs from pinned source candidate")
        canonical_row = {
            "messages": [
                {"role": message["role"], "content": message["content"], "loss_weight": message["weight"]}
                for message in messages
            ]
        }
        canonical_hash = sha256_bytes(canonical_json(canonical_row))
        fireworks_row = {"messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
        row_id = reference["candidate_row_sha256"]
        canonical_rows.append(canonical_row)
        fireworks_rows.append(fireworks_row)
        lineage.append(
            {
                "row_id": row_id,
                "candidate_row_number_1based": reference["candidate_row_number_1based"],
                "arc_id": reference["arc_id"],
                "source_group_sha256": reference["seed_group_sha256"],
                "messages_sha256": reference["messages_sha256"],
                "seed_source_class": reference["seed_source_class"],
                "seed_input_artifacts": reference["seed_input_artifacts"],
                "source_origin_assessment": reference["source_origin_assessment"],
                "machine_audit_verdict": reference["machine_audit_verdict"],
                "split": "dev" if source_split == "validation" else source_split,
                "canonical_row_sha256": canonical_hash,
            }
        )
        mappings.append(
            {
                "row_id": row_id,
                "canonical_row_sha256": canonical_hash,
                "wandb_payload_row_sha256": sha256_bytes(canonical_json(payload)),
                "fireworks_payload_row_sha256": sha256_bytes(canonical_json(fireworks_row)),
                "split": source_split,
            }
        )
    if any(references[source_split].values()):
        raise ValueError(f"unmatched candidate source rows remain in {source_split}")
    return canonical_rows, payload_rows, fireworks_rows, lineage, mappings


def write_split_views(
    output: Path,
    references: dict[str, dict[str, deque[dict[str, Any]]]],
    candidate_rows: dict[int, dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    counts = {}
    all_lineage = []
    all_mappings = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    for source_split, filename in SPLIT_FILES.items():
        canonical_rows, wandb_rows, fireworks_rows, lineage, mappings = render_split(
            source_split, references, candidate_rows
        )
        canonical_split = "dev" if source_split == "validation" else source_split
        write_jsonl(output / "canonical" / f"{canonical_split}.jsonl", canonical_rows)
        write_jsonl(output / "providers/wandb" / filename, wandb_rows)
        write_jsonl(output / "providers/fireworks" / filename, fireworks_rows)
        counts[canonical_split] = len(canonical_rows)
        all_lineage.extend(lineage)
        all_mappings.extend(mappings)
        for row in lineage:
            group_splits[row["source_group_sha256"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("source group crossed the canonical split boundary")
    if counts != {"train": 410, "dev": 72, "test": 40}:
        raise ValueError(f"unexpected canonical split census: {counts}")
    write_jsonl(output / "lineage.jsonl", all_lineage)
    write_jsonl(
        output / "splits.jsonl",
        [
            {"row_id": row["row_id"], "source_group_sha256": row["source_group_sha256"], "split": row["split"]}
            for row in all_lineage
        ],
    )
    write_jsonl(output / "providers/wandb/row_map.jsonl", all_mappings)
    write_jsonl(output / "providers/fireworks/row_map.jsonl", all_mappings)
    return (
        counts,
        all_lineage,
        all_mappings,
        [{"group_sha256": group, "split": next(iter(splits))} for group, splits in sorted(group_splits.items())],
    )


def write_support_files(
    output: Path, freeze: dict[str, Any], render_report: dict[str, Any], lineage: list[dict[str, Any]]
) -> int:
    exclusions = read_jsonl(EXCLUSIONS)
    if sha256_file(EXCLUSIONS) != freeze["selection"]["excluded_sha256"]:
        raise ValueError("heldout exclusion ledger hash mismatch")
    open_gates = {
        "historical_generation_terms": (
            "Historical generation provider/model receipt and output-use terms are unresolved.",
            "Recover the generation receipt and applicable output-use terms, then bind both to this release.",
        ),
        "row_linked_rights_privacy": (
            "Rights, consent, and privacy evidence is not linked to this source row.",
            "Attach admissible row-linked rights/privacy evidence or keep this row local-only.",
        ),
        "content_decision_binding": (
            "The machine audit accepted this row, but no independent content decision is bound to it.",
            "Bind the required research content disposition to this row or keep it out of provider training.",
        ),
    }
    backlog = [
        {
            "row_id": row["row_id"],
            "owner": "unassigned",
            "gate": gate,
            "status": "open",
            "reason": reason,
            "next_action": action,
        }
        for row in lineage
        for gate, (reason, action) in open_gates.items()
    ]
    backlog.extend(
        {
            "row_id": row["candidate_row_sha256"],
            "owner": "unassigned",
            "gate": "heldout_exclusion",
            "status": "excluded",
            "reason": row["exclusion_reason"],
            "next_action": "Keep excluded unless a new group-disjoint split and evaluation review is approved.",
        }
        for row in exclusions
    )
    write_jsonl(output / "backlog.jsonl", backlog)
    write_json(
        output / "source_licenses.json",
        {
            "schema_version": 1,
            "scope": "522 frozen synthetic edge-case rows and source seed artifacts",
            "source_class": "edge_cases",
            "source_origin_assessment": "first-party synthetic edge branch supported by local taxonomy and seed review",
            "historical_generation_provider_and_model": "unverified; no artifact-bound receipt located",
            "generation_output_use_terms": "unverified",
            "source_rights_and_consent_basis": "not established by available row-linked evidence",
            "selected_provider_transfer_authorized": False,
            "evidence": [
                "docs/training/execution-2026-10-17/g03-edge-case-source-provenance-decision.md",
                "docs/training/execution-2026-10-17/g03-historical-run-provenance-followup.md",
                "docs/training/execution-2026-10-17/g03-edge-case-breakaway-frozen-2026-10-19.json",
            ],
        },
    )
    write_data_card(output)
    write_json(
        output / "reports/render_and_lineage.json",
        {
            "schema_version": 1,
            "status": "local_views_rendered_and_row_mapped; transfer_unqualified",
            "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
            "source_payload_hashes_verified": True,
            "candidate_to_canonical_content_match": True,
            "canonical_splits": {"train": 410, "dev": 72, "test": 40},
            "group_overlap_counts": freeze["package"]["split_group_intersections"],
            "heldout_exclusions": {"count": len(exclusions), "sha256": freeze["selection"]["excluded_sha256"]},
            "provider_views": {
                "wandb": "weighted_messages_jsonl_v1; user=0, assistant=1",
                "fireworks": {
                    "format": "documented role/content chat format; loss weights omitted and target masking unverified",
                    "reference": "https://fireworks.ai/blog/supervised-fine-tuning-tutorial",
                },
            },
            "privacy_screen": freeze["privacy_screen"],
            "remote_effects": {"uploads": 0, "training_jobs": 0, "deployments": 0},
            "source_render_report": "docs/training/execution-2026-10-17/g04-edge-render-report.json",
            "legacy_render_status": render_report["status"],
            "warnings": [
                "provider tokenizer/context length checks not run",
                "provider-specific loss semantics not verified",
                "historical generation provenance and terms unresolved",
            ],
        },
    )
    return len(exclusions)


def write_data_card(output: Path) -> None:
    card = """# G04 break-away dataset card

Release: `g04-breakaway-release-20261019-v1`

This private local package contains 522 frozen edge-case conversation rows,
split by source group into 410 train, 72 development, and 40 test rows. It is a
research-only candidate. The test split is reserved for evaluation and must not
enter training, rollouts, reward construction, or preference synthesis.

Each canonical message includes `loss_weight`: user turns use 0 and assistant
turns use 1. Provider views are deterministic renderings of the same canonical
rows. W&B retains the weighted message representation. The Fireworks view
follows its documented role/content chat shape and omits the canonical loss
mask; assistant-only target masking must be confirmed for the selected training
route before a job is created. This is format preparation, not provider
qualification or transfer approval.

The private lineage binds all rows to the pinned 190,139-row candidate, seed
artifact references, and group-disjoint assignments. The source branch is
classified as first-party synthetic, but the historical generation
provider/model receipt and applicable output-use terms are unresolved.
Rights/consent clearance is not established. A six-pattern regex screen found
no matches; this does not establish de-identification or absence of clinical
information. No clinician review or clinical safety validation is claimed.

**Use status:** local research only. Provider upload, training, deployment,
clinical use, and production use are not authorized by this package.
"""
    path = output / "DATA_CARD.md"
    path.write_text(card, encoding="utf-8")
    os.chmod(path, 0o600)


def build_manifest(output: Path, freeze: dict[str, Any], counts: dict[str, int], exclusions: int) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "release_id": "g04-breakaway-release-20261019-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "local_research_views_prepared_provider_transfer_blocked",
        "upload_gate": False,
        "training_job_created": False,
        "scope": "522-row G03 frozen local research package; not clinical or production ready",
        "source_hashes": {
            "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
            "freeze_manifest_sha256": sha256_file(FREEZE),
            "internal_manifest_sha256": sha256_file(SOURCE_PACKAGE / "manifest_internal.jsonl"),
            "builder_sha256": sha256_file(Path(__file__)),
        },
        "candidate_rows": 522,
        "excluded_heldout_rows": exclusions,
        "splits": counts,
        "split_group_intersections": freeze["package"]["split_group_intersections"],
        "python_version": platform.python_version(),
        "open_gates": [
            "historical generation provider/model and output-use terms",
            "row-linked rights/privacy and content decision evidence",
            "Fireworks loss-mask semantics",
            "provider tokenizer and context-length validation",
        ],
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ai/data/curated/recovery/gold-20261017/g04-breakaway-release-20261019-v1",
    )
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen release: {output}")
    freeze, render_report, internal, candidate_rows = load_pinned_source()
    references = make_reference_index(internal, candidate_rows)
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    for folder in ("canonical", "providers/wandb", "providers/fireworks", "reports"):
        (output / folder).mkdir(parents=True, mode=0o700)
    counts, lineage, _, _ = write_split_views(output, references, candidate_rows)
    exclusion_count = write_support_files(output, freeze, render_report, lineage)
    manifest = build_manifest(output, freeze, counts, exclusion_count)
    write_json(output / "manifest.json", manifest)
    descriptor = {
        "release_id": manifest["release_id"],
        "path": str(output.relative_to(ROOT)),
        "manifest_sha256": sha256_file(output / "manifest.json"),
        "status": manifest["status"],
        "upload_gate": False,
        "rows": manifest["candidate_rows"],
        "excluded_rows": exclusion_count,
        "provider_views": ["wandb", "fireworks"],
    }
    write_json(DOCS / "g04-breakaway-release-20261019-v1.json", descriptor)
    LOGGER.info("Built private release metadata: %s", json.dumps(descriptor, sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
