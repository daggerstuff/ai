#!/usr/bin/env python3
"""Bind v4 ARC candidate rows to an immutable private archive snapshot.

This audit only reads the frozen candidate/release and ARC source archive. It
writes a protected archive snapshot, a hash-only row-reference sidecar, and a
metadata-only report. It never writes candidate or provider payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.consolidate_arc_corpus import _session_messages

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "ai/data/curated/recovery/gold-20261017"
CANDIDATE = ROOT / "ai/data/curated/sft_chatml/train_master_gold.jsonl"
ROWREFS = RECOVERY / "wandb-full-corpus-current-master-20261019-v4/rowrefs.jsonl"
ARCHIVE = ROOT / "ai/training/output/arc_corpus/arc_records.jsonl"
OUTPUT = RECOVERY / "private-v4-arc-lineage-20261019"
SNAPSHOT = OUTPUT / "arc_records.snapshot.jsonl"
SIDECAR = OUTPUT / "v4-arc-rowrefs.jsonl"
REPORT = OUTPUT / "report.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_messages_hash(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_private_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    path.chmod(0o600)


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def main() -> None:
    existed = OUTPUT.exists()
    if existed:
        if not SNAPSHOT.is_file() or not SIDECAR.is_file() or not REPORT.is_file():
            raise RuntimeError("existing audit output is incomplete; refusing to overwrite it")
        if (OUTPUT.stat().st_mode & 0o777) != 0o700:
            raise RuntimeError("existing private audit directory permissions changed")
    else:
        OUTPUT.mkdir(mode=0o700, parents=True)
    archive_before = sha256(ARCHIVE.read_bytes())
    if not existed:
        shutil.copyfile(ARCHIVE, SNAPSHOT)
        SNAPSHOT.chmod(0o600)
    snapshot_hash = sha256(SNAPSHOT.read_bytes())
    if snapshot_hash != archive_before:
        raise RuntimeError("ARC snapshot differs from source archive bytes")

    archive_records = read_jsonl(SNAPSHOT)
    archive_rows: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for arc in archive_records:
        for session in arc.get("sessions") or []:
            archive_rows[(str(arc.get("arc_id")), str(session.get("n")))].append((arc, session))

    release_rows = read_jsonl(ROWREFS)
    if len(release_rows) != 190244:
        raise RuntimeError(f"unexpected v4 rowref count: {len(release_rows)}")

    output_rows: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    arc_ids: Counter[str] = Counter()
    candidate_count = 0
    exact_count = 0
    missing = ambiguous = mismatched = 0
    candidate_messages_hashes: set[str] = set()

    with CANDIDATE.open("rb") as candidate_stream:
        for row_number, raw in enumerate(candidate_stream, start=1):
            candidate_record = json.loads(raw)
            provenance = candidate_record.get("provenance") or {}
            if provenance.get("type") != "arc_corpus":
                continue
            candidate_count += 1
            rowref = release_rows[row_number - 1]
            row_hash = sha256(raw.rstrip(b"\n"))
            if rowref.get("candidate_row_number_1based") != row_number or rowref.get("candidate_row_sha256") != row_hash:
                raise RuntimeError(f"v4 candidate rowref mismatch at row {row_number}")

            arc_id = str(provenance.get("arc_id"))
            session_n = str(provenance.get("session_n"))
            matches = archive_rows.get((arc_id, session_n), [])
            if not matches:
                missing += 1
                raise RuntimeError(f"missing ARC source key at candidate row {row_number}")
            if len(matches) != 1:
                ambiguous += 1
                raise RuntimeError(f"ambiguous ARC source key at candidate row {row_number}")

            arc, session = matches[0]
            rendered_messages = _session_messages(session)
            if candidate_record.get("messages") != rendered_messages:
                mismatched += 1
                raise RuntimeError(f"canonical message projection mismatch at candidate row {row_number}")
            if candidate_record.get("source") != f"arc_corpus_{arc_id}":
                raise RuntimeError(f"source label mismatch at candidate row {row_number}")

            audit = arc.get("audit") or {}
            verdict = str(audit.get("verdict") or "")
            verdicts[verdict] += 1
            arc_ids[arc_id] += 1
            messages_hash = canonical_messages_hash(rendered_messages)
            candidate_messages_hashes.add(messages_hash)
            exact_count += 1
            output_rows.append(
                {
                    "release_id": "wandb-full-corpus-current-master-20261019-v4",
                    "candidate_row_number_1based": row_number,
                    "candidate_row_sha256": row_hash,
                    "messages_sha256": messages_hash,
                    "provisional_split": rowref["provisional_split"],
                    "source_label": candidate_record["source"],
                    "arc_id": arc_id,
                    "session_n": session_n,
                    "archive_sha256": snapshot_hash,
                    "archive_audit_verdict": verdict,
                    "writer_model": arc.get("writer_model"),
                    "auditor_model": arc.get("auditor_model"),
                    "plan_path": arc.get("plan_path"),
                    "spec_version": arc.get("spec_version"),
                    "binding": "exact_canonical_session_messages_and_arc_session_key",
                }
            )

    archive_after = sha256(ARCHIVE.read_bytes())
    if archive_after != archive_before:
        raise RuntimeError("ARC source archive changed while the audit was running")
    if candidate_count != 1085 or exact_count != 1085 or missing or ambiguous or mismatched:
        raise RuntimeError("unexpected ARC binding totals")
    if verdicts != Counter({"accept": 1085}):
        raise RuntimeError(f"unexpected machine audit verdict counts: {verdicts}")
    if any(not row["writer_model"] or not row["auditor_model"] or not row["plan_path"] for row in output_rows):
        raise RuntimeError("source archive metadata missing for one or more ARC rows")

    report = {
        "schema_version": 1,
        "release_id": "wandb-full-corpus-current-master-20261019-v4",
        "scope": "ARC provenance/content binding only; no rights, privacy, semantic quality, or clinical clearance claim",
        "candidate_path": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": sha256(CANDIDATE.read_bytes()),
        "candidate_rowrefs_path": str(ROWREFS.relative_to(ROOT)),
        "candidate_rowrefs_sha256": sha256(ROWREFS.read_bytes()),
        "arc_source_archive_path": str(ARCHIVE.relative_to(ROOT)),
        "arc_source_archive_sha256": archive_before,
        "private_archive_snapshot_path": str(SNAPSHOT.relative_to(ROOT)),
        "private_archive_snapshot_sha256": snapshot_hash,
        "private_rowref_sidecar_path": str(SIDECAR.relative_to(ROOT)),
        "private_rowref_sidecar_sha256": sha256(jsonl_bytes(output_rows)),
        "candidate_arc_rows": candidate_count,
        "unique_arc_session_keys": exact_count,
        "exact_canonical_message_matches": exact_count,
        "missing_or_ambiguous_keys": missing + ambiguous,
        "message_projection_mismatches": mismatched,
        "machine_audit_verdict_counts": dict(verdicts),
        "unique_arc_ids": len(arc_ids),
        "split_counts": dict(Counter(row["provisional_split"] for row in output_rows)),
        "candidate_row_sha256_unique_count": len({row["candidate_row_sha256"] for row in output_rows}),
        "message_sha256_unique_count": len(candidate_messages_hashes),
        "source_label_checks": "all ARC candidate source labels equal arc_corpus_<arc_id>",
        "source_metadata_checks": "all bound archive entries have writer_model, auditor_model, and plan_path",
        "privacy_handling": "source archive snapshot and rowref sidecar mode 0600 under mode 0700 directory; no message text written to sidecar/report",
        "limitations": [
            "Machine audit verdict and exact content binding do not establish clinician review or clinical quality.",
            "This binding does not resolve generation-provider identity/terms, source rights, or privacy clearance.",
            "Separate lane-2 human decisions are not transferred because they lack candidate content hashes.",
        ],
    }
    expected_report = (json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if existed:
        if SIDECAR.read_bytes() != jsonl_bytes(output_rows):
            raise RuntimeError("existing private rowrefs differ from current reproducible binding")
        if REPORT.read_bytes() != expected_report:
            raise RuntimeError("existing private report differs from current reproducible binding")
        if any(path.stat().st_mode & 0o777 != 0o600 for path in (SNAPSHOT, SIDECAR, REPORT)):
            raise RuntimeError("private audit file permissions changed")
        print("Existing protected audit artifacts reverified without modification.")
    else:
        write_private_jsonl(SIDECAR, output_rows)
        REPORT.write_bytes(expected_report)
        REPORT.chmod(0o600)
    print(json.dumps({k: report[k] for k in (
        "candidate_arc_rows", "exact_canonical_message_matches", "missing_or_ambiguous_keys",
        "message_projection_mismatches", "machine_audit_verdict_counts", "unique_arc_ids",
        "split_counts", "private_archive_snapshot_sha256", "private_rowref_sidecar_sha256",
    )}, indent=2))


if __name__ == "__main__":
    main()
