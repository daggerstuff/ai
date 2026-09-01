#!/usr/bin/env python3
"""Consolidate staging outputs into stage manifests AND master gold (atomic + dedup).

Step 9 of the clinical-NF pipeline. Consumes the three step-6/7/8 staging files:

  * ``edge_and_nightmare_generated.jsonl``  (step 7 — Stage 3 edge + nightmare fuel)
  * ``clinical_backed_staging.jsonl``       (step 8 — Stages 1/2 hybrid sourcing)
  * ``staged_edge_reconciliation.jsonl``    (step 6 — streamed edge scenarios + DPO pairs)

For every gate-passing record it:

  1. recomputes the *canonical* primary hash (``ingestion_deduplication.
     compute_primary_hash`` = sha256(lowercase(concat(role+content)))) for BOTH the
     master gold and incoming records — fixing the ``content_hashes`` /
     ``compute_primary_hash`` hash-space mismatch left by steps 6/7/8;
  2. routes each record to its highest-priority stage (safety > voice > edge >
     therapeutic > foundation), mapping the *top-level* ``stage`` field the staging
     records carry (``get_stage_priority`` in ``ingestion_deduplication`` only reads
     ``metadata.stage`` and omits ``stage5_safety``);
  3. cliché-gates every record (``cliche_gate.reject_reason_for_record``);
  4. dedups against the master gold + existing manifests (append-only, first-write-
     wins, no duplicate hashes);
  5. appends to ``MASTER_STAGE_N.jsonl`` **and** ``train_master_gold.jsonl`` with
     append + fsync, tagging provenance (``stage``, ``consolidated_at``).

DPO preference pairs (``{prompt, chosen, rejected}``) route to Stage 5 and are
written **only** to ``MASTER_STAGE_5.jsonl`` — they are preference data, not ChatML
SFT, so they are never appended to the flat ChatML gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipelines.ingestion_deduplication import compute_primary_hash
from training.cliche_gate import reject_reason_for_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("consolidate_edge_nightmare")

_AI_ROOT = Path(__file__).resolve().parents[1]
MASTER_GOLD = _AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl"
FINAL_DIR = _AI_ROOT / "training_data_consolidated" / "final"
STAGING_DIR = _AI_ROOT / "training" / "output" / "nightmare_fuel"
DEFAULT_REJECT = STAGING_DIR / "consolidation_rejections.jsonl"

DEFAULT_INPUTS: tuple[Path, ...] = (
    STAGING_DIR / "edge_and_nightmare_generated.jsonl",
    STAGING_DIR / "clinical_backed_staging.jsonl",
    STAGING_DIR / "staged_edge_reconciliation.jsonl",
)

# 10 authoritative edge families (lowercased) — Stage 3 routing.
EDGE_FAMILIES: frozenset[str] = frozenset(
    {
        "ambiguous crisis language",
        "delusion or paranoia",
        "coercion or abuse",
        "substance use",
        "medical uncertainty",
        "minor or dependent person",
        "therapeutic rupture",
        "cultural or identity conflict",
        "boundary testing",
        "multi-problem complexity",
    }
)

# Stage priority with stage5_safety included (ingestion_deduplication.STAGE_PRIORITY
# omits it — step-1 phasing spec flagged this gap).
STAGE_PRIORITY: dict[str, int] = {
    "stage5_safety": 6,
    "stage4_voice_persona": 5,
    "stage3_edge_stress_test": 4,
    "stage2_therapeutic_expertise": 3,
    "stage1_foundation": 2,
    "supplementary": 1,
}

MANIFEST_FILENAME: dict[str, str] = {
    "stage1_foundation": "MASTER_STAGE_1.jsonl",
    "stage2_therapeutic_expertise": "MASTER_STAGE_2.jsonl",
    "stage3_edge_stress_test": "MASTER_STAGE_3.jsonl",
    "stage4_voice_persona": "MASTER_STAGE_4.jsonl",
    "stage5_safety": "MASTER_STAGE_5.jsonl",
}

_MANIFEST_STAGE_BY_FILE = {filename: stage for stage, filename in MANIFEST_FILENAME.items()}

# "stageN" prefix -> canonical stage name (handles both full and short stage values).
_STAGE_BY_PREFIX: dict[str, str] = {
    "stage5": "stage5_safety",
    "stage4": "stage4_voice_persona",
    "stage3": "stage3_edge_stress_test",
    "stage2": "stage2_therapeutic_expertise",
    "stage1": "stage1_foundation",
}

EDGE_STAGE = "stage3_edge_stress_test"
FOUNDATION_STAGE = "stage1_foundation"


@dataclass
class _State:
    """Mutable per-run consolidation state (keeps helper arity low)."""

    gold_hashes: set[str]
    manifest_hashes: dict[str, int]
    manifest_dir: Path
    gold_path: Path
    reject_path: Path
    summary: dict[str, Any] = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line + fsync (durable append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _is_dpo(record: dict[str, Any]) -> bool:
    """True for preference-pair records ({prompt, chosen, rejected} or reconciliation DPO)."""
    if str(record.get("asset_kind") or "").strip().lower() == "dpo":
        return True
    return all(k in record for k in ("prompt", "chosen", "rejected"))


def _stage_of(record: dict[str, Any]) -> str:
    """Route a record to its highest-priority stage (safety > voice > edge > therapeutic > foundation).

    Staging records carry a top-level ``stage`` field (steps 7/8); reconciliation
    records only carry ``family`` / ``asset_kind``. Priority mirrors
    ``stage_organizer.classify_record``.
    """
    if _is_dpo(record):
        return "stage5_safety"

    stage = str(record.get("stage") or "").strip().lower()
    family = str(record.get("family") or "").strip().lower()
    source = str(record.get("source") or "").strip().lower()

    if family == "nightmare fuel" or source.startswith("nightmare_fuel"):
        return EDGE_STAGE
    if source.startswith("clinical_edge_case_") or family in EDGE_FAMILIES:
        return EDGE_STAGE

    stage_name = _STAGE_BY_PREFIX.get(stage[:6])
    if stage_name is not None:
        return stage_name
    if family and family != "unmapped":
        return EDGE_STAGE
    return FOUNDATION_STAGE


def _dpo_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct a canonical {prompt, chosen, rejected} DPO record.

    Reconciliation DPO records carry ``messages`` (prompt->chosen as user/assistant)
    plus a top-level ``rejected``; raw DPO pairs already carry all three keys.
    """
    if all(k in record for k in ("prompt", "chosen")):
        prompt = str(record.get("prompt") or "").strip()
        chosen = str(record.get("chosen") or "").strip()
        rejected = str(record.get("rejected") or "").strip()
    else:
        messages = record.get("messages") or []
        prompt = next(
            (str(m.get("content") or "") for m in messages if m.get("role") == "user"), ""
        ).strip()
        chosen = next(
            (str(m.get("content") or "") for m in messages if m.get("role") == "assistant"), ""
        ).strip()
        rejected = str(record.get("rejected") or "").strip()
    if not prompt or not chosen or not rejected:
        return None
    payload: dict[str, Any] = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = metadata
    return payload


def _dpo_hash(payload: dict[str, Any]) -> str:
    text = f"{payload.get('prompt', '')}{payload.get('chosen', '')}{payload.get('rejected', '')}"
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def _load_gold_hashes(gold_path: Path, max_gold: int | None) -> set[str]:
    """Canonical primary hashes of every ChatML record already in the master gold."""
    hashes: set[str] = set()
    if not gold_path.exists():
        return hashes
    seen = 0
    with gold_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
                continue
            hashes.add(compute_primary_hash(record))
            seen += 1
            if max_gold is not None and seen >= max_gold:
                break
    return hashes


def _load_manifest_hashes(manifest_dir: Path) -> dict[str, int]:
    """Existing manifest record hashes -> priority, keyed by the filename's stage."""
    hashes: dict[str, int] = {}
    if not manifest_dir.exists():
        return hashes
    for path in sorted(manifest_dir.glob("MASTER_STAGE_*.jsonl")):
        priority = STAGE_PRIORITY.get(_MANIFEST_STAGE_BY_FILE.get(path.name, ""), 1)
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                if _is_dpo(record):
                    payload = _dpo_payload(record)
                    if payload is not None:
                        h = _dpo_hash(payload)
                        hashes[h] = max(hashes.get(h, 0), priority)
                elif isinstance(record.get("messages"), list):
                    h = compute_primary_hash(record)
                    hashes[h] = max(hashes.get(h, 0), priority)
    return hashes


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _process_dpo(record: dict[str, Any], state: _State) -> None:
    stage = "stage5_safety"
    payload = _dpo_payload(record)
    if payload is None:
        state.summary["skipped_no_messages"] += 1
        return

    gate_record = {
        "messages": [
            {"role": "user", "content": payload.get("prompt", "")},
            {"role": "assistant", "content": payload.get("chosen", "")},
        ]
    }
    reason = reject_reason_for_record(gate_record, family="stage5_dpo")
    if reason is not None:
        state.summary["rejected"] += 1
        _append(
            state.reject_path,
            {"reason": reason, "hash": _dpo_hash(payload),
             "source": record.get("source"), "stage": stage},
        )
        return

    h = _dpo_hash(payload)
    if h in state.manifest_hashes:
        state.summary["duplicates"] += 1
        return

    payload["stage"] = stage
    payload["consolidated_at"] = _now_iso()
    _append(state.manifest_dir / MANIFEST_FILENAME[stage], payload)
    state.manifest_hashes[h] = STAGE_PRIORITY[stage]
    state.summary["emitted_manifest"] += 1
    state.summary["by_stage"][stage] = state.summary["by_stage"].get(stage, 0) + 1


def _process_chatml(record: dict[str, Any], stage: str, state: _State) -> None:
    if not isinstance(record.get("messages"), list):
        state.summary["skipped_no_messages"] += 1
        return

    reason = reject_reason_for_record(record, family=str(record.get("family") or stage))
    if reason is not None:
        state.summary["rejected"] += 1
        _append(
            state.reject_path,
            {"reason": reason, "hash": compute_primary_hash(record),
             "source": record.get("source"), "stage": stage},
        )
        return

    h = compute_primary_hash(record)
    if h in state.gold_hashes or h in state.manifest_hashes:
        state.summary["duplicates"] += 1
        return

    record["stage"] = stage
    record["consolidated_at"] = _now_iso()
    _append(state.manifest_dir / MANIFEST_FILENAME[stage], record)
    state.manifest_hashes[h] = STAGE_PRIORITY[stage]
    state.summary["emitted_manifest"] += 1

    _append(state.gold_path, record)
    state.gold_hashes.add(h)
    state.summary["emitted_gold"] += 1
    state.summary["by_stage"][stage] = state.summary["by_stage"].get(stage, 0) + 1


def consolidate(
    inputs: Sequence[Path],
    *,
    gold_path: Path,
    manifest_dir: Path,
    reject_path: Path,
    max_gold: int | None = None,
) -> dict[str, Any]:
    """Consolidate staging files into stage manifests + master gold. Returns a summary dict."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    gold_hashes = _load_gold_hashes(gold_path, max_gold)
    manifest_hashes = _load_manifest_hashes(manifest_dir)
    logger.info("Loaded %d gold hashes + %d manifest hashes", len(gold_hashes), len(manifest_hashes))

    summary: dict[str, Any] = {
        "scanned": 0,
        "emitted_gold": 0,
        "emitted_manifest": 0,
        "duplicates": 0,
        "rejected": 0,
        "skipped_no_messages": 0,
        "by_stage": {},
    }
    state = _State(
        gold_hashes=gold_hashes,
        manifest_hashes=manifest_hashes,
        manifest_dir=manifest_dir,
        gold_path=gold_path,
        reject_path=reject_path,
        summary=summary,
    )

    for input_path in inputs:
        if not input_path.exists():
            logger.warning("staging input missing: %s", input_path)
            continue
        for record in _iter_records(input_path):
            state.summary["scanned"] += 1
            if _is_dpo(record):
                _process_dpo(record, state)
            else:
                _process_chatml(record, _stage_of(record), state)

    logger.info("Consolidation summary:\n%s", json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--gold", type=Path, default=MASTER_GOLD)
    parser.add_argument("--manifest-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--reject", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--max-gold", type=int, default=None, help="cap master hash build (smoke test)")
    args = parser.parse_args()

    consolidate(
        args.inputs,
        gold_path=args.gold,
        manifest_dir=args.manifest_dir,
        reject_path=args.reject,
        max_gold=args.max_gold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
