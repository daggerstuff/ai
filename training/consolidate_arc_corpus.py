#!/usr/bin/env python3
"""Consolidate accepted arc-corpus sessions into master gold (atomic + dedup).

Consumes ``ai/training/output/arc_corpus/arc_records.jsonl`` — one record per
completed arc (spec §9) — and appends every session of every ``accept``-verdict
arc to ``train_master_gold.jsonl`` as ChatML SFT records:

  1. converts each session's writer turns (``client``/``therapist`` + ledger) to
     strict alternating ChatML (client→user, therapist→assistant). The
     therapist's one-line JSON ledger stays IN the assistant content (spec §9:
     the think-block disposition is training signal, not QC scaffolding to
     strip) as ``<ledger JSON>\\n<spoken reply>``;
  2. merges consecutive client turns (writer self-interruption beats) into one
     user message and drops trailing client turns — every training record must
     end with an assistant turn or the model learns to end sessions unresponded;
  3. cliché-gates every record (``cliche_gate.reject_reason_for_record``);
  4. quadit-gates every record (deterministic ``ai.research.quadit`` dataset
     gate — banned phrases / director superlatives / PHI patterns);
  5. dedups against the master gold (append-only, first-write-wins, canonical
     ``compute_primary_hash`` hash space);
  6. appends passing records to the master gold with append + fsync, tagging
     provenance (``arc_id``, ``session_n``, writer/auditor models, verdict).

One gold record per session: an 8-35 turn arc session is one coherent
long-form exchange, unlike the short-track single-exchange records.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai.research.quadit.dataset_gate import (
    DEFAULT_AUDITORS,
    DatasetGateError,
    chatml_to_audit_item,
    gate_should_block,
)
from ai.research.quadit.personas import AuditorDescriptor, load_auditor_descriptor
from ai.research.quadit.review import run_quadit_audit
from pipelines.ingestion_deduplication import compute_primary_hash
from training.cliche_gate import reject_reason_for_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("consolidate_arc_corpus")

_AI_ROOT = Path(__file__).resolve().parents[1]
MASTER_GOLD = _AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl"
ARC_OUTPUT_DIR = _AI_ROOT / "training" / "output" / "arc_corpus"
DEFAULT_INPUTS: tuple[Path, ...] = (ARC_OUTPUT_DIR / "arc_records.jsonl",)
DEFAULT_REJECT = ARC_OUTPUT_DIR / "consolidation_rejections.jsonl"

FAMILY = "arc_corpus"
TASK_TYPE = "long_session_arc"


@dataclass
class _State:
    """Mutable per-run consolidation state (keeps helper arity low)."""

    gold_hashes: set[str]
    gold_path: Path
    reject_path: Path
    quadit_auditors: tuple[AuditorDescriptor, ...]
    dry_run: bool
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


def _session_messages(session: dict[str, Any]) -> list[dict[str, str]]:
    """Convert one session's writer turns to strict alternating ChatML.

    client→user, therapist→assistant with the one-line JSON ledger kept in the
    assistant content followed by the spoken reply. Consecutive client turns
    (writer self-interruption beats) merge into one user message; trailing
    client turns are dropped so every record ends with an assistant turn.
    Raises ``ValueError`` on structural violations (missing ledger, consecutive
    therapist turns, unknown roles) so the session lands in the reject log
    instead of silently corrupting the corpus.
    """
    messages: list[dict[str, str]] = []
    for turn in session.get("turns") or []:
        role = str(turn.get("role") or "")
        content = str(turn.get("content") or "").strip()
        if not content:
            raise ValueError(f"empty {role or 'unknown'} turn")
        if role == "client":
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + content
            else:
                messages.append({"role": "user", "content": content})
        elif role == "therapist":
            if messages and messages[-1]["role"] == "assistant":
                raise ValueError("consecutive therapist turns")
            ledger = turn.get("ledger")
            if not isinstance(ledger, dict) or not ledger:
                raise ValueError("therapist turn without ledger")
            messages.append({"role": "assistant", "content": f"{json.dumps(ledger, ensure_ascii=False)}\n{content}"})
        else:
            raise ValueError(f"unknown role {role!r}")
    while messages and messages[-1]["role"] == "user":
        messages.pop()
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("session must start with client and end with therapist")
    return messages


def _load_plan(plan_path: Any) -> dict[str, Any] | None:
    """Load the arc plan for seed metadata (domain, severity); None when absent."""
    if not plan_path:
        return None
    path = _AI_ROOT / str(plan_path)
    if not path.exists():
        logger.warning("plan not found: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("unparseable plan %s: %s", path, exc)
        return None


def _gold_record(
    arc: dict[str, Any],
    session: dict[str, Any],
    messages: list[dict[str, str]],
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    arc_id = str(arc.get("arc_id") or "")
    seed = (plan or {}).get("seed") or {}
    domain = str(seed.get("domain") or "")
    audit = arc.get("audit") or {}
    return {
        "messages": messages,
        "source": f"{FAMILY}_{arc_id}",
        "task_type": TASK_TYPE,
        "tier": "T1_GOLD",
        "diagnostic_tag": domain.replace("_", " "),
        "family": FAMILY,
        "difficulty": str(seed.get("severity") or ""),
        "provenance": {
            "type": FAMILY,
            "arc_id": arc_id,
            "session_n": session.get("n"),
            "plan_path": arc.get("plan_path"),
            "writer_model": arc.get("writer_model"),
            "auditor_model": arc.get("auditor_model"),
            "verdict": audit.get("verdict"),
            "spec_version": arc.get("spec_version"),
            "domain": domain,
        },
        "consolidated_at": _now_iso(),
    }


def _reject_entry(reason: str, record: dict[str, Any] | None, arc_id: str, session_n: Any) -> dict[str, Any]:
    return {
        "reason": reason,
        "hash": compute_primary_hash(record) if record is not None else None,
        "arc_id": arc_id,
        "session_n": session_n,
        "source": f"{FAMILY}_{arc_id}",
    }


def _reject(reason: str, record: dict[str, Any], arc_id: str, session_n: Any, state: _State) -> None:
    state.summary["rejected"] += 1
    if not state.dry_run:
        _append(state.reject_path, _reject_entry(reason, record, arc_id, session_n))


def _process_session(arc: dict[str, Any], session: dict[str, Any], plan: dict[str, Any] | None, state: _State) -> None:
    arc_id = str(arc.get("arc_id") or "")
    session_n = session.get("n")

    try:
        messages = _session_messages(session)
    except ValueError as exc:
        state.summary["rejected_invalid"] += 1
        if not state.dry_run:
            _append(state.reject_path, _reject_entry(f"invalid_session: {exc}", None, arc_id, session_n))
        return

    record = _gold_record(arc, session, messages, plan)

    reason = reject_reason_for_record(record, family=FAMILY)
    if reason is not None:
        _reject(reason, record, arc_id, session_n, state)
        return

    try:
        item = chatml_to_audit_item(record)
    except DatasetGateError as exc:
        _reject(f"quadit-malformed: {exc}", record, arc_id, session_n, state)
        return
    report = run_quadit_audit([item], auditors=state.quadit_auditors)
    if gate_should_block(report):
        _reject(f"quadit: {report.summary}", record, arc_id, session_n, state)
        return

    h = compute_primary_hash(record)
    if h in state.gold_hashes:
        state.summary["duplicates"] += 1
        return

    if not state.dry_run:
        _append(state.gold_path, record)
    state.gold_hashes.add(h)
    state.summary["emitted_gold"] += 1
    state.summary["by_arc"][arc_id] = state.summary["by_arc"].get(arc_id, 0) + 1


def consolidate(
    inputs: Sequence[Path],
    *,
    gold_path: Path,
    reject_path: Path,
    max_gold: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Consolidate accepted arc records into the master gold. Returns a summary dict."""
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    gold_hashes = _load_gold_hashes(gold_path, max_gold)
    logger.info("Loaded %d gold hashes%s", len(gold_hashes), " (capped)" if max_gold is not None else "")

    summary: dict[str, Any] = {
        "scanned_arcs": 0,
        "scanned_sessions": 0,
        "skipped_verdict": 0,
        "emitted_gold": 0,
        "duplicates": 0,
        "rejected": 0,
        "rejected_invalid": 0,
        "by_arc": {},
    }
    state = _State(
        gold_hashes=gold_hashes,
        gold_path=gold_path,
        reject_path=reject_path,
        quadit_auditors=tuple(load_auditor_descriptor(name) for name in DEFAULT_AUDITORS),
        dry_run=dry_run,
        summary=summary,
    )
    if dry_run:
        logger.info("DRY RUN — nothing will be written")

    for input_path in inputs:
        if not input_path.exists():
            logger.warning("staging input missing: %s", input_path)
            continue
        for arc in _iter_records(input_path):
            summary["scanned_arcs"] += 1
            if (arc.get("audit") or {}).get("verdict") != "accept":
                summary["skipped_verdict"] += 1
                continue
            plan = _load_plan(arc.get("plan_path"))
            for session in arc.get("sessions") or []:
                summary["scanned_sessions"] += 1
                _process_session(arc, session, plan, state)

    logger.info("Consolidation summary:\n%s", json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--gold", type=Path, default=MASTER_GOLD)
    parser.add_argument("--reject", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--max-gold", type=int, default=None, help="cap master hash build (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="gate + count everything, write nothing")
    args = parser.parse_args()

    consolidate(
        args.inputs,
        gold_path=args.gold,
        reject_path=args.reject,
        max_gold=args.max_gold,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
