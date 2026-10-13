#!/usr/bin/env python3
"""Emit the Fireworks pilot payload from the arc-corpus gold slice.

Implements ``docs/training/fireworks-export-plan.md`` (2026-10-10) — local
only, fail-closed:

  1. re-checks the frozen snapshot identity (plan section 6) against the
     current master gold and refuses to run on any mismatch;
  2. groups every accepted-arc row per the frozen D4 construction
     (``source_family`` = the plan's ``seed.source``; ``source_id`` =
     ``seed.scenario_id`` falling back to the arc's own id) and assigns
     train/val/test with the splitter's own imported ``stratified_split``;
  3. emits provider rows — OpenAI chat JSONL, client -> ``user`` weight 0,
     therapist -> ``assistant`` weight 1, the 9-field ledger inline in each
     assistant turn (decision D2) — plus the internal provenance manifest;
     every other gold field is stripped from the provider payload;
  4. validates the emission: content fidelity, ledger contract, PII screen,
     duplicates, split census (plan section 5.3), and records the emitted
     file identities in ``identity.json``;
  5. stops. Nothing is uploaded, provisioned, or spent — the upload gate
     remains the owner's (readiness section 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.dataset_splitter_stratified import get_convo_hash, integrity_gates, stratified_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("export_fireworks_payload")

_AI_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = _AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl"
DEFAULT_OUT_DIR = _AI_ROOT / "training" / "output" / "arc_corpus" / "export" / "fireworks_pilot_v1"
SPLIT_NAMES = ("train", "val", "test")

# Frozen snapshot identity (export plan section 6, computed 2026-10-10).
EXPECTED_GOLD_SHA256 = "d357b8b573740f418c657c5d3e3421542c31d93b36ccc1d65e9a004a5f386bef"
EXPECTED_SLICE_SHA256 = "0babff8580cf9b2c176187ece019592280e337c03b222f1ab9bca631bf2283a3"
EXPECTED_MANIFEST_SHA256 = "d6c4147d3287f84b7407dc2762aed87c6451af8b2f1ad65a372690c07a3db272"

# Frozen census (export plan sections 3-5.3, computed 2026-10-10).
EXPECTED_SLICE_ROWS = 576
EXPECTED_SPLIT_ROWS: dict[str, int] = {"train": 418, "val": 80, "test": 78}
EXPECTED_SPLIT_GROUPS: dict[str, int] = {"train": 163, "val": 31, "test": 30}
EXPECTED_TOKENS = 4_325_355  # message-content chars / 4
EXPECTED_RISK_GRAMMAR_FLAGS = 28  # risk value not token-leading (plan section 4.1)
EXPECTED_EMPTY_VALUE_FLAGS = 35  # ledger value empty (all `hx: ""`, plan section 4.1)
EXPECTED_ARC_FALLBACK_ROWS = 351
EXPECTED_SEED_GROUPS = 77
EXPECTED_SHARED_SEED_SCENARIOS = 11
EXPECTED_DIFFICULTY_SPLIT: dict[str, dict[str, int]] = {
    "severe": {"train": 160, "val": 25, "test": 32},
    "moderate": {"train": 95, "val": 16, "test": 16},
    "high": {"train": 88, "val": 29, "test": 8},
    "adversarial": {"train": 65, "val": 10, "test": 22},
    "catastrophic": {"train": 10, "val": 0, "test": 0},
}

LEDGER_KEYS = {"dx", "def", "soma", "risk", "hx", "onset", "track", "tx", "tl"}
RISK_TOKEN_RE = re.compile(r"^\s*(none|passive|active)\b", re.IGNORECASE)

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "card": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
    "url": re.compile(r"https?://\S+"),
    "ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}


@dataclass(frozen=True)
class _LedgerCheck:
    """Ledger contract state for one assistant turn (``False`` = failed)."""

    parses: bool
    keyset: bool
    values: bool
    risk: bool


@dataclass
class _EmissionFacts:
    """Everything ``identity.json`` records beyond the emitted-file hashes."""

    gold_sha: str
    slice_sha: str
    manifest_sha: str
    census: dict[str, Any]
    tokens: int
    assistant_turns: int
    risk_flags: dict[str, int] = field(default_factory=dict)
    empty_value_flags: dict[str, int] = field(default_factory=dict)
    deviation_issues: list[str] = field(default_factory=list)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gold_slice(gold_path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[bytes], str]:
    """Stream gold once: (line_no, record) per arc row, raw arc lines, gold sha256.

    ``line_no`` is the 1-based line number within the whole gold file.
    """
    slice_rows: list[tuple[int, dict[str, Any]]] = []
    raw_lines: list[bytes] = []
    gold_sha = hashlib.sha256()
    with gold_path.open("rb") as f:
        for line_no, raw in enumerate(f, start=1):
            gold_sha.update(raw)
            record = json.loads(raw.decode("utf-8"))
            if (record.get("provenance") or {}).get("arc_id"):
                slice_rows.append((line_no, record))
                raw_lines.append(raw)
    return slice_rows, raw_lines, gold_sha.hexdigest()


def _check_identity(raw_lines: list[bytes], gold_sha: str) -> tuple[str, str, list[str]]:
    """Return (slice_sha, manifest_sha, issues) for the plan section 6 identity."""
    issues: list[str] = []
    if gold_sha != EXPECTED_GOLD_SHA256:
        issues.append(f"gold sha256 mismatch: expected {EXPECTED_GOLD_SHA256}, got {gold_sha}")
    if len(raw_lines) != EXPECTED_SLICE_ROWS:
        issues.append(f"slice row count mismatch: expected {EXPECTED_SLICE_ROWS}, got {len(raw_lines)}")
    slice_sha = _sha256(b"".join(raw_lines))
    if slice_sha != EXPECTED_SLICE_SHA256:
        issues.append(f"slice sha256 mismatch: expected {EXPECTED_SLICE_SHA256}, got {slice_sha}")
    manifest_sha = _sha256("\n".join(_sha256(line) for line in raw_lines).encode())
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        issues.append(f"manifest sha256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, got {manifest_sha}")
    return slice_sha, manifest_sha, issues


def _group_parts(record: dict[str, Any], cache: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Frozen D4 construction: (source_family, source_id) for one row."""
    prov = record["provenance"]
    arc_id = prov["arc_id"]
    if arc_id not in cache:
        plan_path = _AI_ROOT / prov["plan_path"]
        seed = json.loads(plan_path.read_text(encoding="utf-8")).get("seed") or {}
        source = str(seed.get("source") or "?")
        scenario_id = seed.get("scenario_id")
        cache[arc_id] = (source, str(scenario_id) if scenario_id is not None else arc_id)
    return cache[arc_id]


def _assign_splits(
    slice_rows: list[tuple[int, dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, str], dict[str, tuple[str, str]]]:
    """Group per D4 and split via the splitter's own deterministic rule."""
    cache: dict[str, tuple[str, str]] = {}
    enriched: list[dict[str, Any]] = []
    for line_no, record in slice_rows:
        source_family, source_id = _group_parts(record, cache)
        enriched.append(
            {
                **record,
                "metadata": {"source_family": source_family, "source_id": source_id},
                "_gold_line_no": line_no,
            }
        )
    splits = stratified_split(enriched)
    line_split: dict[int, str] = {}
    for split_name, records in splits.items():
        for record in records:
            line_split[record["_gold_line_no"]] = split_name
    return splits, line_split, cache


def _census_for_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows, groups, arc-fallback rows, difficulty counts, seed-group arc map for one split."""
    group_keys: set[str] = set()
    difficulty: Counter[str] = Counter()
    seed_arcs: dict[str, set[str]] = {}
    fallback_rows = 0
    for record in records:
        metadata = record["metadata"]
        group_keys.add(f"{metadata['source_family']}::{metadata['source_id']}")
        difficulty[str(record.get("difficulty") or "?")] += 1
        if metadata["source_id"] == record["provenance"]["arc_id"]:
            fallback_rows += 1
        else:
            key = f"{metadata['source_family']}::{metadata['source_id']}"
            seed_arcs.setdefault(key, set()).add(record["provenance"]["arc_id"])
    return {
        "rows": len(records),
        "groups": len(group_keys),
        "fallback_rows": fallback_rows,
        "difficulty": dict(difficulty),
        "seed_arcs": seed_arcs,
    }


def _census_issues(
    census: dict[str, Any],
    per_split: dict[str, dict[str, Any]],
    seed_group_arcs: dict[str, set[str]],
) -> list[str]:
    """Compare the computed census against the frozen plan section 5.3."""
    issues: list[str] = []
    for split_name in SPLIT_NAMES:
        if census["rows"][split_name] != EXPECTED_SPLIT_ROWS[split_name]:
            issues.append(
                f"split row mismatch: {split_name}: expected "
                f"{EXPECTED_SPLIT_ROWS[split_name]}, got {census['rows'][split_name]}"
            )
        if census["groups"][split_name] != EXPECTED_SPLIT_GROUPS[split_name]:
            issues.append(
                f"split group mismatch: {split_name}: expected "
                f"{EXPECTED_SPLIT_GROUPS[split_name]}, got {census['groups'][split_name]}"
            )
    for difficulty, expected in EXPECTED_DIFFICULTY_SPLIT.items():
        for split_name in SPLIT_NAMES:
            actual = per_split[split_name]["difficulty"].get(difficulty, 0)
            if actual != expected[split_name]:
                issues.append(
                    f"difficulty census mismatch: {difficulty}/{split_name}: "
                    f"expected {expected[split_name]}, got {actual}"
                )
    total_fallback = sum(census["fallback_rows"].values())
    if total_fallback != EXPECTED_ARC_FALLBACK_ROWS:
        issues.append(f"arc-fallback row mismatch: expected {EXPECTED_ARC_FALLBACK_ROWS}, got {total_fallback}")
    if len(seed_group_arcs) != EXPECTED_SEED_GROUPS:
        issues.append(f"seed group mismatch: expected {EXPECTED_SEED_GROUPS}, got {len(seed_group_arcs)}")
    shared = sum(1 for arcs in seed_group_arcs.values() if len(arcs) > 1)
    if shared != EXPECTED_SHARED_SEED_SCENARIOS:
        issues.append(f"shared seed scenario mismatch: expected {EXPECTED_SHARED_SEED_SCENARIOS}, got {shared}")
    return issues


def _split_census(splits: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[str]]:
    """Count rows/groups/fallbacks per split; return (census, issues) vs the frozen plan."""
    census: dict[str, Any] = {"rows": {}, "groups": {}, "fallback_rows": {}}
    per_split: dict[str, dict[str, Any]] = {}
    seed_group_arcs: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        info = _census_for_split(splits[split_name])
        per_split[split_name] = info
        for key in ("rows", "groups", "fallback_rows"):
            census[key][split_name] = info[key]
        for key, arcs in info["seed_arcs"].items():
            seed_group_arcs.setdefault(key, set()).update(arcs)
    return census, _census_issues(census, per_split, seed_group_arcs)


def _provider_row(record: dict[str, Any]) -> dict[str, Any]:
    """Provider payload row: messages with roles, content, and weights only."""
    messages: list[dict[str, Any]] = []
    for msg in record["messages"]:
        role = msg["role"]
        if role not in ("user", "assistant"):
            raise ValueError(f"unexpected role {role!r} in {record['provenance']['arc_id']}")
        messages.append({"role": role, "content": msg["content"], "weight": 0 if role == "user" else 1})
    return {"messages": messages}


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write newline-terminated JSONL atomically (tmp + fsync + rename)."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _scan_pii(text: str) -> Counter[str]:
    hits: Counter[str] = Counter()
    for name, pattern in PII_PATTERNS.items():
        for _ in pattern.findall(text):
            hits[name] += 1
    return hits


def _ledger_check(content: str) -> _LedgerCheck:
    """Classify one assistant turn's ledger against the spec section 4 contract."""
    newline = content.find("\n")
    if newline <= 0:
        return _LedgerCheck(parses=False, keyset=False, values=False, risk=False)
    try:
        ledger = json.loads(content[:newline])
    except json.JSONDecodeError:
        return _LedgerCheck(parses=False, keyset=False, values=False, risk=False)
    keyset = set(ledger.keys()) == LEDGER_KEYS
    values = keyset and all(isinstance(value, str) and value.strip() for value in ledger.values())
    risk = isinstance(ledger.get("risk"), str) and bool(RISK_TOKEN_RE.match(ledger["risk"]))
    return _LedgerCheck(parses=True, keyset=keyset, values=values, risk=risk)


def _ledger_precheck(
    slice_rows: list[tuple[int, dict[str, Any]]],
) -> tuple[int, Counter[str], Counter[str], int]:
    """Return (hard violations, empty-value flags, risk-grammar flags, assistant turns).

    Hard violations are parse or key-set failures (fail closed). The two flag
    counts are documented deviations (plan section 4.1) and must not drift.
    """
    violations = 0
    empty_flags: Counter[str] = Counter()
    risk_flags: Counter[str] = Counter()
    assistant_turns = 0
    for _, record in slice_rows:
        prov = record["provenance"]
        session = f"{prov['arc_id']} s{prov['session_n']}"
        for msg in record["messages"]:
            if msg["role"] != "assistant":
                continue
            assistant_turns += 1
            check = _ledger_check(msg["content"])
            if not check.parses or not check.keyset:
                violations += 1
            else:
                if not check.values:
                    empty_flags[session] += 1
                if not check.risk:
                    risk_flags[session] += 1
    return violations, empty_flags, risk_flags, assistant_turns


def _emit_payload(
    out_dir: Path,
    slice_rows: list[tuple[int, dict[str, Any]]],
    raw_lines: list[bytes],
    line_split: dict[int, str],
    group_cache: dict[str, tuple[str, str]],
) -> None:
    """Write train/val/test provider files plus the internal manifest."""
    payload_lines: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    manifest_lines: list[str] = []
    for idx, (line_no, record) in enumerate(slice_rows):
        split_name = line_split[line_no]
        payload_lines[split_name].append(json.dumps(_provider_row(record), ensure_ascii=False))
        prov = record["provenance"]
        source_family, source_id = _group_parts(record, group_cache)
        manifest_lines.append(
            json.dumps(
                {
                    "gold_line_no": line_no,
                    "arc_id": prov["arc_id"],
                    "session_n": prov["session_n"],
                    "split": split_name,
                    "source_family": source_family,
                    "source_id": source_id,
                    "line_sha256": _sha256(raw_lines[idx]),
                    "difficulty": record.get("difficulty"),
                    "domain": prov.get("domain"),
                    "writer_model": prov.get("writer_model"),
                    "auditor_model": prov.get("auditor_model"),
                    "verdict": prov.get("verdict"),
                    "consolidated_at": prov.get("consolidated_at"),
                },
                ensure_ascii=False,
            )
        )
    for split_name in SPLIT_NAMES:
        _write_lines(out_dir / f"{split_name}.jsonl", payload_lines[split_name])
    _write_lines(out_dir / "manifest_internal.jsonl", manifest_lines)


def _row_scan(
    row: dict[str, Any], split_name: str, convo_seen: set[str], pii_hits: Counter[str]
) -> tuple[list[str], int, int, int, int]:
    """Scan one emitted row; returns (issues, content chars, assistant turns, empty, risk flags)."""
    issues: list[str] = []
    chars = 0
    assistant_turns = 0
    empty_flags = 0
    risk_flags = 0
    convo_hash = get_convo_hash(row)
    if convo_hash in convo_seen:
        issues.append(f"duplicate content hash in emitted payload: {convo_hash[:12]}")
    convo_seen.add(convo_hash)
    for msg in row["messages"]:
        role = msg["role"]
        content = msg["content"]
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            issues.append(f"invalid role or empty content in {split_name}")
            continue
        if msg.get("weight") != (0 if role == "user" else 1):
            issues.append(f"weight mask mismatch for role {role} in {split_name}")
        chars += len(content)
        pii_hits.update(_scan_pii(content))
        if role == "assistant":
            assistant_turns += 1
            check = _ledger_check(content)
            if not check.parses or not check.keyset:
                issues.append(f"assistant turn fails the ledger parse/key-set contract in {split_name}")
            elif not check.values:
                empty_flags += 1
            if check.parses and not check.risk:
                risk_flags += 1
    return issues, chars, assistant_turns, empty_flags, risk_flags


def _validate_structure(
    out_dir: Path, census: dict[str, Any]
) -> tuple[list[str], dict[str, list[dict[str, Any]]], int, int, int, int]:
    """Scan emitted rows: roles, weights, ledger contract, PII, duplicates.

    Returns (issues, emitted rows by split, total content chars, assistant
    turns, empty-value flags, risk-grammar flags).
    """
    issues: list[str] = []
    emitted: dict[str, list[dict[str, Any]]] = {}
    total_chars = 0
    assistant_turns = 0
    empty_flags = 0
    risk_flags = 0
    convo_seen: set[str] = set()
    pii_hits: Counter[str] = Counter()
    for split_name in SPLIT_NAMES:
        rows: list[dict[str, Any]] = []
        with (out_dir / f"{split_name}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        emitted[split_name] = rows
        if len(rows) != census["rows"][split_name]:
            issues.append(f"emitted row count drift in {split_name}")
        for row in rows:
            row_issues, chars, turns, empty, risk = _row_scan(row, split_name, convo_seen, pii_hits)
            issues.extend(row_issues)
            total_chars += chars
            assistant_turns += turns
            empty_flags += empty
            risk_flags += risk
    if pii_hits:
        issues.append(f"PII screen hits in emitted payload: {dict(pii_hits)}")
    return issues, emitted, total_chars, assistant_turns, empty_flags, risk_flags


def _manifest_entry_issue(
    entry: dict[str, Any],
    record: dict[str, Any],
    seen_sessions: set[tuple[str, int]],
    group_split: dict[str, str],
    arc_splits: dict[str, set[str]],
) -> str | None:
    """Lineage checks for one manifest entry; mutates the tracking sets."""
    prov = record["provenance"]
    session = (prov["arc_id"], prov["session_n"])
    if session in seen_sessions:
        return f"session emitted twice: {session}"
    seen_sessions.add(session)
    group_key = f"{entry['source_family']}::{entry['source_id']}"
    prior_split = group_split.setdefault(group_key, entry["split"])
    if prior_split != entry["split"]:
        return f"group {group_key} spans splits: {prior_split} / {entry['split']}"
    arc_splits.setdefault(prov["arc_id"], set()).add(entry["split"])
    if entry["arc_id"] != prov["arc_id"] or entry["session_n"] != prov["session_n"]:
        return f"manifest lineage mismatch at gold line {entry['gold_line_no']}"
    return None


def _validate_lineage(
    out_dir: Path,
    emitted: dict[str, list[dict[str, Any]]],
    gold_by_line: dict[int, dict[str, Any]],
    expected_rows: int = EXPECTED_SLICE_ROWS,
) -> list[str]:
    """Manifest fidelity: every session once, groups intact, content identical."""
    issues: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    with (out_dir / "manifest_internal.jsonl").open(encoding="utf-8") as f:
        for line in f:
            manifest_rows.append(json.loads(line))
    if len(manifest_rows) != expected_rows:
        issues.append(f"manifest row count mismatch: expected {expected_rows}, got {len(manifest_rows)}")
    seen_sessions: set[tuple[str, int]] = set()
    group_split: dict[str, str] = {}
    arc_splits: dict[str, set[str]] = {}
    pointers: dict[str, int] = dict.fromkeys(SPLIT_NAMES, 0)
    for entry in manifest_rows:
        record = gold_by_line.get(entry["gold_line_no"])
        if record is None:
            issues.append(f"manifest references unknown gold line {entry['gold_line_no']}")
            continue
        issue = _manifest_entry_issue(entry, record, seen_sessions, group_split, arc_splits)
        if issue:
            issues.append(issue)
        split_name = entry["split"]
        idx = pointers[split_name]
        pointers[split_name] = idx + 1
        if idx >= len(emitted[split_name]):
            issues.append(f"more manifest rows than emitted rows in {split_name}")
            continue
        emitted_pairs = [(m["role"], m["content"]) for m in emitted[split_name][idx]["messages"]]
        source_pairs = [(m["role"], m["content"]) for m in record["messages"]]
        if emitted_pairs != source_pairs:
            issues.append(f"content fidelity mismatch at gold line {entry['gold_line_no']}")
    for split_name in SPLIT_NAMES:
        if pointers[split_name] != len(emitted[split_name]):
            issues.append(f"unmatched emitted rows remain in {split_name}")
    if any(len(names) > 1 for names in arc_splits.values()):
        issues.append("an arc's sessions straddle splits")
    return issues


def _record_identity(out_dir: Path, facts: _EmissionFacts, issues: list[str]) -> None:
    """Write identity.json: frozen identity, emitted-file hashes, census, issues."""
    files_record: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        path = out_dir / f"{split_name}.jsonl"
        files_record[f"{split_name}.jsonl"] = {
            "rows": facts.census["rows"][split_name],
            "sha256": _sha256_file(path),
        }
    manifest_path = out_dir / "manifest_internal.jsonl"
    files_record["manifest_internal.jsonl"] = {
        "rows": EXPECTED_SLICE_ROWS,
        "sha256": _sha256_file(manifest_path),
    }
    identity_record = {
        "spec": "docs/training/fireworks-export-plan.md (2026-10-10)",
        "basis": "owner instruction 2026-10-10 'Begin working on those 4' - local emission only",
        "emitted_at_utc": datetime.now(UTC).isoformat(),
        "emitter": {"path": "training/export_fireworks_payload.py", "sha256": _sha256_file(Path(__file__).resolve())},
        "identity": {
            "gold_sha256": facts.gold_sha,
            "slice_sha256": facts.slice_sha,
            "manifest_sha256": facts.manifest_sha,
            "slice_rows": EXPECTED_SLICE_ROWS,
        },
        "files": files_record,
        "splits": {
            name: {
                "rows": facts.census["rows"][name],
                "groups": facts.census["groups"][name],
                "fallback_rows": facts.census["fallback_rows"][name],
            }
            for name in SPLIT_NAMES
        },
        "tokens_content_div4": facts.tokens,
        "assistant_turns": facts.assistant_turns,
        "risk_grammar_flags": facts.risk_flags,
        "empty_ledger_value_flags": facts.empty_value_flags,
        "gate_deviations": facts.deviation_issues,
        "validation_issues": issues,
    }
    _write_lines(out_dir / "identity.json", [json.dumps(identity_record, ensure_ascii=False, indent=2)])


def _prechecks(
    splits: dict[str, list[dict[str, Any]]],
    slice_rows: list[tuple[int, dict[str, Any]]],
    gold_sha: str,
    slice_sha: str,
    manifest_sha: str,
) -> tuple[_EmissionFacts, list[str]]:
    """Census + integrity gates + ledger prechecks vs the frozen plan (steps 3-4)."""
    census, census_issues = _split_census(splits)
    gates_ok, gate_issues = integrity_gates(splits)
    leakage_issues = [i for i in gate_issues if "disjoint" in i or "evaluation-in-train" in i]
    deviation_issues = [i for i in gate_issues if i not in leakage_issues]
    logger.info(
        "integrity gates: pass=%s; %d documented ratio/balance deviations (accepted per owner "
        "instruction 2026-10-10), %d leakage failures",
        gates_ok,
        len(deviation_issues),
        len(leakage_issues),
    )
    hard, empty_flags, risk_flags, assistant_turns = _ledger_precheck(slice_rows)
    issues = [*census_issues, *leakage_issues]
    if hard:
        issues.append(f"{hard} ledger parse/key-set violations in the slice")
    if sum(empty_flags.values()) != EXPECTED_EMPTY_VALUE_FLAGS:
        issues.append(
            f"empty ledger-value flag count changed: expected {EXPECTED_EMPTY_VALUE_FLAGS}, "
            f"got {sum(empty_flags.values())}"
        )
    if sum(risk_flags.values()) != EXPECTED_RISK_GRAMMAR_FLAGS:
        issues.append(
            f"risk-grammar flag count changed: expected {EXPECTED_RISK_GRAMMAR_FLAGS}, got {sum(risk_flags.values())}"
        )
    logger.info(
        "ledger: %d assistant turns, %d parse/key-set violations, %d empty-value flags across "
        "%d sessions, %d risk-grammar flags across %d sessions (documented deviations, ship-as-is)",
        assistant_turns,
        hard,
        sum(empty_flags.values()),
        len(empty_flags),
        sum(risk_flags.values()),
        len(risk_flags),
    )
    facts = _EmissionFacts(
        gold_sha=gold_sha,
        slice_sha=slice_sha,
        manifest_sha=manifest_sha,
        census=census,
        tokens=0,
        assistant_turns=assistant_turns,
        risk_flags=dict(risk_flags),
        empty_value_flags=dict(empty_flags),
        deviation_issues=deviation_issues,
    )
    return facts, issues


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the Fireworks pilot payload (local only).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run identity + census prechecks and report, but write nothing",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir: Path = args.out_dir

    # Step 1: identity re-check (plan section 8.1) — fail-closed before anything runs.
    logger.info("step 1: identity re-check against %s", GOLD_PATH)
    slice_rows, raw_lines, gold_sha = _load_gold_slice(GOLD_PATH)
    slice_sha, manifest_sha, identity_issues = _check_identity(raw_lines, gold_sha)
    if identity_issues:
        for issue in identity_issues:
            logger.error("identity: %s", issue)
        logger.error("gold changed since the plan was frozen — re-screen and update the plan first; aborting")
        return 1
    logger.info("identity OK: %d arc rows, gold sha256 %s…", len(raw_lines), gold_sha[:12])

    # Steps 2-4: D4 grouping, splitter assignment, census, gates, ledger prechecks.
    logger.info("step 2: seed-level grouping and split assignment")
    splits, line_split, group_cache = _assign_splits(slice_rows)
    logger.info("step 3-4: split census, integrity gates, ledger prechecks vs plan sections 4-5")
    facts, issues = _prechecks(splits, slice_rows, gold_sha, slice_sha, manifest_sha)

    if args.dry_run:
        for issue in issues:
            logger.error("dry run issue: %s", issue)
        logger.info("dry run complete: %d issues; no files written", len(issues))
        return 1 if issues else 0
    if issues:
        for issue in issues:
            logger.error("precheck: %s", issue)
        logger.error("prechecks failed; nothing written")
        return 1

    # Step 5: emit provider files + internal manifest (plan sections 2.2-2.4).
    logger.info("step 5: emitting payload files to %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_payload(out_dir, slice_rows, raw_lines, line_split, group_cache)

    # Step 6: validate the emission (plan section 8.3).
    logger.info("step 6: validating emitted files")
    gold_by_line = dict(slice_rows)
    structure_issues, emitted, total_chars, assistant_turns, empty_flags, risk_flags = _validate_structure(
        out_dir, facts.census
    )
    lineage_issues = _validate_lineage(out_dir, emitted, gold_by_line)
    tokens = round(total_chars / 4)
    if tokens != EXPECTED_TOKENS:
        issues.append(f"token recount mismatch: expected {EXPECTED_TOKENS}, got {tokens}")
    if empty_flags != EXPECTED_EMPTY_VALUE_FLAGS:
        issues.append(f"emitted empty-value flag mismatch: expected {EXPECTED_EMPTY_VALUE_FLAGS}, got {empty_flags}")
    if risk_flags != EXPECTED_RISK_GRAMMAR_FLAGS:
        issues.append(f"emitted risk-grammar flag mismatch: expected {EXPECTED_RISK_GRAMMAR_FLAGS}, got {risk_flags}")
    if assistant_turns != facts.assistant_turns:
        issues.append(f"assistant turn count drift: {facts.assistant_turns} -> {assistant_turns}")
    issues.extend(structure_issues)
    issues.extend(lineage_issues)
    logger.info(
        "validated: %d rows, %d assistant turns, %d tokens (chars/4)",
        sum(facts.census["rows"].values()),
        assistant_turns,
        tokens,
    )

    # Step 7: record identities, then stop at the upload gate.
    facts.tokens = tokens
    _record_identity(out_dir, facts, issues)
    if issues:
        for issue in issues:
            logger.error("validation: %s", issue)
        logger.error("emission FAILED with %d issues; artifacts left in %s for inspection", len(issues), out_dir)
        return 1
    logger.info(
        "emission complete: train %d / val %d / test %d rows, identities in %s",
        facts.census["rows"]["train"],
        facts.census["rows"]["val"],
        facts.census["rows"]["test"],
        out_dir / "identity.json",
    )
    logger.info("stopped at the upload gate — nothing leaves this environment")
    return 0


if __name__ == "__main__":
    sys.exit(main())
