"""Arc-corpus DPO pair builder (handoff Part 9, Phase B pairing).

Builds ``{prompt, chosen, rejected}`` preference pairs from the arc
pipeline's own revision history. Pairs feed Stage-5 DPO training
(``stage5_safety``) via consolidate_edge_nightmare.py, which gates
prompt+chosen (cliche + quadit) and appends to ``MASTER_STAGE_5.jsonl``.
DPO pairs are preference data, never ChatML gold.

Modes:
  session   same plan, same session: a superseded transcript from a
            historical snapshot (``.pre_*``) vs the accepted transcript in
            the current checkpoint. These are audit-driven regeneration
            pairs - the rejected arm is a transcript K3 later flagged or
            that the revision cycle replaced.
  attempt   same plan, same session: a gate-failed writer attempt
            (``failed_attempts/<arc>_s<n>_a<k>.txt``) vs the accepted
            transcript. Defect labels come from the ``# gates:`` header.
  case      same seed case: an accepted arc vs a terminal-HR arc built from
            the same seed (variation passes). Pairs share a case brief and
            compare session 1 only (session 2+ carry-forward differs).

The brief (prompt) is reconstructed from the plan + the accepted side's
prior sessions (``build_session_prompt``), prefixed with the writer system
prompt. Corrective notes that shaped some attempts are not reconstructable;
metadata marks the brief as reconstructed.

Output: ``output/arc_corpus/arc_dpo_pairs.jsonl`` (append + fsync,
hash-deduped, idempotent).

Run (from ai/):
  /home/vivi/pixelated/.venv/bin/python training/build_arc_dpo.py \
      [--mode session|attempt|case|all] [--dry-run]
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRAIN_DIR = _HERE.parents[0]  # ai/training
_AI_DIR = _HERE.parents[1]  # ai
_REPO_DIR = _HERE.parents[2]  # pixelated
sys.path.insert(0, str(_AI_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_REPO_DIR / ".env", override=True)

from training.build_arc_plans import DEFAULT_SEED_FILES, _load_seeds  # noqa: E402
from training.generate_arc_corpus import SYSTEM_PROMPT, build_session_prompt  # noqa: E402

OUT_DIR = _TRAIN_DIR / "output" / "arc_corpus"
PLANS_DIR = _TRAIN_DIR / "arc_plans"
PAIRS_PATH = OUT_DIR / "arc_dpo_pairs.jsonl"
CHECKPOINT_PATH = OUT_DIR / "sessions_checkpoint.jsonl"
RECORDS_PATH = OUT_DIR / "arc_records.jsonl"
FAILED_ATTEMPTS_DIR = OUT_DIR / "failed_attempts"

_ATTEMPT_FILE_RE = re.compile(r"^(?P<arc>.+)_s(?P<n>\d+)_a(?P<att>\d+)\.txt$")
_VARIATION_SUFFIX_RE = re.compile(r":v\d+$")
_LENGTH_RATIO_MIN = 0.5
_LENGTH_RATIO_MAX = 2.0


# ---------------------------------------------------------------------------
# Loaders (last-wins, matching generator/auditor semantics)
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> list[dict]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _load_records(path: Path) -> dict[str, dict]:
    """arc_id -> latest record row (last-wins)."""
    latest: dict[str, dict] = {}
    for row in _iter_jsonl(path):
        arc_id = row.get("arc_id")
        if arc_id:
            latest[arc_id] = row
    return latest


def _load_checkpoint(path: Path) -> dict[tuple[str, int], dict]:
    """(arc_id, session_n) -> latest checkpoint row (last-wins)."""
    latest: dict[tuple[str, int], dict] = {}
    for row in _iter_jsonl(path):
        arc_id, n = row.get("arc_id"), row.get("session_n")
        if arc_id and isinstance(n, int):
            latest[(arc_id, n)] = row
    return latest


def _verdict(record: dict) -> str:
    audit = record.get("audit") or {}
    return str(audit.get("verdict") or "unknown")


def _flag_sessions(record: dict) -> dict[int, list[str]]:
    """session_n -> K3 flag categories for that session (may be empty)."""
    audit = record.get("audit") or {}
    by_session: dict[int, list[str]] = {}
    for flag in audit.get("flags_final") or []:
        if not isinstance(flag, dict):
            continue
        session = flag.get("session")
        if not isinstance(session, int):
            continue
        category = str(flag.get("category") or "unknown")
        by_session.setdefault(session, []).append(category)
    return by_session


def _load_plan(arc_id: str) -> dict | None:
    path = PLANS_DIR / f"{arc_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Rendering and hashing
# ---------------------------------------------------------------------------


def _render_session(turns: list) -> str | None:
    """Writer-format transcript: [C] / [T|THINK] {ledger} / [T] lines."""
    lines: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        role = t.get("role")
        content = str(t.get("content") or "").strip()
        if not content:
            return None
        if role == "client":
            lines.append(f"[C] {content}")
        elif role == "therapist":
            ledger = t.get("ledger")
            if isinstance(ledger, dict):
                lines.append("[T|THINK] " + json.dumps(ledger, ensure_ascii=False))
            lines.append(f"[T] {content}")
        else:
            return None
    return "\n".join(lines) if lines else None


def _dpo_hash(prompt: str, chosen: str, rejected: str) -> str:
    text = f"{prompt}{chosen}{rejected}"
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


def _length_ok(chosen: str, rejected: str) -> bool:
    if not rejected:
        return False
    ratio = len(rejected) / max(len(chosen), 1)
    return _LENGTH_RATIO_MIN <= ratio <= _LENGTH_RATIO_MAX


# Gate heads that emit mechanical-gate failure strings. Failure detail often
# contains ';' separators between hits, so fragmenting on ';' then taking the
# pre-colon text yields garbage (e.g. "['-4y']"); match known heads instead.
_GATE_HEAD_RE = re.compile(
    r"\b(marker_format|turn_count|ledger_parse|ledger_fields|ledger_count"
    r"|cliche_gate|calendar_dates|tl_drift|beat_content|ledger_traceability"
    r"|spoken_specificity)\b"
)


def _gate_heads(header: str) -> list[str]:
    heads: list[str] = []
    for fragment in header.split(";"):
        match = _GATE_HEAD_RE.search(fragment)
        if match and match.group(1) not in heads:
            heads.append(match.group(1))
    return heads or ["unknown"]


# ---------------------------------------------------------------------------
# Session briefs
# ---------------------------------------------------------------------------


def _session_brief(plan: dict, n: int, priors: list[dict], with_system: bool) -> str:
    brief = build_session_prompt(plan, n, priors, None)
    if not with_system:
        return brief
    return f"{SYSTEM_PROMPT}\n\n{brief}"


def _case_brief(seed: dict) -> str:
    demog = ", ".join(seed.get("demographic_tags") or []) or "none"
    return (
        "WRITE SESSION 1 OF A THERAPY ARC for the case below.\n\n"
        f"seed: {json.dumps(seed['seed'])}\n"
        f"family: {seed['family']}\n"
        f"case: {seed['case']}\n"
        f"- {seed['extra']}\n"
        f"diagnostic tag: {seed.get('diagnostic_tag') or 'none'}\n"
        f"linguistic style: {seed.get('linguistic_style') or 'none'}\n"
        f"demographic tags: {demog}\n\n"
        "SEED TRANSCRIPT (user=client, assistant=therapist):\n"
        f"{seed['transcript']}\n\n"
        "Write session 1 now: 8-20 client turns, alternating [C]/[T|THINK]/[T], "
        "starting with [C]. Every [T|THINK] block is one line of strict JSON "
        "with all nine ledger fields. Keep the case's central conflict."
    )


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------


def _session_pairs(
    records: dict[str, dict], current: dict[tuple[str, int], dict], snapshot_path: Path, with_system: bool
) -> list[dict]:
    """Superseded-snapshot transcript vs accepted current transcript."""
    accepted = {a for a, r in records.items() if _verdict(r) == "accept"}
    snapshot = _load_checkpoint(snapshot_path)
    priors_cache: dict[str, list[dict]] = {}
    pairs = []
    for (arc_id, n), row in sorted(snapshot.items()):
        if arc_id not in accepted or (arc_id, n) not in current:
            continue
        chosen_row = current[(arc_id, n)]
        chosen = _render_session(chosen_row.get("turns") or [])
        rejected = _render_session(row.get("turns") or [])
        if not chosen or not rejected or rejected == chosen:
            continue
        if not _length_ok(chosen, rejected):
            continue
        plan = _load_plan(arc_id)
        if plan is None:
            continue
        if arc_id not in priors_cache:
            priors_cache[arc_id] = [
                r for (_, sn), r in sorted((k, v) for k, v in current.items() if k[0] == arc_id) if sn < n
            ]
        priors_cache[arc_id] = [
            r
            for r in priors_cache[arc_id]
            if (r.get("n"), r.get("session_n")) and (r.get("n") or r.get("session_n")) < n
        ]
        prompt = _session_brief(plan, n, priors_cache[arc_id], with_system)
        defect = _flag_sessions(records[arc_id]).get(n, [])
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "mode": "session",
                    "arc_id": arc_id,
                    "session_n": n,
                    "defect": defect or ["superseded_by_revision"],
                    "rejected_source": snapshot_path.parent.name,
                    "brief_reconstructed": True,
                    "writer_model": chosen_row.get("writer_model"),
                },
            }
        )
    return pairs


def _attempt_pairs(records: dict[str, dict], current: dict[tuple[str, int], dict], with_system: bool) -> list[dict]:
    """Gate-failed writer attempt vs accepted current transcript."""
    accepted = {a for a, r in records.items() if _verdict(r) == "accept"}
    priors_cache: dict[tuple[str, int], list[dict]] = {}
    pairs = []
    for path in sorted(FAILED_ATTEMPTS_DIR.glob("*.txt")):
        match = _ATTEMPT_FILE_RE.match(path.name)
        if not match:
            continue
        arc_id = match.group("arc")
        n = int(match.group("n"))
        if arc_id not in accepted or (arc_id, n) not in current:
            continue
        text = path.read_text(encoding="utf-8")
        header, _, rejected = text.partition("\n")
        if not header.startswith("# gates:"):
            continue
        rejected = rejected.strip()
        if not rejected:
            continue
        chosen_row = current[(arc_id, n)]
        chosen = _render_session(chosen_row.get("turns") or [])
        if not chosen or rejected == chosen or not _length_ok(chosen, rejected):
            continue
        plan = _load_plan(arc_id)
        if plan is None:
            continue
        key = (arc_id, n)
        if key not in priors_cache:
            priors_cache[key] = [
                r for (_, sn), r in sorted((k, v) for k, v in current.items() if k[0] == arc_id) if sn < n
            ]
        prompt = _session_brief(plan, n, priors_cache[key], with_system)
        gates = _gate_heads(header[len("# gates:") :])
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "mode": "attempt",
                    "arc_id": arc_id,
                    "session_n": n,
                    "attempt": int(match.group("att")),
                    "defect": gates,
                    "rejected_source": f"failed_attempts/{path.name}",
                    "brief_reconstructed": True,
                    "writer_model": chosen_row.get("writer_model"),
                },
            }
        )
    return pairs


def _case_pairs(records: dict[str, dict], seed_map_path: Path) -> list[dict]:
    """Same-seed accepted-vs-HR session-1 pairs (variation passes)."""
    seeds = {s["key"]: s for s in _load_seeds(DEFAULT_SEED_FILES)}
    base_key: dict[str, str] = {}
    for row in _iter_jsonl(seed_map_path):
        arc_id, key = row.get("arc_id"), row.get("key")
        if arc_id and key:
            base_key[arc_id] = _VARIATION_SUFFIX_RE.sub("", key)
    accepted: dict[str, str] = {}
    hr: dict[str, str] = {}
    for arc_id, record in records.items():
        verdict = _verdict(record)
        key = base_key.get(arc_id)
        if not key or key not in seeds:
            continue
        if verdict == "accept":
            accepted.setdefault(key, arc_id)
        elif verdict == "hr":
            hr.setdefault(key, arc_id)
    current = _load_checkpoint(CHECKPOINT_PATH)
    pairs = []
    for key, hr_arc in sorted(hr.items()):
        acc_arc = accepted.get(key)
        if not acc_arc:
            continue
        chosen_row = current.get((acc_arc, 1))
        rejected_row = current.get((hr_arc, 1))
        if not chosen_row or not rejected_row:
            continue
        chosen = _render_session(chosen_row.get("turns") or [])
        rejected = _render_session(rejected_row.get("turns") or [])
        if not chosen or not rejected or rejected == chosen:
            continue
        if not _length_ok(chosen, rejected):
            continue
        prompt = f"{SYSTEM_PROMPT}\n\n{_case_brief(seeds[key])}"
        defect = _flag_sessions(records[hr_arc]).get(1, [])
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "mode": "case",
                    "arc_id": acc_arc,
                    "rejected_arc_id": hr_arc,
                    "session_n": 1,
                    "seed_key": key,
                    "defect": defect or ["k3_hr_terminal"],
                    "rejected_source": f"arc:{hr_arc}",
                    "brief_reconstructed": False,
                    "writer_model": chosen_row.get("writer_model"),
                },
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _existing_hashes(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    hashes = set()
    for row in _iter_jsonl(path):
        if isinstance(row.get("metadata"), dict):
            h = row.get("pair_hash") or ""
            if h:
                hashes.add(str(h))
    return hashes


def _write_pairs(pairs: list[dict], dry_run: bool) -> dict[str, int]:
    seen = _existing_hashes(PAIRS_PATH)
    summary = {"built": len(pairs), "emitted": 0, "duplicates": 0}
    fresh = []
    for pair in pairs:
        h = _dpo_hash(pair["prompt"], pair["chosen"], pair["rejected"])
        if h in seen:
            summary["duplicates"] += 1
            continue
        seen.add(h)
        summary["emitted"] += 1
        fresh.append({**pair, "pair_hash": h})
    if not dry_run and fresh:
        with PAIRS_PATH.open("a", encoding="utf-8") as f:
            for row in fresh:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build arc-corpus DPO preference pairs.")
    parser.add_argument("--mode", default="all", choices=["session", "attempt", "case", "all"])
    parser.add_argument(
        "--with-system",
        dest="with_system",
        action="store_true",
        default=True,
        help="prefix prompts with the writer system prompt (default)",
    )
    parser.add_argument("--no-system", dest="with_system", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="count pairs; no writes")
    args = parser.parse_args(argv)

    records = _load_records(RECORDS_PATH)
    current = _load_checkpoint(CHECKPOINT_PATH)
    pairs: list[dict] = []
    if args.mode in ("session", "all"):
        for snapshot_dir in sorted(p for p in OUT_DIR.glob(".pre*") if p.is_dir()):
            snap = snapshot_dir / "sessions_checkpoint.jsonl"
            if snap.is_file():
                pairs.extend(_session_pairs(records, current, snap, args.with_system))
    if args.mode in ("attempt", "all"):
        pairs.extend(_attempt_pairs(records, current, args.with_system))
    if args.mode in ("case", "all"):
        pairs.extend(_case_pairs(records, PLANS_DIR / "seed_map.jsonl"))

    summary = _write_pairs(pairs, args.dry_run)
    by_mode: dict[str, int] = {}
    for pair in pairs:
        mode = pair["metadata"]["mode"]
        by_mode[mode] = by_mode.get(mode, 0) + 1
    sys.stdout.write(
        f"dpo: built={summary['built']} emitted={summary['emitted']} "
        f"duplicates={summary['duplicates']} by_mode={by_mode} "
        f"dry_run={args.dry_run}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
