"""Run-2 Adaption export: split the expanded gold arc corpus, validate, unroll.

Reuses the run-1 export machinery (deterministic stratified split, ledger
contract, PII screen, structure + lineage validation) from
``export_fireworks_payload`` without its frozen run-1 identity pins — the
corpus grew by design (§13.10-13.13), so this script reports a fresh census
and records new SHAs instead of asserting old ones.

Emissions into ``training/output/arc_corpus/export/adaption_run2_v1/``:
- ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` — dialogue rows
  (``{"messages": [...]}`` with 0/1 weights), same as the Fireworks export.
- ``manifest_internal.jsonl`` — per-session lineage (split, source group,
  writer/auditor, verdict, SHAs).
- ``adaption_train_unrolled.jsonl`` — the Adaption raw-mode upload file:
  one datum per assistant turn, cumulative prompt (all turns through the
  preceding user message, weights preserved), completion = the assistant
  turn verbatim with ledger intact (run-1 format, verified against the
  6,951-row file).
- ``identity_run2.json`` — fresh census + SHA pins for this snapshot.

val/test are held out locally (run-1 discipline); only the unrolled train
file is an upload candidate.

Run (from ai/):
  PYTHONPATH=.. uv run python -m training.export_adaption_run2 [--dry-run]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from training.export_fireworks_payload import (
    GOLD_PATH,
    SPLIT_NAMES,
    _assign_splits,
    _emit_payload,
    _ledger_precheck,
    _load_gold_slice,
    _sha256_file,
    _split_census,
    _validate_lineage,
    _validate_structure,
)

_OUT_DIR = Path(__file__).resolve().parents[1] / "training" / "output" / "arc_corpus" / "export" / "adaption_run2_v1"
RUN1_FLOOR_DATUMS = 10_000  # Adaption effective-row floor for Qwen3.6-35B-A3B


def _unroll_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """One datum per assistant turn: prompt = turns through preceding user, weights kept."""
    datums: list[dict] = []
    issues: list[str] = []
    for row in rows:
        msgs = row["messages"]
        for i, msg in enumerate(msgs):
            if msg["role"] != "assistant":
                continue
            if i == 0 or msgs[i - 1]["role"] != "user":
                issues.append(f"unroll alternation violation: assistant at index {i} not preceded by user")
                continue
            datums.append({"prompt": msgs[:i], "completion": [msg]})
    return datums, issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, default=GOLD_PATH)
    ap.add_argument("--out-dir", type=Path, default=_OUT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="validate and report; write nothing")
    args = ap.parse_args()

    slice_rows, raw_lines, gold_sha = _load_gold_slice(args.gold)
    print(f"gold sha256: {gold_sha}")
    print(f"arc-family rows: {len(slice_rows)}")

    violations, empty_flags, risk_flags, assistant_turns = _ledger_precheck(slice_rows)
    print(f"ledger precheck: assistant_turns={assistant_turns} hard_violations={violations} "
          f"empty_value_flags={len(empty_flags)} risk_grammar_flags={len(risk_flags)}")
    if violations:
        print("FAIL: ledger contract violations — corpus not exportable")
        return 1

    splits, line_split, group_cache = _assign_splits(slice_rows)
    # run-1 pins do not apply to the expanded corpus; report the fresh census only
    census, _ = _split_census(splits)
    for split_name in SPLIT_NAMES:
        print(f"split {split_name}: rows={census['rows'][split_name]} groups={census['groups'][split_name]}")
    difficulty_total = Counter()
    for split_name in SPLIT_NAMES:
        per_split = splits[split_name]
        for record in per_split:
            difficulty_total[str(record.get("difficulty") or "?")] += 1
    print(f"difficulty census (all splits): {dict(difficulty_total)}")

    if args.dry_run:
        print("dry run — nothing written")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _emit_payload(args.out_dir, slice_rows, raw_lines, line_split, group_cache)
    print(f"emitted dialogues + manifest -> {args.out_dir}")

    issues, emitted, total_chars, turns, empty2, risk2 = _validate_structure(args.out_dir, census)
    print(f"structure scan: chars={total_chars} assistant_turns={turns} "
          f"empty_flags={empty2} risk_flags={risk2}")
    issues += _validate_lineage(
        args.out_dir, emitted, dict(slice_rows),
        expected_rows=len(slice_rows),
    )

    # unroll the train dialogues into the Adaption raw upload format
    datums, unroll_issues = _unroll_rows(emitted["train"])
    issues += unroll_issues
    unrolled_path = args.out_dir / "adaption_train_unrolled.jsonl"
    with unrolled_path.open("w", encoding="utf-8") as f:
        for datum in datums:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")
        f.flush()
    print(f"unrolled train datums: {len(datums)} -> {unrolled_path}")
    if len(datums) < RUN1_FLOOR_DATUMS:
        print(f"WARNING: {len(datums)} datums is below the 10,000 Adaption floor for "
              "Qwen3.6-35B-A3B — recommend_hyperparams will not validate without augmentation")

    identity = {
        "snapshot": "adaption_run2_v1",
        "gold_sha256": gold_sha,
        "arc_rows": len(slice_rows),
        "assistant_turns_scanned": assistant_turns,
        "splits": {name: census["rows"][name] for name in SPLIT_NAMES},
        "train_unrolled_datums": len(datums),
        "total_content_chars": total_chars,
        "sha256": {
            "train.jsonl": _sha256_file(args.out_dir / "train.jsonl"),
            "val.jsonl": _sha256_file(args.out_dir / "val.jsonl"),
            "test.jsonl": _sha256_file(args.out_dir / "test.jsonl"),
            "adaption_train_unrolled.jsonl": _sha256_file(unrolled_path),
        },
        "issues": issues,
    }
    with (args.out_dir / "identity_run2.json").open("w", encoding="utf-8") as f:
        json.dump(identity, f, indent=1, ensure_ascii=False)
        f.write("\n")

    if issues:
        print(f"VALIDATION ISSUES ({len(issues)}):")
        for issue in issues[:20]:
            print(f"  - {issue}")
        return 1
    print("EXPORT VALID — identity_run2.json records the fresh pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
