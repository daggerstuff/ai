"""Merge a generation lane's checkpoint/records into the canonical files.

Multi-lane generation (ARC_ENV_FILE lane overrides, 2026-10-12) runs a
second writer lane with its own --output/--checkpoint files. This merges one
lane's outputs into the canonical files with canonical-wins semantics:

- checkpoint rows key on (arc_id, session n): canonical rows win; lane rows
  for absent keys are appended.
- records key on arc_id: canonical records win (audited records are
  authoritative — the auditor's accept verdicts must never be replaced by an
  un-audited lane regeneration); lane records fill arcs absent from
  canonical (e.g. revise arcs whose canonical record the auditor removed).

The merge is idempotent: re-running after the lane writes more rows appends
only the new ones.

Run after a lane finishes (from ai/):
  uv run python -m training.merge_lane_checkpoints --lane-prefix laneW
"""

import argparse
import json
import os
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output" / "arc_corpus"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lane-prefix", required=True,
                    help="lane file prefix, e.g. laneW (files <prefix>_sessions_checkpoint.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be appended without writing")
    args = ap.parse_args()

    lane_ckpt_path = OUT_DIR / f"{args.lane_prefix}_sessions_checkpoint.jsonl"
    lane_rec_path = OUT_DIR / f"{args.lane_prefix}_records.jsonl"
    ckpt_path = OUT_DIR / "sessions_checkpoint.jsonl"
    rec_path = OUT_DIR / "arc_records.jsonl"

    lane_ckpt = load_jsonl(lane_ckpt_path)
    lane_recs = load_jsonl(lane_rec_path)
    ckpt = load_jsonl(ckpt_path)
    recs = load_jsonl(rec_path)

    have_ckpt = {(r["arc_id"], int(r["n"])) for r in ckpt}
    have_recs = {r["arc_id"] for r in recs}
    new_ckpt = [r for r in lane_ckpt if (r["arc_id"], int(r["n"])) not in have_ckpt]
    new_recs = [r for r in lane_recs if r["arc_id"] not in have_recs]

    print(f"=== LANE MERGE ({args.lane_prefix}) ===")
    print(f"lane checkpoint rows: {len(lane_ckpt)} (new: {len(new_ckpt)})")
    print(f"lane records: {len(lane_recs)} (new arcs: {len(new_recs)})")
    if args.dry_run:
        print("dry run — nothing written")
        return
    if new_ckpt:
        append_rows(ckpt_path, new_ckpt)
    if new_recs:
        append_rows(rec_path, new_recs)
    print(f"appended {len(new_ckpt)} checkpoint rows, {len(new_recs)} records -> canonical")


if __name__ == "__main__":
    main()
